"""3-D view: the warmed rock as a translucent heat cloud you can turn, slice and cut.

    Heat cloud   volume rendering of the temperature field. Undisturbed rock is
                 invisible; opacity grows with the temperature rise.
    Workings     surface of the water-filled cells (blue), Pump In / Out markers.
    Slices       three flat sheets (plan layer, section, cross-section) that follow
                 the Maps tab and the sliders here.
    Cut plane    a plane you drag and tilt; it shows the exact cross-section on its
                 face, and optionally hides everything on one side so the cut face is
                 visible inside the cloud.

The VTK viewport is created on first use (`activate`), so the rest of the app
does not depend on PyVista or on an OpenGL display.
"""
from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel,
                               QPushButton, QSlider, QVBoxLayout, QWidget)

from .theme import INK2

BACKGROUND = "#3b3b3b"
CMAPS = {"plasma": "plasma (blue → yellow)", "theme": "heat (as the maps)",
         "turbo": "turbo (rainbow)", "viridis": "viridis"}


def _theme_cmap():
    """The maps' heat colour scale as a matplotlib colormap, so the 3-D view can match the 2-D maps."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("heat1", ["#fdf3ea", "#f5b27a", "#d95926", "#7a2a0c"])


def view_unavailable_reason() -> str | None:
    """Why the 3-D viewport can't be created here, or None if it can."""
    from importlib.util import find_spec
    if find_spec("pyvista") is None or find_spec("pyvistaqt") is None:
        return "The 3-D view needs PyVista:  pip install pyvista pyvistaqt"
    if QGuiApplication.platformName() in ("offscreen", "minimal"):
        return "The 3-D view needs an OpenGL display (not available on the offscreen platform)."
    return None


def _slider(lo=0, hi=100):
    """A horizontal slider with a sensible minimum width."""
    s = QSlider(Qt.Horizontal)
    s.setRange(lo, hi)
    s.setMinimumWidth(90)
    return s


class VolumeView(QWidget):
    """The 3-D tab: a heat cloud, the workings, three slices and a cut plane you can move.

    It's created empty and only builds a VTK viewport when the tab is first shown, so the rest
    of the app works without PyVista or an OpenGL display. Moving the slice sliders emits
    `slicesChanged` so the 2-D maps stay in step with the 3-D view.
    """
    slicesChanged = Signal(int, int, int)        # i, j, k picked with the sliders here

    def __init__(self):
        super().__init__()
        self.plotter = None
        self.data = None
        self._field = None
        self._levels = None
        self._ijk = (0, 0, 0)
        self._actors: dict = {}
        self._sheets: dict = {}
        self._clip_plane = None
        self._cut_true = None
        self.graph = None
        self.T_rock = 0.0

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        row1, row2 = QHBoxLayout(), QHBoxLayout()
        self.chk_cloud = QCheckBox("Heat cloud"); self.chk_cloud.setChecked(True)
        self.chk_workings = QCheckBox("Workings"); self.chk_workings.setChecked(True)
        self.chk_slices = QCheckBox("Slices"); self.chk_slices.setChecked(False)
        self.chk_cut = QCheckBox("Cut plane")
        self.chk_cutaway = QCheckBox("cut away front"); self.chk_cutaway.setChecked(True)
        self.opacity = _slider(); self.opacity.setValue(75)
        self.exag = QDoubleSpinBox(); self.exag.setRange(1, 20); self.exag.setSingleStep(0.5)
        self.exag.setSuffix(" ×"); self.exag.setKeyboardTracking(False)
        self.cmap = QComboBox()
        for key, label in CMAPS.items():
            self.cmap.addItem(label, key)
        self.reset_btn = QPushButton("Reset view")
        self.save_btn = QPushButton("Save image…")
        for w in (self.chk_cloud, self.chk_workings, self.chk_slices, self.chk_cut, self.chk_cutaway):
            row1.addWidget(w)
        row1.addSpacing(12)
        row1.addWidget(QLabel("Opacity")); row1.addWidget(self.opacity)
        row1.addWidget(QLabel("Vertical ×")); row1.addWidget(self.exag)
        row1.addWidget(QLabel("Colours")); row1.addWidget(self.cmap)
        row1.addStretch(1)
        row1.addWidget(self.reset_btn); row1.addWidget(self.save_btn)
        self.sx, self.sy, self.sz = _slider(), _slider(), _slider()
        row2.addWidget(QLabel("Slices:  x")); row2.addWidget(self.sx)
        row2.addWidget(QLabel("y")); row2.addWidget(self.sy)
        row2.addWidget(QLabel("layer")); row2.addWidget(self.sz)
        self.hint = QLabel("Drag: rotate · Shift-drag: pan · Wheel: zoom · "
                           "Cut plane: drag the arrow to tilt it, the plane to move it")
        self.hint.setStyleSheet(f"color: {INK2};")
        row2.addWidget(self.hint, 1)
        lay.addLayout(row1)
        lay.addLayout(row2)
        self.placeholder = QLabel("Build a model to see it in 3-D.")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setStyleSheet(f"background: {BACKGROUND}; color: #dddddd; font-size: 12pt;")
        lay.addWidget(self.placeholder, 1)
        self._body = lay

        for chk in (self.chk_cloud, self.chk_workings, self.chk_slices):
            chk.toggled.connect(self._apply_visibility)
        self.chk_cut.toggled.connect(self._toggle_cut)
        self.chk_cutaway.toggled.connect(self._update_clip)
        self.opacity.sliderReleased.connect(self._rebuild_cloud)
        self.cmap.currentIndexChanged.connect(self._restyle)
        self.exag.valueChanged.connect(self._apply_scale)
        self.reset_btn.clicked.connect(self.reset_camera)
        self.save_btn.clicked.connect(lambda: self.save_image())
        for s in (self.sx, self.sy, self.sz):
            s.valueChanged.connect(self._sliders_moved)

    #  lifecycle
    def activate(self) -> bool:
        """Create the VTK viewport (first time the tab is shown). False if impossible."""
        if self.plotter is not None:
            return True
        reason = view_unavailable_reason()
        if reason:
            self.placeholder.setText(reason)
            return False
        os.environ.setdefault("QT_API", "pyside6")
        from pyvistaqt import QtInteractor
        self.plotter = QtInteractor(self)
        self.plotter.set_background(BACKGROUND)
        # (no depth peeling: on some OpenGL drivers it drops translucent surfaces entirely)
        self._body.addWidget(self.plotter.interactor, 1)
        if self.data is not None:
            self.placeholder.hide()
            self._build_scene(reset_camera=True)
        else:
            self.plotter.interactor.hide()          # keep the "build a model" hint until there is one
        return True

    def dispose(self) -> None:
        """Close the VTK viewport."""
        if self.plotter is not None:
            self.plotter.close()
            self.plotter = None

    #  model
    def set_problem(self, graph, T_rock_C: float, ijk: tuple[int, int, int]) -> None:
        """Show a new model: set up the datasets, slider ranges and a fitting vertical
        exaggeration, or explain why this grid can't be shown in 3-D.
        """
        from ..viz3d import supports_3d
        self.data, self._field, self._levels, self._cut_true = None, None, None, None
        self.graph, self.T_rock = graph, T_rock_C
        if not supports_3d(graph):
            self.placeholder.setText("The 3-D view needs a grid with real 3-D geometry: the 3-D block "
                                     "grid or a mine file. Radial and 2-D slab grids have no plan width.")
            self.placeholder.show()
            if self.plotter is not None:
                self.plotter.clear()
                self.plotter.interactor.hide()
            return
        nx, ny, nz = graph.lattice.shape
        for sl, n in ((self.sx, nx), (self.sy, ny), (self.sz, nz)):
            sl.blockSignals(True); sl.setRange(0, n - 1); sl.blockSignals(False)
        from ..viz3d import VolumeData
        self.exag.blockSignals(True)
        self.exag.setValue(VolumeData.default_exaggeration(graph))
        self.exag.blockSignals(False)
        self._make_data()
        self.set_slices(*ijk, emit=False, render=False)
        if self.plotter is not None:
            self.plotter.interactor.show()
            self.placeholder.hide()
            self._build_scene(reset_camera=True)

    def _make_data(self):
        """Rebuild the 3-D datasets at the current vertical exaggeration, carrying over the latest
        temperatures.
        """
        from ..viz3d import VolumeData
        self.data = VolumeData(self.graph, self.T_rock, z_scale=self.exag.value())
        if self._field is not None:
            self.data.update(self._field)

    def set_field(self, field: np.ndarray, levels: tuple[float, float]) -> None:
        """New temperatures. Cheap unless the colour range moved a lot."""
        self._field = field
        if self.data is None:
            return
        self.data.update(field)
        for sheet in self._sheets.values():
            self.data.fill_sheet(sheet, field)
        if self.plotter is None:
            self._levels = levels
            return
        old = self._levels
        if old is None or abs(levels[1] - old[1]) > 0.05 * max(old[1] - old[0], 1e-6):
            self._levels = levels
            self._restyle()                      # colour range changed: re-add coloured actors
            return
        if self._cut_true is not None:
            self._recut()
        self.plotter.render()

    def set_slices(self, i: int, j: int, k: int, emit: bool = False, render: bool = True) -> None:
        """Move the three slices to the given lattice indices and redraw them."""
        self._ijk = (i, j, k)
        for sl, v in ((self.sx, i), (self.sy, j), (self.sz, k)):
            sl.blockSignals(True); sl.setValue(v); sl.blockSignals(False)
        if self.data is None:
            return
        for axis, idx in (("x", i), ("y", j), ("z", k)):
            self._sheets[axis] = self.data.axis_slice(axis, idx)
            if self._field is not None:
                self.data.fill_sheet(self._sheets[axis], self._field)
        if self.plotter is not None and self._levels is not None:
            self._add_sheets()
            if render:
                self.plotter.render()
        if emit:
            self.slicesChanged.emit(i, j, k)

    #  scene
    def _cmap(self):
        """The selected colour map (a name, or the app's own heat map)."""
        key = self.cmap.currentData()
        return _theme_cmap() if key == "theme" else key

    def _opacity_curve(self):
        """Opacity for the heat cloud: undisturbed rock is invisible, and the cloud thickens as the
        rock warms.
        """
        strength = self.opacity.value() / 100.0
        r = np.linspace(0.0, 1.0, 64)
        return np.clip((r - 0.04) / 0.96, 0, 1) ** 1.2 * strength      # undisturbed rock: invisible

    @staticmethod
    def _surface_opacity():
        """Slices and the cut face: undisturbed rock as translucent 'glass', anything warmed
        by 30%+ of the range opaque, so the heat stands out and the model behind shows."""
        r = np.linspace(0.0, 1.0, 64)
        return 0.4 + 0.6 * np.clip((r - 0.02) / 0.3, 0, 1) ** 0.8

    def _build_scene(self, reset_camera: bool = False):
        """Rebuild everything in the viewport, keeping the camera unless asked to reset it."""
        p = self.plotter
        cam = None if reset_camera else p.camera_position
        p.clear_plane_widgets()
        p.clear()
        self._actors.clear()
        if self._levels is None:
            self._levels = (self.T_rock, self.T_rock + 1.0)
        self._add_cloud()
        if self.data.workings.n_cells:
            self._actors["workings"] = p.add_mesh(self.data.workings, color="#4da6ff", opacity=0.3,
                                                  name="workings", pickable=False)
        self._add_markers()
        self._add_sheets()
        self._add_axes()
        if self.chk_cut.isChecked():
            self._start_cut_widget()
        self._apply_visibility(render=False)
        if cam is None:
            self.reset_camera()
        else:
            p.camera_position = cam
            p.render()

    def _add_cloud(self):
        """Add the volume-rendered heat cloud."""
        d = self.data
        lo, hi = self._levels
        self._actors["cloud"] = self.plotter.add_volume(
            d.image, scalars="T", clim=(lo, hi), cmap=self._cmap(), opacity=self._opacity_curve(),
            opacity_unit_distance=3 * float(d.sample_spacing.mean()), name="cloud",
            show_scalar_bar=False)
        self._update_clip()

    def _add_sheets(self):
        """The three slices. The colour bar hangs off the plan slice, which is opaque
        (a bar taken from the translucent cloud would show washed-out colours)."""
        lo, hi = self._levels
        for axis, sheet in self._sheets.items():
            bar = axis == "z"
            self._actors[f"sheet_{axis}"] = self.plotter.add_mesh(
                sheet, scalars="T", clim=(lo, hi), cmap=self._cmap(), name=f"sheet_{axis}",
                opacity=self._surface_opacity(), pickable=False, show_scalar_bar=bar,
                scalar_bar_args=dict(title="Temperature (°C)", color="white", vertical=True,
                                     position_x=0.9, position_y=0.15, height=0.7, width=0.05,
                                     fmt="%.1f", n_labels=5) if bar else None)
        self._apply_visibility(render=False)

    def _add_markers(self):
        """Mark Pump In and Pump Out with labelled spheres."""
        import pyvista as pv
        g, d = self.graph, self.data
        x0, x1, y0, y1 = d.bounds[:4]
        size = 0.012 * max(x1 - x0, y1 - y0)
        for name, cell, color, text in (("pump_in", g.inlet_cell, "#eb6834", "Pump In"),
                                        ("pump_out", g.outlet_cell, "#2a78d6", "Pump Out")):
            c = d.to_display(g.centroids[cell])
            self._actors[name] = self.plotter.add_mesh(pv.Sphere(radius=size, center=c), color=color,
                                                       name=name, pickable=False)
            self.plotter.add_point_labels([c + [0, 0, 2.5 * size]], [text], name=name + "_label",
                                          text_color="white", font_size=12, shape=None,
                                          show_points=False, always_visible=True)

    def _add_axes(self):
        """Draw axes labelled in true metres, even though the display is vertically stretched."""
        d, labels = self.data, self.graph.lattice.axis_labels
        self.plotter.show_bounds(bounds=d.bounds, axes_ranges=list(d.true_bounds), grid="back",
                                 location="outer", ticks="outside", color="white", font_size=11,
                                 xtitle=labels[0], ytitle=labels[1],
                                 ztitle=f"{labels[2]}  (vertical ×{d.z_scale:g})",
                                 n_xlabels=5, n_ylabels=5, n_zlabels=5, fmt="%.0f")

    def _restyle(self, *_):
        """Colour map / range changed: re-add the coloured actors."""
        if self.plotter is None or self.data is None:
            return
        for name in ("cloud", "cut", "sheet_x", "sheet_y", "sheet_z"):
            if name in self._actors:
                self.plotter.remove_actor(self._actors.pop(name), render=False)
        self._add_cloud()
        self._add_sheets()
        if self._cut_true is not None:
            self._recut()
        self._apply_visibility(render=True)

    def _rebuild_cloud(self):
        """Re-add the heat cloud after the opacity slider is released."""
        if self.plotter is not None and self.data is not None:
            self.plotter.remove_actor(self._actors.pop("cloud", None), render=False)
            self._add_cloud()
            self._apply_visibility(render=True)

    def _apply_visibility(self, *_, render=True):
        """Show or hide the cloud, workings and slices according to the checkboxes."""
        a = self._actors
        if "cloud" in a:
            a["cloud"].SetVisibility(self.chk_cloud.isChecked())
        if "workings" in a:
            a["workings"].SetVisibility(self.chk_workings.isChecked())
        for axis in "xyz":
            if f"sheet_{axis}" in a:
                a[f"sheet_{axis}"].SetVisibility(self.chk_slices.isChecked())
        if render and self.plotter is not None:
            self.plotter.render()

    def _apply_scale(self, *_):
        """New vertical exaggeration: rebuild the stretched datasets and the scene."""
        if self.data is None:
            return
        self._make_data()
        self.set_slices(*self._ijk, render=False)
        if self.plotter is not None:
            self._build_scene(reset_camera=True)

    def reset_camera(self):
        """Return to the default slightly-from-below isometric view, framed on the model."""
        if self.plotter is None:
            return
        p = self.plotter
        p.view_isometric()
        p.camera.azimuth = -20
        p.camera.elevation = -6
        p.reset_camera()
        x0, x1, y0, y1, z0, z1 = self.data.bounds if self.data is not None else (0, 1, 0, 1, 0, 1)
        tall = (z1 - z0) > 0.9 * max(x1 - x0, y1 - y0)
        p.camera.zoom(1.0 if tall else 1.15)
        p.render()

    #  cut plane
    # The cut is stored in true coordinates (metres) so it survives a change of exaggeration.
    def _toggle_cut(self, on):
        """Turn the cut plane on (starting as a near-vertical plane through the middle) or off."""
        if self.plotter is None or self.data is None:
            return
        if not on:
            self.plotter.clear_plane_widgets()
            self._cut_true = None
            if "cut" in self._actors:
                self.plotter.remove_actor(self._actors.pop("cut"), render=False)
            self._update_clip()
            return
        if self._cut_true is None:
            x0, x1, y0, y1 = self.data.true_bounds[:4]
            self._cut_true = (np.array([1.0, 0.35, 0.0]) / np.hypot(1.0, 0.35),
                              np.array([(x0 + x1) / 2, (y0 + y1) / 2, 0.0]))
        self._start_cut_widget()

    def _start_cut_widget(self):
        """Place the draggable plane widget at the stored cut and draw the cut face."""
        if self._cut_true is None:
            return
        normal, origin = self._display_cut()
        self.plotter.clear_plane_widgets()
        self.plotter.add_plane_widget(self._plane_moved, normal=tuple(normal), origin=tuple(origin),
                                      bounds=self.data.bounds, factor=1.0, outline_translation=False,
                                      implicit=True, normal_rotation=True, color="white", tubing=False,
                                      interaction_event="always", outline_opacity=0.0)
        self._recut()
        self._update_clip()

    def _display_cut(self):
        """The stored cut plane converted from true metres to stretched display coordinates.

        Points just scale with the vertical exaggeration, but a plane's normal has to scale the
        opposite way to stay perpendicular to the stretched plane.
        """
        n, o = self._cut_true
        zs = self.data.z_scale
        nd = n / np.array([1.0, 1.0, zs])
        return nd / np.linalg.norm(nd), o * np.array([1.0, 1.0, zs])

    def _plane_moved(self, normal, origin):
        """The user moved the plane widget: store the new cut in true metres and redraw it."""
        zs = self.data.z_scale
        n = np.array(normal, float) * np.array([1.0, 1.0, zs])
        self._cut_true = (n / np.linalg.norm(n), np.array(origin, float) / np.array([1.0, 1.0, zs]))
        self._recut()
        self._update_clip()

    def _recut(self):
        """Draw the exact cross-section on the current cut plane."""
        if self.plotter is None or self._cut_true is None:
            return
        normal, origin = self._display_cut()
        mesh = self.data.cut(normal, origin)
        if mesh.n_cells == 0:
            return
        lo, hi = self._levels
        self._actors["cut"] = self.plotter.add_mesh(mesh, scalars="T", clim=(lo, hi), cmap=self._cmap(),
                                                    opacity=self._surface_opacity(), name="cut",
                                                    show_scalar_bar=False, pickable=False)

    def _update_clip(self, *_):
        """Hide the cloud and workings in front of the cut plane so the cut face shows."""
        if self.plotter is None:
            return
        # import only the module needed: `import vtk` loads every VTK DLL, and Windows
        # Smart App Control may block some of them (e.g. vtkFiltersGeneric)
        from vtkmodules.vtkCommonDataModel import vtkPlane
        plane = None
        if self._cut_true is not None and self.chk_cutaway.isChecked():
            normal, origin = self._display_cut()
            plane = vtkPlane()
            plane.SetNormal(*(-normal))
            plane.SetOrigin(*origin)
        for name in ("cloud", "workings"):
            actor = self._actors.get(name)
            if actor is None:
                continue
            actor.mapper.RemoveAllClippingPlanes()
            if plane is not None:
                actor.mapper.AddClippingPlane(plane)
        self.plotter.render()

    #  sliders / output
    def _sliders_moved(self, *_):
        """The slice sliders moved: update the slices and tell the maps."""
        self.set_slices(self.sx.value(), self.sy.value(), self.sz.value(), emit=True)

    def save_image(self, path: str | None = None):
        """Save the current 3-D view as a PNG, asking for a path if none is given."""
        if self.plotter is None:
            return
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Save 3-D image", "mine_3d.png", "PNG (*.png)")
        if path:
            self.plotter.screenshot(path)
