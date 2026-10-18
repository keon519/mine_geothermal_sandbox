"""A heat map of one slice through the model (plan layer, section, or cross-section).

The lattice can have uneven cell sizes (thin layers near the workings,
growing margins), so a slice is resampled once onto a uniform pixel grid at
the finest cell size; every frame is then a single fancy-index lookup.
Water cells get an outline so the workings are always visible.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from .theme import HEAT_CMAP, INK2, WATER_OUTLINE

MAX_PX = 900


class SliceResampler:
    """Maps a slice of uneven cells onto an even pixel grid, once, so each frame is a single
    lookup.

    Model cells range from thin layers at the workings to wide margin cells far away, but images
    need square pixels. Pixels are sized to the finest cell (capped at MAX_PX across) and each
    pixel remembers which model cell it falls in. Drawing a new frame then just gathers each
    pixel's temperature through that table.
    """

    def __init__(self, ids: np.ndarray, a_edges: np.ndarray, b_edges: np.ndarray):
        self.ids, self.a_edges, self.b_edges = ids, a_edges, b_edges
        self.ia = self._pixels(a_edges)
        self.ib = self._pixels(b_edges)
        self.pixel_ids = ids[self.ia][:, self.ib]                 # (na_px, nb_px)
        self.valid = self.pixel_ids >= 0
        a0, a1, b0, b1 = a_edges[0], a_edges[-1], b_edges[0], b_edges[-1]
        self.rect = QRectF(a0, b0, a1 - a0, b1 - b0)

    @staticmethod
    def _pixels(edges):
        """For each pixel along one axis, the index of the (uneven) cell its centre falls in."""
        span = edges[-1] - edges[0]
        res = max(np.diff(edges).min(), span / MAX_PX)
        n = max(1, int(np.ceil(span / res)))
        centres = edges[0] + (np.arange(n) + 0.5) * span / n
        return np.clip(np.searchsorted(edges, centres) - 1, 0, len(edges) - 2)

    def image(self, values: np.ndarray) -> np.ndarray:
        """(nb_px, na_px) row-major image, NaN where there is no cell."""
        img = np.where(self.valid, values[np.maximum(self.pixel_ids, 0)], np.nan)
        return img.T

    def node_at(self, a: float, b: float) -> int:
        """Model cell under a point in slice coordinates, or -1 outside the slice."""
        i = np.searchsorted(self.a_edges, a) - 1
        j = np.searchsorted(self.b_edges, b) - 1
        if 0 <= i < self.ids.shape[0] and 0 <= j < self.ids.shape[1]:
            return int(self.ids[i, j])
        return -1


class SliceView(QWidget):
    """One heat-map panel showing a slice through the model, with the workings outlined in blue.

    It also draws pump markers and dashed guide lines where the other two slices cross it. Left
    click emits cellClicked(node, Qt.LeftButton) and right click emits cellClicked(node,
    Qt.RightButton); hovering emits cellHovered with the cell under the mouse.
    """
    cellClicked = Signal(int, object)
    cellHovered = Signal(int)

    def __init__(self, title: str, hint: str = "", colorbar: bool = False):
        super().__init__()
        self.plot = pg.PlotWidget()
        self.plot.setTitle(title, color=INK2, size="10pt")
        self.plot.showGrid(x=False, y=False)
        self.plot.setMenuEnabled(False)
        self.img = pg.ImageItem()
        self.img.setLookupTable(HEAT_CMAP.getLookupTable(nPts=256))
        self.plot.addItem(self.img)
        self.bar = None
        if colorbar:
            self.bar = pg.ColorBarItem(values=(0, 1), colorMap=HEAT_CMAP, interactive=False,
                                       label="temperature [°C]", width=14)
            self.bar.setImageItem(self.img, insert_in=self.plot.getPlotItem())
        self.outline = pg.IsocurveItem(level=0.5, pen=pg.mkPen(WATER_OUTLINE, width=1.5))
        self.outline.setParentItem(self.img)
        self.markers = pg.ScatterPlotItem(pxMode=True)
        self.plot.addItem(self.markers)
        self.guide_a = pg.InfiniteLine(angle=90, pen=pg.mkPen(INK2, style=Qt.DashLine))
        self.guide_b = pg.InfiniteLine(angle=0, pen=pg.mkPen(INK2, style=Qt.DashLine))
        for g in (self.guide_a, self.guide_b):
            g.setVisible(False)
            self.plot.addItem(g)
        self.resampler: SliceResampler | None = None
        self._levels = (0.0, 1.0)
        self._needs_rect = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.plot)
        if hint:
            h = QLabel(hint)
            h.setStyleSheet(f"color: {INK2}; font-size: 9pt;")
            lay.addWidget(h)
        self.plot.scene().sigMouseClicked.connect(self._clicked)
        self.plot.scene().sigMouseMoved.connect(self._moved)

    #  geometry
    def set_slice(self, resampler: SliceResampler, water: np.ndarray, a_label: str, b_label: str,
                  title: str | None = None, lock_aspect: bool | None = None) -> None:
        """Switch to a new slice: new pixel lookup, new water outline, new axis labels.

        The aspect ratio is locked only when the slice is reasonably square, so long thin
        sections still fill the panel. The zoom is kept if only the slice position changed.
        """
        prev = None if self.resampler is None else self.resampler.rect
        self.resampler = resampler
        self._needs_rect = True                       # applied once the new image is set
        mask = np.where(resampler.valid, water[np.maximum(resampler.pixel_ids, 0)], False)
        self.outline.setData(mask.T.astype(float))
        self.plot.setLabel("bottom", a_label)
        self.plot.setLabel("left", b_label)
        if title:
            self.plot.setTitle(title, color=INK2, size="10pt")
        r = resampler.rect
        if lock_aspect is None:
            ratio = r.width() / max(r.height(), 1e-9)
            lock_aspect = 0.25 < ratio < 4
        self.plot.setAspectLocked(lock_aspect)
        if prev is None or prev != r:                 # keep the user's zoom when only the slice moves
            self.plot.setRange(rect=r, padding=0.02)

    def set_values(self, values: np.ndarray, levels: tuple[float, float]) -> None:
        """Draw a temperature field with the given colour levels."""
        if self.resampler is None:
            return
        self._levels = levels
        self.img.setImage(self.resampler.image(values), autoLevels=False, levels=levels)
        if self.bar is not None:
            self.bar.setLevels(levels)
        if self._needs_rect:                          # setRect scales by the current image size,
            self.img.setRect(self.resampler.rect)     # so it must follow the first setImage
            self._needs_rect = False

    def set_markers(self, spots: list[dict]) -> None:
        """Show pump and probe markers (pyqtgraph scatter spot dicts)."""
        self.markers.setData(spots)

    def set_guides(self, a: float | None = None, b: float | None = None) -> None:
        """Show dashed lines where the other slices cross this one (None hides a line)."""
        self.guide_a.setVisible(a is not None)
        self.guide_b.setVisible(b is not None)
        if a is not None:
            self.guide_a.setPos(a)
        if b is not None:
            self.guide_b.setPos(b)

    def dispose(self) -> None:
        """See TimeSeriesPanel.dispose."""
        self.plot.scene().sigMouseClicked.disconnect(self._clicked)
        self.plot.scene().sigMouseMoved.disconnect(self._moved)
        self.plot.clear()

    #  mouse
    def _view_pos(self, scene_pos):
        """Convert a scene position to plot coordinates, or None if it's outside the plot area."""
        vb = self.plot.getPlotItem().vb
        if not vb.sceneBoundingRect().contains(scene_pos):
            return None
        return vb.mapSceneToView(scene_pos)

    def _clicked(self, ev):
        """Report a click on a model cell, with which mouse button was used."""
        if self.resampler is None:
            return
        p = self._view_pos(ev.scenePos())
        if p is None:
            return
        node = self.resampler.node_at(p.x(), p.y())
        if node >= 0:
            self.cellClicked.emit(node, ev.button())

    def _moved(self, scene_pos):
        """Report which cell the mouse is over."""
        if self.resampler is None:
            return
        p = self._view_pos(scene_pos)
        if p is not None:
            self.cellHovered.emit(self.resampler.node_at(p.x(), p.y()))
