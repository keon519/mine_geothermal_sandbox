"""Main window: lays the widgets out and routes signals between them.

    left    SettingsPanel (what to simulate)
    centre  LayerSelector | tabs: "Maps" (plan view of the selected layer, section,
            cross-section) and "3-D" (VolumeView: heat cloud, slices, cut plane)
    bottom  TimeSeriesPanel (loop temperatures + probes, heat, effectiveness)
    top     playback toolbar: play/pause, step, speed, time scrubber, colour scale, export
    status  progress bar: build stages with the time left, then run progress with the
            simulation speed, elapsed time and ETA; Cancel stops a build

Clicking a view moves the other views' slice lines through that cell
(left button) or pins a temperature probe there (right button). The
simulation runs in its own process (ProcessWorker); this window only draws
what its History holds, so it stays responsive however big the model is.
"""
from __future__ import annotations

import math
import time

import numpy as np
from PySide6.QtCore import QSettings, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QComboBox, QDockWidget, QFileDialog, QLabel, QMainWindow, QMessageBox,
                               QProgressBar, QPushButton, QSlider, QSplitter, QTabWidget, QToolBar,
                               QVBoxLayout, QWidget)

from ..live import History
from ..live.session import BuildInfo
from ..mines import mine_statistics
from .charts import TimeSeriesPanel
from .layer_selector import LayerSelector
from .settings_panel import SettingsPanel
from .slice_view import SliceResampler, SliceView
from .theme import PROBE_COLORS, SERIES, apply_pyqtgraph_theme
from .volume_view import VolumeView
from .worker import ProcessWorker, SimulationWorker, TickInfo

SPEEDS = [1, 5, 10, 30, 90, 180, 365, math.inf]          # simulated days per second (inf: as fast as possible)
RENDER_S = 1 / 30                                         # redraw at most ~30 times a second while playing
LAG_PROBE_S = 0.05                                        # event-loop latency probe interval
BACKENDS = ("process", "thread", "inline")


def _clock(seconds: float | None) -> str:
    """Format a duration for the status bar: tenths of a second when short, m:ss or h:mm:ss when
    longer.
    """
    if seconds is None or not math.isfinite(seconds):
        return "–"
    seconds = max(0.0, seconds)
    if seconds < 10:
        return f"{seconds:.1f} s"
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class MainWindow(QMainWindow):
    """The live simulator's main window.

    It owns no physics. The settings panel produces a config, a worker builds and runs it
    (normally in another process), and the window draws whatever the run's History holds. It
    keeps track of which three slices are shown (the i, j, k indices), the pinned probes, and
    whether you're following the live edge of the run or replaying an earlier frame.

    Drawing is throttled: results can arrive faster than they can be drawn, so redraws are
    capped at about 30 a second and spaced out further if the event loop starts to lag.
    """
    requestBuild = Signal(object)
    requestPlay = Signal()
    requestPause = Signal()
    requestStep = Signal()
    requestSpeed = Signal(float)
    requestShutdown = Signal()

    def __init__(self, threaded: bool = True, backend: str | None = None):
        """backend: "process" (default when threaded), "thread", or "inline" (synchronous;
        the default when threaded=False, used by tests and screenshots)."""
        super().__init__()
        backend = backend or ("process" if threaded else "inline")
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        self.backend = backend
        apply_pyqtgraph_theme()
        self.setWindowTitle("Mine-water heat sink — live simulator")
        self.resize(1500, 950)
        self.info: BuildInfo | None = None
        self.history: History | None = None
        self.k = self.j = self.i = 0
        self.probes: dict[str, int] = {}
        self.follow_live = True
        self.frame_index = 0
        self.playing = False
        self._last_tick: TickInfo | None = None
        self._last_render = 0.0
        self._render_pending = False
        self._build_progress = None            # (BuildProgress, perf_counter when received)
        self._build_t0 = None
        self._run_wall = 0.0                   # wall seconds spent playing this run
        self._play_since = None
        self._prefs = QSettings("mine_sandbox", "live_simulator")

        #  widgets
        self.settings = SettingsPanel()
        dock = QDockWidget("Settings", self)
        dock.setWidget(self.settings)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        self.layers = LayerSelector()
        self.plan = SliceView("Plan view of the selected layer",
                              "Left-click: move the section lines here · Right-click: add/remove a probe · "
                              "Wheel: zoom", colorbar=True)
        self.section = SliceView("Section along x")
        self.cross = SliceView("Cross-section")
        self.charts = TimeSeriesPanel()
        self.volume = VolumeView()
        self._last_3d = 0.0
        self.banner = QLabel("")
        self.banner.setVisible(False)
        self.banner.setWordWrap(True)

        left = QWidget(); lv = QVBoxLayout(left); lv.setContentsMargins(0, 0, 0, 0)
        lv.addWidget(QLabel("<b>Layers</b> (click to view)"))
        lv.addWidget(self.layers)
        right = QSplitter(Qt.Vertical); right.addWidget(self.section); right.addWidget(self.cross)
        maps = QSplitter(Qt.Horizontal)
        maps.addWidget(self.plan); maps.addWidget(right)
        maps.setStretchFactor(0, 3); maps.setStretchFactor(1, 2)
        self.tabs = QTabWidget()
        self.tabs.addTab(maps, "Maps")
        self.tabs.addTab(self.volume, "3-D view")
        self.tabs.currentChanged.connect(self._tab_changed)
        top = QSplitter(Qt.Horizontal)
        top.addWidget(left); top.addWidget(self.tabs)
        top.setStretchFactor(0, 0); top.setStretchFactor(1, 1)
        top.setSizes([260, 1200])
        centre = QSplitter(Qt.Vertical)
        centre.addWidget(top); centre.addWidget(self.charts)
        centre.setSizes([620, 300])
        wrap = QWidget(); wv = QVBoxLayout(wrap); wv.setContentsMargins(4, 4, 4, 4)
        wv.addWidget(self.banner); wv.addWidget(centre)
        self.setCentralWidget(wrap)
        self._build_toolbar()
        self.status = QLabel("Choose settings and press Build model.")
        self.hover = QLabel("")
        self.statusBar().addWidget(self.status, 1)
        self.statusBar().addPermanentWidget(self.hover)
        self._build_progress_widgets()

        #worker: its own process (default), a thread, or inline
        self.thread = None
        if backend == "process":
            self.worker = ProcessWorker()
        else:
            self.worker = SimulationWorker()
            if backend == "thread":
                self.thread = QThread(self)
                self.worker.moveToThread(self.thread)
                self.thread.started.connect(self.worker.start)
        try:
            self.worker.speed_factor = float(self._prefs.value("build/speed_factor", 1.0))
        except (TypeError, ValueError):
            pass
        self.requestBuild.connect(self.worker.build)
        self.requestPlay.connect(self.worker.play)
        self.requestPause.connect(self.worker.pause)
        self.requestStep.connect(self.worker.step_once)
        self.requestSpeed.connect(self.worker.set_speed)
        if self.thread is not None:
            self.requestShutdown.connect(self.worker.shutdown, Qt.BlockingQueuedConnection)
        else:
            self.requestShutdown.connect(self.worker.shutdown)
        self.worker.built.connect(self._on_built)
        self.worker.failed.connect(self._on_failed)
        self.worker.busy.connect(self._on_busy)
        self.worker.ticked.connect(self._on_tick)
        self.worker.progress.connect(self._on_progress)
        self.worker.devicesFound.connect(self.settings.set_devices)
        self.worker.notice.connect(lambda text: self._show_banner(text, "#eda100"))
        self.worker.runFailed.connect(self._on_run_failed)
        if self.thread:
            self.thread.start()
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render)
        # keep the window responsive on slow machines: watch how late the event loop runs
        # and space redraws out when it falls behind (back to ~30/s once it keeps up)
        self._render_interval = RENDER_S
        self._lag_ema = 0.0
        self._lag_last = time.perf_counter()
        self._lag_timer = QTimer(self)
        self._lag_timer.setInterval(int(1000 * LAG_PROBE_S))
        self._lag_timer.timeout.connect(self._measure_lag)
        self._lag_timer.start()

        # routing
        self.settings.buildRequested.connect(self._build)
        self.layers.layerSelected.connect(self._set_layer)
        self.plan.cellClicked.connect(lambda n, b: self._clicked(n, b, "plan"))
        self.section.cellClicked.connect(lambda n, b: self._clicked(n, b, "section"))
        self.cross.cellClicked.connect(lambda n, b: self._clicked(n, b, "cross"))
        for v in (self.plan, self.section, self.cross):
            v.cellHovered.connect(self._hovered)
        self.charts.seekRequested.connect(self._seek_time)
        self.volume.slicesChanged.connect(self._slices_from_3d)
        self.requestSpeed.emit(float(self.speed.currentData()))

    #toolbar
    def _build_toolbar(self):
        """Playback toolbar: play/pause, step, speed, time scrubber, colour scale mode and exports."""
        tb = QToolBar("Playback"); tb.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, tb)
        self.act_play = QAction("▶  Play", self)
        self.act_play.setShortcut(QKeySequence(Qt.Key_Space))
        self.act_play.triggered.connect(self._toggle_play)
        self.act_step = QAction("Step ▸", self)
        self.act_step.setShortcut(QKeySequence(Qt.Key_Right))
        self.act_step.triggered.connect(lambda: self.requestStep.emit())
        tb.addAction(self.act_play); tb.addAction(self.act_step)
        tb.addSeparator()
        tb.addWidget(QLabel(" Speed "))
        self.speed = QComboBox()
        for s in SPEEDS:
            self.speed.addItem("max (as fast as the solver goes)" if math.isinf(s) else f"{s} days/s", s)
        self.speed.setToolTip("Simulated days per second. 'max' runs flat out; the status bar shows "
                              "the speed actually reached.")
        self.speed.setCurrentIndex(SPEEDS.index(30))
        self.speed.currentIndexChanged.connect(lambda _: self.requestSpeed.emit(float(self.speed.currentData())))
        tb.addWidget(self.speed)
        tb.addSeparator()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimumWidth(300)
        self.slider.setRange(0, 0)
        self.slider.sliderMoved.connect(self._scrub)
        tb.addWidget(self.slider)
        self.time_label = QLabel(" day 0 ")
        self.time_label.setMinimumWidth(150)
        tb.addWidget(self.time_label)
        self.live_btn = QPushButton("● Live")
        self.live_btn.clicked.connect(self._go_live)
        tb.addWidget(self.live_btn)
        tb.addSeparator()
        tb.addWidget(QLabel(" Colours "))
        self.scale_mode = QComboBox()
        self.scale_mode.addItem("rock → hottest so far", "fixed")
        self.scale_mode.addItem("this frame only", "auto")
        self.scale_mode.currentIndexChanged.connect(lambda _: self._render())
        tb.addWidget(self.scale_mode)
        tb.addSeparator()
        act_png = QAction("Save picture…", self); act_png.triggered.connect(lambda: self._save_png())
        act_csv = QAction("Export CSV…", self); act_csv.triggered.connect(lambda: self._export_csv())
        act_vtk = QAction("Export ParaView…", self); act_vtk.triggered.connect(lambda: self._export_paraview())
        act_vtk.setToolTip("Write every stored frame as .vtr files plus a .pvd time series for ParaView")
        tb.addAction(act_png); tb.addAction(act_csv); tb.addAction(act_vtk)

    #  build
    def _build(self, config):
        """Start a build with the settings panel's config and switch the progress bar to build mode."""
        self._config = config
        self._build_t0 = time.perf_counter()
        self._build_progress = None
        self._set_progress_mode("build")
        self.requestBuild.emit(config)

    def _on_busy(self, text):
        """Show a status message while the worker gets going."""
        self.status.setText(text)
        self.banner.setVisible(False)

    def _on_failed(self, msg):
        """The build failed or was cancelled: say so, with a red banner for real failures."""
        self._set_progress_mode("idle")
        if msg == "build cancelled":
            self.status.setText("Build cancelled.")
            return
        self._show_banner(f"Could not build the model — {msg}", "#e34948")
        self.status.setText("Build failed.")

    def _on_run_failed(self, msg):
        """The run hit an error while stepping: stop and show it."""
        self.playing = False
        self.act_play.setText("▶  Play")
        self._show_banner(f"The run stopped — {msg}", "#e34948")

    #  progress bar / times
    def _build_progress_widgets(self):
        """Status-bar progress bar, timing label and Cancel button."""
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setFixedWidth(300)
        self.progress.setTextVisible(True)
        self.progress.setVisible(False)
        self.timing = QLabel("")
        self.timing.setMinimumWidth(260)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setToolTip("Stop this build (the simulation process is restarted)")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.clicked.connect(self._cancel_build)
        sb = self.statusBar()
        sb.addPermanentWidget(self.progress)
        sb.addPermanentWidget(self.timing)
        sb.addPermanentWidget(self.cancel_btn)
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(100)
        self._progress_timer.timeout.connect(self._refresh_build_progress)
        self._progress_mode = "idle"

    def _set_progress_mode(self, mode):
        """Switch the progress bar between idle, building and running."""
        self._progress_mode = mode
        self.progress.setVisible(mode != "idle")
        self.cancel_btn.setVisible(mode == "build" and self.backend == "process")
        if mode == "build":
            self.progress.setValue(0)
            self.progress.setFormat("Building… %p%")
            self.timing.setText("starting…")
            self._progress_timer.start()
        else:
            self._progress_timer.stop()
        if mode == "idle":
            self.timing.setText("")

    def _cancel_build(self):
        """Ask the worker to abandon the current build."""
        self.worker.cancel_build()

    def _on_progress(self, p):
        """A build stage started: remember the report and refresh the bar."""
        self._build_progress = (p, time.perf_counter())
        self._refresh_build_progress()
        if self.backend == "inline":          # nothing else repaints during a synchronous build
            self.progress.repaint()
            self.timing.repaint()

    def _refresh_build_progress(self):
        """Update the build bar and time left, extrapolating from the last report so the bar keeps
        moving between stages.
        """
        if self._progress_mode != "build":
            return
        elapsed = time.perf_counter() - (self._build_t0 or time.perf_counter())
        if self._build_progress is None:
            self.timing.setText(f"elapsed {_clock(elapsed)}")
            return
        p, at = self._build_progress
        since = time.perf_counter() - at
        self.progress.setValue(int(1000 * p.fraction(since)))
        self.progress.setFormat(f"{p.text} — %p%")
        eta = p.eta_s(since)
        left = "almost done" if eta is not None and eta < 0.05 else (
            f"~{_clock(eta)} left" if eta is not None else "taking longer than expected")
        self.timing.setText(f"elapsed {_clock(elapsed)} · {left}")

    def _update_run_progress(self, tick: TickInfo):
        """Update the run bar and timing text: speed reached, time per step, elapsed time, ETA, and
        whether the solver is the bottleneck.
        """
        total = tick.sim_days or self.sim_days
        frac = 0.0 if not total else min(1.0, tick.t_d / total)
        self.progress.setValue(int(1000 * frac))
        self.progress.setFormat(f"day {tick.t_d:,.0f} / {total:,.0f}  (%p%)")
        now = time.perf_counter()
        wall = self._run_wall + (now - self._play_since if self._play_since is not None else 0.0)
        parts = []
        if tick.playing:
            if tick.days_per_s > 0:
                parts.append(f"{tick.days_per_s:,.0f} d/s")
            parts.append(f"{tick.step_ms:.1f} ms/step")
            parts.append(f"elapsed {_clock(wall)}")
            if tick.days_per_s > 0:
                parts.append(f"ETA {_clock((total - tick.t_d) / tick.days_per_s)}")
            want = float(self.speed.currentData())
            if math.isfinite(want) and tick.days_per_s > 0 and tick.days_per_s < 0.85 * want and wall > 2:
                parts.append("solver-limited")
        elif tick.done:
            parts.append(f"finished · {_clock(wall)} of simulation" if wall > 0 else "finished")
        else:
            parts.append(f"paused · {tick.step_ms:.1f} ms/step" if tick.n_steps else
                         f"built in {_clock(self.info.build_seconds)} · press Play")
        self.timing.setText(" · ".join(parts))

    def _on_built(self, info: BuildInfo, history: History):
        """A new model is ready: set up the layer list, slices, charts and 3-D view for it.

        The starting slices pass through the middle of the main water route, and the panel under
        Build shows the model's headline numbers (cells, water volume, wall area, flush time)
        and which device is solving it.
        """
        self.info, self.history = info, history
        self._run_wall, self._play_since = 0.0, None
        self._set_progress_mode("run")
        try:
            self._prefs.setValue("build/speed_factor", float(info.speed_factor))
        except Exception:  # noqa: BLE001 - preferences are best-effort
            pass
        g = info.problem.graph
        lat = g.lattice
        self.rocks = info.problem.scenario.ground.geology.rocks_at(g.centroids)
        rock_names = np.array([r.name for r in self.rocks], dtype=object)
        self.T_rock = info.problem.scenario.ground.T_rock_C
        self.sim_days = info.problem.scenario.numerics.sim_days
        self.layer_nodes = [np.unique(lat.layer(k)[lat.layer(k) >= 0]) for k in range(lat.shape[2])]
        water = g.is_drift
        water_layers = {k for k in range(lat.shape[2]) if water[self.layer_nodes[k]].any()}
        rock_labels = []
        for k in range(lat.shape[2]):
            nodes = self.layer_nodes[k]
            rock_labels.append(", ".join(sorted(set(rock_names[nodes[~water[nodes]]].tolist()))))
        mid = int(g.path_cells[len(g.path_cells) // 2])
        self.i, self.j, self.k = lat.voxel_of(mid)
        self.layers.set_layers(lat.z_edges, rock_labels, water_layers,
                               radial=not g.resolves_depth, select=self.k)
        self.probes.clear()
        self.charts.reset(self.T_rock)
        self.follow_live, self.frame_index = True, 0
        self.banner.setVisible(False)
        # set the 3-D view's new grid first: _update_slices indexes into it
        self.volume.set_problem(g, self.T_rock, (self.i, self.j, self.k))
        self._update_slices()
        st = mine_statistics(g)
        lines = [f"<b>{g.n_cells:,}</b> cells, <b>{st['water_cells']:,}</b> water",
                 f"water volume {st['water_volume_m3']:,.0f} m³ · wall area {st['wall_area_m2']:,.0f} m²",
                 f"main route {st['route_length_m']:,.0f} m · flushes in {info.problem.residence_time_d * 24:,.1f} h"]
        if "open_area_m2" in st and g.resolves_depth:
            lines.insert(1, f"open area {st['open_area_m2']:,.0f} m²")
        lines.append(f"<b>compute:</b> {info.compute_label}")
        lines.extend(f"<i>{n}</i>" for n in info.notices)
        if info.stage_seconds:
            from ..live.progress import STAGE_LABELS
            slow = sorted(info.stage_seconds.items(), key=lambda kv: -kv[1])[:3]
            lines.append(f"built in {info.build_seconds:.2f} s — slowest: " + ", ".join(
                f"{STAGE_LABELS.get(k, k).lower()} {v:.2f} s" for k, v in slow))
        self.settings.show_stats("<br>".join(lines))
        self.status.setText(f"{info.engine_label} · built in {info.build_seconds:.1f} s · {info.summary}")
        gpu_asked = getattr(getattr(self, "_config", None), "device", "auto") not in ("auto", "cpu")
        if info.notices and gpu_asked:
            self._show_banner(" ".join(info.notices), "#eda100")
        self.progress.setValue(0)
        self.progress.setFormat(f"day 0 / {self.sim_days:,.0f}  (%p%)")
        self.timing.setText(f"built in {_clock(info.build_seconds)} · press Play")
        self._render()

    #  slices
    def _update_slices(self):
        """Point the three map views (and the 3-D slices) at the current i, j, k, with guide lines
        where they cross.
        """
        g = self.info.problem.graph
        lat = g.lattice
        xl, yl, zl = lat.axis_labels
        x_e, y_e, z_e = lat.x_edges, lat.y_edges, lat.z_edges
        zc, xc, yc = lat.centres(z_e), lat.centres(x_e), lat.centres(y_e)
        water = g.is_drift
        if g.resolves_depth:
            plan_title = f"Plan view — layer {zc[self.k]:+.1f} m ({z_e[self.k]:+.1f} to {z_e[self.k+1]:+.1f})"
        else:   # axisymmetric: a "layer" is a ring around the drift
            plan_title = (f"Ring {z_e[self.k]:.1f}–{z_e[self.k+1]:.1f} m from the drift axis, "
                          "unrolled along the drift")
        self.plan.set_slice(SliceResampler(lat.layer(self.k), x_e, y_e), water, xl, yl,
                            title=plan_title, lock_aspect=None if g.resolves_depth else False)
        self.section.set_slice(SliceResampler(lat.row(self.j), x_e, z_e), water, xl, zl,
                               title=f"Section at y = {yc[self.j]:.1f} m", lock_aspect=False)
        self.cross.set_slice(SliceResampler(lat.column(self.i), y_e, z_e), water, yl, zl,
                             title=f"Cross-section at x = {xc[self.i]:.1f} m",
                             lock_aspect=None if g.resolves_depth else False)
        self.volume.set_slices(self.i, self.j, self.k)
        self.plan.set_guides(a=xc[self.i], b=yc[self.j])
        self.section.set_guides(a=xc[self.i], b=zc[self.k])
        self.cross.set_guides(a=yc[self.j], b=zc[self.k])
        self._update_markers()

    def _update_markers(self):
        """Draw the pumps and probes on whichever views currently pass through them."""
        g = self.info.problem.graph
        lat = g.lattice
        xc, yc, zc = (lat.centres(e) for e in (lat.x_edges, lat.y_edges, lat.z_edges))
        pts = {"plan": [], "section": [], "cross": []}
        def add(node, brush, symbol, size=12):
            i, j, k = lat.voxel_of(node)
            spot = dict(brush=brush, symbol=symbol, size=size, pen="#0b0b0b")
            if k == self.k:
                pts["plan"].append(dict(pos=(xc[i], yc[j]), **spot))
            if j == self.j:
                pts["section"].append(dict(pos=(xc[i], zc[k]), **spot))
            if i == self.i:
                pts["cross"].append(dict(pos=(yc[j], zc[k]), **spot))
        add(g.inlet_cell, SERIES[0], "t1", 14)      # Pump In: triangle
        add(g.outlet_cell, SERIES[1], "t", 14)      # Pump Out: inverted triangle
        for idx, node in enumerate(self.probes.values()):
            add(node, PROBE_COLORS[idx % len(PROBE_COLORS)], "o", 10)
        self.plan.set_markers(pts["plan"])
        self.section.set_markers(pts["section"])
        self.cross.set_markers(pts["cross"])

    def _slices_from_3d(self, i, j, k):
        """The 3-D view's sliders moved the slices: follow them in the maps and layer list."""
        if self.info is None:
            return
        layer_changed = k != self.k
        self.i, self.j, self.k = i, j, k
        if layer_changed:
            self.layers.blockSignals(True)
            self.layers.select_layer(k)
            self.layers.blockSignals(False)
        self._update_slices()
        self._render()

    def _tab_changed(self, index):
        """Create the 3-D viewport the first time its tab is opened, then draw it."""
        if self.tabs.widget(index) is self.volume and self.volume.activate():
            self._render(force_3d=True)

    def _set_layer(self, k):
        """Show a different layer in the plan view."""
        if self.info is None or k == self.k:
            return
        self.k = k
        self._update_slices()
        self._render()

    def _clicked(self, node, button, view):
        """Left click moves the other slices through the clicked cell; right click toggles a probe
        there.
        """
        if self.info is None:
            return
        if button == Qt.RightButton:
            self._toggle_probe(node)
            return
        i, j, k = self.info.problem.graph.lattice.voxel_of(node)
        if view == "plan":
            self.i, self.j = i, j
        elif view == "section":
            self.i, self.k = i, k
            self.layers.select_layer(k)
        else:
            self.j, self.k = j, k
            self.layers.select_layer(k)
        self._update_slices()
        self._render()

    def _toggle_probe(self, node):
        """Pin a temperature probe on a cell, or remove it if one is already there (up to six at once)."""
        for name, n in list(self.probes.items()):
            if n == node:
                del self.probes[name]
                self.charts.remove_probe(name)
                break
        else:
            if len(self.probes) >= len(PROBE_COLORS):
                self._show_banner(f"At most {len(PROBE_COLORS)} probes — right-click one to remove it.", "#eda100")
                return
            c = self.info.problem.graph.centroids[node]
            self.probes[f"probe {len(self.probes) + 1} ({c[0]:.0f}, {c[1]:.0f}, {c[2]:+.1f})"] = node
        self._rename_probes()
        self._update_markers()
        self._render()

    def _rename_probes(self):
        """Drop all probe curves so they're redrawn with fresh names and colours on the next render."""
        for name in list(self.charts.probe_curves):
            self.charts.remove_probe(name)

    def _hovered(self, node):
        """Show the position, material and temperature of the cell under the mouse."""
        if self.info is None or node < 0:
            self.hover.setText("")
            return
        g = self.info.problem.graph
        field = self._field()
        x, y, z = g.centroids[node]
        what = "water" if g.is_drift[node] else self.rocks[node].name
        T = f"{field[node]:.2f} °C" if field is not None else ""
        self.hover.setText(f"({x:.1f}, {y:.1f}, {z:+.2f}) m · {what} · {T}")

    #  playback
    def _toggle_play(self):
        """Play from the live edge, or pause."""
        if self.info is None or self._progress_mode == "build":
            return
        if self.playing:
            self.requestPause.emit()
        else:
            self._go_live()
            self.requestPlay.emit()

    def _on_tick(self, tick: TickInfo):
        """Update playback state, the scrubber, banners and status from the worker's latest report,
        then schedule a redraw.
        """
        now = time.perf_counter()
        if tick.playing and self._play_since is None:
            self._play_since = now
        elif not tick.playing and self._play_since is not None:
            self._run_wall += now - self._play_since
            self._play_since = None
        was_done = self._last_tick is not None and self._last_tick.done
        self._last_tick = tick
        self.playing = tick.playing
        self.act_play.setText("⏸  Pause" if tick.playing else "▶  Play")
        if self.history is None or self.info is None:
            return
        n = self.history.n_frames
        self.slider.setRange(0, max(0, n - 1))
        if self.follow_live:
            self.slider.setValue(n - 1)
        if tick.runaway:
            self._show_banner(f"Loop water passed the {self.settings.T_max.value():.0f} °C limit on day "
                              f"{tick.t_d:.1f}: this mine cannot reject this heat load. Run stopped.", "#e34948")
        elif tick.done and not was_done:
            self._show_banner(f"Run complete: {tick.t_d:.0f} days simulated in {_clock(self._run_wall)}.",
                              "#1baf7a")
        label = tick.compute_label.split(" · ")[0] if tick.compute_label else ""
        self.status.setText(f"{self.info.engine_label} · {label + ' · ' if label else ''}"
                            f"{self.info.problem.graph.n_cells:,} cells · "
                            f"{tick.step_ms:.1f} ms per step · {tick.n_steps:,} steps")
        self._update_run_progress(tick)
        self._request_render()

    def _measure_lag(self):
        """Watch how late the event loop is running and adjust the redraw rate to match.

        A timer that should fire every 50 ms is checked for how late it actually fires. If the
        lag builds up, redraws are spaced out so mouse and keyboard stay responsive; once it
        settles, the rate creeps back to about 30 a second.
        """
        now = time.perf_counter()
        lag = max(0.0, now - self._lag_last - LAG_PROBE_S)
        self._lag_last = now
        self._lag_ema = 0.7 * self._lag_ema + 0.3 * lag
        if self._lag_ema > 0.015:                                   # input would start to feel sticky
            self._render_interval = min(0.5, self._render_interval * 1.3)
        elif self._lag_ema < 0.005:
            self._render_interval = max(RENDER_S, self._render_interval * 0.9)

    def _request_render(self):
        """Redraw now, or once the frame budget allows (ticks can arrive faster than we draw)."""
        wait = self._render_interval - (time.perf_counter() - self._last_render)
        if wait <= 0 or self.backend == "inline":
            self._render_timer.stop()
            self._render()
        elif not self._render_timer.isActive():
            self._render_timer.start(int(1000 * wait) + 1)

    def _scrub(self, value):
        """Show a stored frame instead of the live edge."""
        self.follow_live = False
        self.frame_index = value
        self._render()

    def _seek_time(self, t):
        """Replay the stored frame closest to a given day (from a chart click)."""
        if self.history is None:
            return
        times = self.history.frame_times()
        if len(times):
            self._scrub(int(np.argmin(np.abs(times - t))))
            self.slider.setValue(self.frame_index)

    def _go_live(self):
        """Go back to following the live edge of the run."""
        self.follow_live = True
        self._render()

    #  drawing
    def _field(self):
        """The temperature field being shown: the latest one when live, otherwise the replayed frame."""
        if self.history is None:
            return None
        if self.follow_live:
            return self.history.latest()
        return self.history.frame(self.frame_index)[1]

    def _shown_time(self):
        """The simulated day being shown."""
        if self.follow_live:
            s = self.history.scalars()["t_d"]
            return float(s[-1]) if len(s) else 0.0
        return self.history.frame(self.frame_index)[0]

    def _render(self, force_3d: bool = False):
        """Draw the current field everywhere: maps, 3-D view, layer list, charts and probes.

        The colour scale runs either from rock temperature to the hottest seen so far (so
        colours mean the same thing across frames) or just across this frame. The 3-D view is
        the most expensive part, so while playing it's refreshed only a few times a second.
        """
        if self.info is None or self.history is None:
            return
        self._last_render = time.perf_counter()
        field = self._field()
        if field is None:
            return
        s = self.history.scalars()
        if self.scale_mode.currentData() == "auto":
            lo, hi = float(np.min(field)), float(np.max(field))
        else:
            hottest = max(float(np.max(field)), float(np.max(s["T_in_C"])) if len(s["T_in_C"]) else 0.0)
            lo, hi = self.T_rock, max(hottest, self.T_rock + 1.0)
        levels = (lo, max(hi, lo + 1e-3))
        for v in (self.plan, self.section, self.cross):
            v.set_values(field, levels)
        # the 3-D view costs more to draw: refresh it at most ~6 times a second while playing
        if self.tabs.currentWidget() is self.volume and self.volume.plotter is not None:
            now = time.perf_counter()
            if force_3d or not self.playing or now - self._last_3d > 0.16:
                self.volume.set_field(field, levels)
                self._last_3d = now
        rise = np.array([float(np.max(field[n])) - self.T_rock if len(n) else 0.0 for n in self.layer_nodes])
        self.layers.update_rise(rise, max(levels[1] - self.T_rock, 1e-3))
        self.charts.update_series(s)
        for idx, (name, node) in enumerate(self.probes.items()):
            t, T = self.history.probe(node)
            self.charts.set_probe(name, PROBE_COLORS[idx % len(PROBE_COLORS)], t, T)
        t_shown = self._shown_time()
        self.charts.set_cursor(t_shown)
        mode = "live" if self.follow_live else "replay"
        self.time_label.setText(f" day {t_shown:,.1f} / {self.sim_days:,.0f}  ({mode}) ")
        self.live_btn.setEnabled(not self.follow_live)

    def _show_banner(self, text, color):
        """Show a coloured message across the top of the window."""
        self.banner.setText(text)
        self.banner.setStyleSheet(f"background: {color}; color: white; padding: 6px; font-weight: bold;")
        self.banner.setVisible(True)

    #  export
    def _save_png(self, path: str | None = None):
        """Save a picture of the whole window."""
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Save picture", "mine_sim.png", "PNG (*.png)")
        if path:
            self.grab().save(path)

    def _export_csv(self, path: str | None = None):
        """Export the loop series and probes as CSV."""
        if self.history is None:
            return
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Export run", "mine_sim.csv", "CSV (*.csv)")
        if path:
            self.history.to_csv(path, probes=self.probes)

    def _export_paraview(self, folder: str | None = None):
        """Export every stored frame as a ParaView time series, if PyVista is installed and the
        grid is 3-D.
        """
        if self.history is None:
            return
        try:
            from ..viz3d import Not3DError, VolumeData, export_paraview_series, history_frames
        except ImportError:
            QMessageBox.information(self, "ParaView export", "Exporting needs PyVista:  pip install pyvista")
            return
        if not folder:
            folder = QFileDialog.getExistingDirectory(self, "Folder for the ParaView time series")
        if not folder:
            return
        try:
            vd = VolumeData(self.info.problem.graph, self.T_rock)
        except Not3DError as e:
            QMessageBox.information(self, "ParaView export", str(e))
            return
        pvd = export_paraview_series(vd, history_frames(self.history), folder,
                                     name=self.info.problem.scenario.name)
        self.status.setText(f"Wrote {self.history.n_frames} frames: {pvd}")
        return pvd

    #  test/screenshot helpers
    def run_steps(self, n: int) -> None:
        """Synchronous stepping (inline windows only)."""
        if self.backend != "inline":
            raise RuntimeError("run_steps is for inline (threaded=False) windows")
        if self.worker.session:
            self.worker.session.step(n)
            self.worker._emit()

    def closeEvent(self, ev):
        """Stop timers, shut the worker down and release the plots before the window goes away."""
        self._render_timer.stop()
        self._progress_timer.stop()
        self._lag_timer.stop()
        self.requestShutdown.emit()
        for v in (self.charts, self.plan, self.section, self.cross, self.volume):
            v.dispose()
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait(3000)
        super().closeEvent(ev)
