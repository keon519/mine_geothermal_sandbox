"""Settings panel: edits a LiveConfig. Knows nothing about running or drawing.

Emits buildRequested(LiveConfig) when the user presses Build. Mine files are
opened, generated and saved here (they are part of the configuration).
"""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea, QSpinBox,
                               QStackedWidget, QVBoxLayout, QWidget)

from ..compute import CUPY_HINT
from ..engines import ENGINES
from ..live import DRIFT_GRIDS, SURFACE_MODES, LiveConfig
from ..materials import ROCKS
from ..mines import MineFileError, MineSpec, generate_mine, load_mine, save_mine
from ..mines.generate import SIZES
from .theme import INK2

MINE_FILTER = "Mine files (*.json *.yaml *.yml);;All files (*)"


def _spin(lo, hi, value, step=1.0, decimals=1, suffix=""):
    """A decimal spin box with range, step, precision and unit, which only reports a change once
    editing is done.
    """
    s = QDoubleSpinBox()
    s.setRange(lo, hi); s.setValue(value); s.setSingleStep(step); s.setDecimals(decimals)
    if suffix:
        s.setSuffix(" " + suffix)
    s.setKeyboardTracking(False)
    return s


class SettingsPanel(QScrollArea):
    """The left-hand panel where the run is set up.

    Its widgets map one-to-one onto a LiveConfig. Any change marks the Build button bold so you
    know the model on screen no longer matches the settings; pressing it emits `buildRequested`
    with the current config. Mine files are opened, generated and saved here, since a mine is
    part of the configuration.
    """
    buildRequested = Signal(object)          # LiveConfig
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setMinimumWidth(300)
        self.mine: MineSpec | None = None
        self.devices = None
        body = QWidget()
        col = QVBoxLayout(body)
        col.addWidget(self._workings_group())
        col.addWidget(self._loop_group())
        col.addWidget(self._ground_group())
        col.addWidget(self._run_group())
        self.build_btn = QPushButton("Build model")
        self.build_btn.setMinimumHeight(34)
        self.build_btn.clicked.connect(self.request_build)
        col.addWidget(self.build_btn)
        self.stats = QLabel("")
        self.stats.setWordWrap(True)
        self.stats.setStyleSheet(f"color: {INK2};")
        col.addWidget(self.stats)
        col.addStretch(1)
        self.setWidget(body)
        self._sync_enabled()
        for w in body.findChildren(QDoubleSpinBox) + body.findChildren(QSpinBox):
            w.valueChanged.connect(self._on_changed)
        for w in body.findChildren(QComboBox):
            w.currentIndexChanged.connect(self._on_changed)
        for w in body.findChildren(QCheckBox):
            w.toggled.connect(self._on_changed)

    #  groups
    def _workings_group(self):
        """Controls for what is being cooled: a single drift (grid, size, rock) or a mine file
        (open, generate, save, plan cell).
        """
        g = QGroupBox("Workings")
        v = QVBoxLayout(g)
        self.source = QComboBox()
        self.source.addItem("Single drift", "drift")
        self.source.addItem("Mine file", "mine")
        v.addWidget(self.source)
        self.stack = QStackedWidget()
        # --- single drift
        d = QWidget(); f = QFormLayout(d); f.setContentsMargins(0, 0, 0, 0)
        self.grid_kind = QComboBox()
        for key, (label, _) in DRIFT_GRIDS.items():
            self.grid_kind.addItem(label, key)
        self.grid_kind.setCurrentIndex(list(DRIFT_GRIDS).index("block3d"))
        self.length = _spin(50, 5000, 500, 50, 0, "m")
        self.width = _spin(1, 20, 6, 0.5, 1, "m")
        self.height = _spin(1, 10, 2, 0.5, 1, "m")
        self.rock = QComboBox()
        for name in ROCKS:
            if name != "water":
                self.rock.addItem(name)
        self.rock.setCurrentText("sandstone")
        for label, w in (("Grid", self.grid_kind), ("Length", self.length), ("Width", self.width),
                         ("Height", self.height), ("Rock", self.rock)):
            f.addRow(label, w)
        self.stack.addWidget(d)
        # --- mine file
        m = QWidget(); mv = QVBoxLayout(m); mv.setContentsMargins(0, 0, 0, 0)
        self.mine_label = QLabel("No mine loaded.")
        self.mine_label.setWordWrap(True)
        mv.addWidget(self.mine_label)
        row = QHBoxLayout()
        b_open = QPushButton("Open…"); b_open.clicked.connect(self._open_mine)
        self.b_save = QPushButton("Save as…"); self.b_save.clicked.connect(self._save_mine)
        row.addWidget(b_open); row.addWidget(self.b_save)
        mv.addLayout(row)
        gen = QHBoxLayout()
        self.gen_size = QComboBox(); self.gen_size.addItems(list(SIZES)); self.gen_size.setCurrentText("small")
        self.gen_seed = QSpinBox(); self.gen_seed.setRange(0, 10**6); self.gen_seed.setPrefix("seed ")
        b_gen = QPushButton("Generate"); b_gen.clicked.connect(self._generate)
        gen.addWidget(self.gen_size); gen.addWidget(self.gen_seed); gen.addWidget(b_gen)
        mv.addLayout(gen)
        mf = QFormLayout()
        self.plan_cell = _spin(0, 50, 0, 0.5, 1, "m")
        self.plan_cell.setSpecialValueText("from file")
        mf.addRow("Plan cell", self.plan_cell)
        mv.addLayout(mf)
        self.stack.addWidget(m)
        v.addWidget(self.stack)
        self.source.currentIndexChanged.connect(lambda i: self.stack.setCurrentIndex(i))
        return g

    def _loop_group(self):
        """Controls for the surface loop: flow, inlet temperature or heat load, and the runaway
        temperature limit.
        """
        g = QGroupBox("Loop"); f = QFormLayout(g)
        self.flow = _spin(10, 5000, 360, 10, 0, "m³/h")
        self.T_in = _spin(5, 95, 35, 1, 1, "°C")
        self.surface = QComboBox()
        for key, label in SURFACE_MODES.items():
            self.surface.addItem(label, key)
        self.load = _spin(0.01, 200, 0.3, 0.1, 2, "MW")
        self.T_max = _spin(30, 99, 95, 1, 0, "°C")
        for label, w in (("Flow", self.flow), ("Inlet temperature", self.T_in), ("Surface", self.surface),
                         ("Heat load", self.load), ("Stop if water exceeds", self.T_max)):
            f.addRow(label, w)
        return g

    def _ground_group(self):
        """Controls for the ground: rock temperature (or derive it from mine depth) and the wall
        film coefficient.
        """
        g = QGroupBox("Ground"); f = QFormLayout(g)
        self.T_rock = _spin(0, 80, 13, 0.5, 1, "°C")
        self.T_rock_from_file = QCheckBox("from mine depth")
        self.T_rock_from_file.setChecked(True)
        row = QHBoxLayout(); row.addWidget(self.T_rock); row.addWidget(self.T_rock_from_file)
        self.h_film = _spin(50, 5000, 800, 50, 0, "W/m²K")
        f.addRow("Rock temperature", row)
        f.addRow("Wall film", self.h_film)
        return g

    def _run_group(self):
        """Controls for the run: engine, compute device, run length and time step. Engines that
        can't run here are greyed out with the reason as a tooltip.
        """
        g = QGroupBox("Run"); f = QFormLayout(g)
        self.engine = QComboBox()
        for key, info in ENGINES.items():
            self.engine.addItem(info.label, key)
            reason = info.check()
            item = self.engine.model().item(self.engine.count() - 1)
            item.setToolTip(info.note if reason is None else f"Unavailable: {reason}")
            if reason is not None:
                item.setEnabled(False)
        self.device = QComboBox()
        self.device.setToolTip("Where the heat solve runs. Auto uses an NVIDIA GPU when CuPy can drive "
                               "one and falls back to the CPU otherwise.")
        self._fill_devices(None)
        self.days = _spin(1, 3650, 365, 30, 0, "days")
        self.dt = _spin(0.05, 10, 0.5, 0.25, 2, "days")
        f.addRow("Engine", self.engine)
        f.addRow("Compute", self.device)
        f.addRow("Simulate", self.days)
        f.addRow("Time step", self.dt)
        return g

    #  compute devices
    def _fill_devices(self, report) -> None:
        """Auto / CPU / each working GPU. Before the simulation process has looked, the GPU
        row says so; if there is no usable GPU it says why (tooltip)."""
        current = self.device.currentData() or "auto"
        self.device.blockSignals(True)
        self.device.clear()
        self.device.addItem("Auto (GPU if available)", "auto")
        self.device.addItem("CPU", "cpu")
        if report is None:
            self.device.addItem("GPU: checking…", None)
            self._disable_last(self.device, "The simulation process is checking for a CUDA GPU.")
        elif not report.gpus:
            self.device.addItem("GPU: not available", None)
            self._disable_last(self.device, f"{report.gpu_reason}\n\nFor GPU support: {CUPY_HINT}")
        else:
            for d in report.gpus:
                self.device.addItem(d.label, d.key)
        i = self.device.findData(current)
        self.device.setCurrentIndex(i if i >= 0 else 0)
        self.device.blockSignals(False)
        self._sync_enabled()

    @staticmethod
    def _disable_last(combo, tip):
        """Grey out the last entry of a combo box and explain why in its tooltip."""
        item = combo.model().item(combo.count() - 1)
        item.setEnabled(False)
        item.setToolTip(tip)

    def set_devices(self, report) -> None:
        """Show the compute devices the simulation process found, and which one Auto will pick."""
        self.devices = report
        self._fill_devices(report)
        gpu = report.gpus[0].label if report.gpus else None
        self.device.setItemText(0, f"Auto ({gpu.split(':', 1)[-1].strip()})" if gpu else "Auto (CPU — no GPU)")

    #  state
    def config(self) -> LiveConfig:
        """Read every widget into a LiveConfig."""
        is_mine = self.source.currentData() == "mine"
        T_rock = None if (is_mine and self.T_rock_from_file.isChecked()) else self.T_rock.value()
        return LiveConfig(
            source=self.source.currentData(), drift_grid=self.grid_kind.currentData(),
            length_m=self.length.value(), width_m=self.width.value(), height_m=self.height.value(),
            rock=self.rock.currentText(), mine=self.mine,
            plan_cell_m=self.plan_cell.value() or None,
            flow_m3_h=self.flow.value(), T_in_C=self.T_in.value(), surface=self.surface.currentData(),
            load_MW=self.load.value(), T_max_C=self.T_max.value(), T_rock_C=T_rock,
            h_film_W_m2K=self.h_film.value(), engine=self.engine.currentData(),
            device=self.device.currentData() or "auto",
            sim_days=self.days.value(), dt_days=self.dt.value())

    def request_build(self) -> None:
        """Clear the 'needs rebuild' state and ask for a build with the current settings."""
        self.build_btn.setText("Build model")
        self.build_btn.setStyleSheet("")
        self.buildRequested.emit(self.config())

    def set_mine(self, mine: MineSpec) -> None:
        """Load a mine into the panel, adopting its operating conditions and rock temperature."""
        self.mine = mine
        self.source.setCurrentIndex(1)
        self.mine_label.setText(f"<b>{mine.name}</b><br>{mine.description or mine.layout.describe()}")
        c = mine.conditions                       # adopt the file's operating conditions
        self.flow.setValue(c.flow_m3_h)
        self.T_in.setValue(c.T_in_C)
        self.h_film.setValue(c.h_film_W_m2K)
        try:
            self.T_rock.setValue(mine.T_rock_C)
            self.T_rock_from_file.setChecked(True)
        except ValueError:                        # file gives neither T_rock nor depth
            self.T_rock_from_file.setChecked(False)
        self._on_changed()

    def load_mine_file(self, path: str) -> None:
        """Read a mine file from disk and load it into the panel."""
        self.set_mine(load_mine(path))

    def show_stats(self, text: str) -> None:
        """Show build and compute details under the Build button."""
        self.stats.setText(text)

    #  actions
    def _open_mine(self):
        """Ask for a mine file and load it, showing any reading error in a dialog."""
        path, _ = QFileDialog.getOpenFileName(self, "Open mine file", "", MINE_FILTER)
        if path:
            try:
                self.load_mine_file(path)
            except MineFileError as e:
                QMessageBox.warning(self, "Could not read mine file", str(e))

    def _save_mine(self):
        """Save the current mine to a file of the user's choice."""
        if self.mine is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save mine file", f"{self.mine.name}.json", MINE_FILTER)
        if path:
            save_mine(self.mine, path)

    def _generate(self):
        """Generate a random mine from the chosen size and seed, then advance the seed for next time."""
        seed = self.gen_seed.value()
        self.set_mine(generate_mine(seed=seed, size=self.gen_size.currentText()))
        self.gen_seed.setValue(seed + 1)

    def _on_changed(self, *_):
        """Something changed: update which controls apply and flag that a rebuild is needed."""
        self._sync_enabled()
        self.build_btn.setText("Rebuild to apply changes")
        self.build_btn.setStyleSheet("font-weight: bold;")
        self.changed.emit()

    def _sync_enabled(self):
        """Enable only the controls that matter for the current choices (e.g. heat load only with
        TESPy).
        """
        is_mine = self.source.currentData() == "mine"
        self.T_rock_from_file.setVisible(is_mine)
        self.T_rock.setEnabled(not (is_mine and self.T_rock_from_file.isChecked()))
        self.b_save.setEnabled(self.mine is not None)
        tespy = self.surface.currentData() == "fixed_load"
        self.load.setEnabled(tespy)
        self.T_in.setEnabled(not tespy)
        if hasattr(self, "device") and hasattr(self, "engine"):
            info = ENGINES.get(self.engine.currentData())
            self.device.setEnabled(bool(info and info.supports_device))
