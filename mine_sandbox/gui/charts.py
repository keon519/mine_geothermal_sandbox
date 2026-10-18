"""Time-series panel: loop temperatures (+ probes), heat, and cooling effectiveness.

The three plots share one time axis. A vertical cursor marks the time being
shown in the maps; drag it (or click a plot) to scrub back through the run.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from .theme import INK2, SERIES

FLUSH_DAYS = 5.0

class TimeSeriesPanel(QWidget):
    """The three linked charts under the maps: loop and probe temperatures, heat into the rock, and
    cooling achieved.

    They share a time axis and a cursor marking the moment shown in the maps. Dragging the
    cursor or clicking a chart emits `seekRequested` with that day, which the main window uses
    to replay that moment.
    """
    seekRequested = Signal(float)            # simulated day

    def __init__(self):
        super().__init__()
        self.win = pg.GraphicsLayoutWidget()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.win)
        self.p_temp = self.win.addPlot(row=0, col=0)
        self.p_heat = self.win.addPlot(row=0, col=1)
        self.p_eff = self.win.addPlot(row=0, col=2)
        for p, title, unit in ((self.p_temp, "Water and probe temperatures", "°C"),
                               (self.p_heat, "Heat into the rock", "MW"),
                               (self.p_eff, "Cooling achieved", "% of possible")):
            p.setTitle(title, color=INK2, size="10pt")
            if p is not self.p_temp:
                p.setTitle(title + f"  (scaled after day {FLUSH_DAYS:g})", color=INK2, size="10pt")
            p.setLabel("left", unit)
            p.getAxis("left").enableAutoSIPrefix(False)
            p.getAxis("bottom").enableAutoSIPrefix(False)
            p.setLabel("bottom", "day")
            p.showGrid(x=True, y=True, alpha=0.25)
            p.setMenuEnabled(False)
            p.addLegend(offset=(-5, -5) if p is self.p_temp else (-5, 5), labelTextColor=INK2)
        self.p_heat.setXLink(self.p_temp)
        self.p_eff.setXLink(self.p_temp)
        pen = lambda c, **kw: pg.mkPen(c, width=2, **kw)
        self.c_in = self.p_temp.plot(pen=pen(SERIES[0]), name="T in")
        self.c_out = self.p_temp.plot(pen=pen(SERIES[1]), name="T out")
        self.rock_line = pg.InfiniteLine(angle=0, pen=pg.mkPen(INK2, width=1, style=Qt.DashLine),
                                         label="rock", labelOpts={"color": INK2, "position": 0.05})
        self.p_temp.addItem(self.rock_line)
        self.c_heat = self.p_heat.plot(pen=pen(SERIES[1]), name="removed by the mine")
        self.c_load = self.p_heat.plot(pen=pen(SERIES[0], style=Qt.DashLine), name="surface load")
        self.c_eff = self.p_eff.plot(pen=pen(SERIES[1]), name="(T_in − T_out)/(T_in − T_rock)")
        for c in (self.c_in, self.c_out, self.c_heat, self.c_load, self.c_eff):
            self._light(c)
        self.probe_curves: dict[str, pg.PlotDataItem] = {}
        self.cursors = []
        for p in (self.p_temp, self.p_heat, self.p_eff):
            cur = pg.InfiniteLine(angle=90, movable=True, pen=pg.mkPen(INK2, width=1))
            cur.sigDragged.connect(lambda line: self._sync_cursor(line.value(), emit=True))
            p.addItem(cur)
            self.cursors.append(cur)
        self.win.scene().sigMouseClicked.connect(self._clicked)
        self._T_rock = 0.0

    @staticmethod
    def _light(curve) -> None:
        """Long runs have tens of thousands of points: draw about one per pixel."""
        curve.setDownsampling(auto=True, method="peak")
        curve.setClipToView(True)
        curve.setSkipFiniteCheck(True)

    def dispose(self) -> None:
        """Remove every plot while Python still owns them. Leaving pyqtgraph items for
        Qt to destroy sends them resize events after their Python side is gone, and
        PySide 6.12 crashes reporting the resulting error."""
        self.win.scene().sigMouseClicked.disconnect(self._clicked)
        self.win.clear()

    def reset(self, T_rock_C: float) -> None:
        """Clear all curves and probes for a new run with the given rock temperature."""
        self._T_rock = T_rock_C
        self.rock_line.setPos(T_rock_C)
        for c in (self.c_in, self.c_out, self.c_heat, self.c_load, self.c_eff):
            c.setData([], [])
        for name in list(self.probe_curves):
            self.remove_probe(name)
        self._sync_cursor(0.0)

    def update_series(self, s: dict[str, np.ndarray]) -> None:
        """Redraw the loop curves from the History's scalar series.

        Cooling achieved is computed here as a percentage of the most possible. Once the initial
        cold-water flush is over, the heat and cooling charts are scaled to the data after it so
        the long-term trend stays readable.
        """
        t = s["t_d"]
        if len(t) == 0:
            return
        self.c_in.setData(t, s["T_in_C"])
        self.c_out.setData(t, s["T_out_C"])
        self.c_heat.setData(t, s["heat_removed_W"] / 1e6)
        load = s["Q_load_W"]
        self.c_load.setData(t, load / 1e6) if np.all(np.isfinite(load)) else self.c_load.setData([], [])
        with np.errstate(divide="ignore", invalid="ignore"):
            eff = 100 * (s["T_in_C"] - s["T_out_C"]) / (s["T_in_C"] - self._T_rock)
        self.c_eff.setData(t, eff)
        # the first days are the flush of the cold water that filled the workings;
        # once past them, scale to what follows so the long-term trend is readable
        late = t > FLUSH_DAYS
        if late.sum() > 4:
            for plot, y in ((self.p_heat, s["heat_removed_W"] / 1e6), (self.p_eff, eff)):
                hi = np.nanmax(np.abs(y[late]))
                if np.isfinite(hi) and hi > 0:
                    plot.setYRange(min(0.0, float(np.nanmin(y[late]))), 1.3 * hi, padding=0)

    def set_probe(self, name: str, color: str, t: np.ndarray, T: np.ndarray) -> None:
        """Create or update the temperature curve for a pinned probe."""
        curve = self.probe_curves.get(name)
        if curve is None:
            curve = self.p_temp.plot(pen=pg.mkPen(color, width=1.5), name=name)
            self._light(curve)
            self.probe_curves[name] = curve
        curve.setData(t, T)

    def remove_probe(self, name: str) -> None:
        """Remove a probe's curve and its legend entry."""
        curve = self.probe_curves.pop(name, None)
        if curve is not None:
            self.p_temp.removeItem(curve)
            self.p_temp.legend.removeItem(curve)

    def set_cursor(self, t_d: float) -> None:
        """Move the time cursor on all three charts without emitting a seek."""
        self._sync_cursor(t_d)

    def _sync_cursor(self, t, emit=False):
        """Move every chart's cursor to the same day, optionally asking the window to seek there."""
        for c in self.cursors:
            c.blockSignals(True)
            c.setValue(t)
            c.blockSignals(False)
        if emit:
            self.seekRequested.emit(float(t))

    def _clicked(self, ev):
        """Seek to the day under the mouse when any of the charts is clicked."""
        for p in (self.p_temp, self.p_heat, self.p_eff):
            if p.vb.sceneBoundingRect().contains(ev.scenePos()):
                self._sync_cursor(p.vb.mapSceneToView(ev.scenePos()).x(), emit=True)
                return
