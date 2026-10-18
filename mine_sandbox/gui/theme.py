"""Colours shared by every widget (one place to restyle the app)."""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg

BACKGROUND = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
WATER_OUTLINE = "#0b3d91"

# fixed categorical order, never cycled: T_in, T_out, then probes
SERIES = ["#eb6834", "#2a78d6", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
PROBE_COLORS = SERIES[2:]

# sequential, one hue: light = undisturbed rock, dark = hottest
_HEAT_STOPS = [(0.0, "#fdf3ea"), (0.35, "#f5b27a"), (0.7, "#d95926"), (1.0, "#7a2a0c")]
HEAT_CMAP = pg.ColorMap(np.array([p for p, _ in _HEAT_STOPS]),
                        [pg.mkColor(c) for _, c in _HEAT_STOPS])


def apply_pyqtgraph_theme() -> None:
    """Set pyqtgraph's global look to match the app (background, ink colour, antialiasing, image
    axis order).
    """
    pg.setConfigOptions(antialias=True, background=BACKGROUND, foreground=INK2, imageAxisOrder="row-major")
