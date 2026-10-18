"""The layer picker: one row per horizontal layer of the model, top to bottom.

Each row shows the layer's height band, its rock, and how much it has
warmed (shaded with the same colour scale as the maps), so you can see at a
glance which layers the heat has reached and click one to view it in plan.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem

from .theme import HEAT_CMAP, INK


class LayerSelector(QTableWidget):
    """Table of the model's horizontal layers, listed top to bottom, that doubles as a layer
    picker.

    Layers holding water are marked and bolded. Clicking a row emits `layerSelected` with the
    layer's lattice index (counted from the bottom), which the main window uses to switch the
    plan view.
    """
    layerSelected = Signal(int)              # lattice k index

    def __init__(self):
        super().__init__(0, 3)
        self.setHorizontalHeaderLabels(["Height (m)", "Rock", "Max rise"])
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        h = self.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.Stretch)
        h.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.setMinimumWidth(250)
        self._nz = 0
        self.itemSelectionChanged.connect(self._changed)

    def set_layers(self, z_edges: np.ndarray, rock_labels: list[str], water_layers: set[int],
                   radial: bool = False, select: int | None = None) -> None:
        """Fill the table for a new model: each layer's height band and rock, with water layers
        flagged. Signals are held off while filling.
        """
        nz = len(z_edges) - 1
        self._nz = nz
        self.setHorizontalHeaderLabels(["Radius (m)" if radial else "Height (m)", "Rock", "Max rise"])
        self.blockSignals(True)
        self.setRowCount(nz)
        for row in range(nz):
            k = nz - 1 - row                      # top layer first
            lo, hi = z_edges[k], z_edges[k + 1]
            band = f"r {lo:.1f}–{hi:.1f}" if radial else f"{lo:+.1f} to {hi:+.1f}"
            rock = rock_labels[k]
            if k in water_layers:
                rock = "water ◀" if rock in ("", "water") else rock + "  + water ◀"
            for col, text in enumerate((band, rock, "")):
                item = QTableWidgetItem(text)
                if k in water_layers:
                    f = item.font(); f.setBold(True); item.setFont(f)
                self.setItem(row, col, item)
        self.blockSignals(False)
        if select is not None:
            self.select_layer(select)

    def select_layer(self, k: int) -> None:
        """Select a layer by its lattice index (bottom is 0), translating to the table's top-down
        row order.
        """
        if 0 <= k < self._nz:
            self.selectRow(self._nz - 1 - k)

    def update_rise(self, rise_per_layer: np.ndarray, scale: float) -> None:
        """Shade the third column by each layer's maximum temperature rise."""
        lut = HEAT_CMAP.getLookupTable(nPts=256)
        for row in range(self._nz):
            k = self._nz - 1 - row
            item = self.item(row, 2)
            if item is None:
                continue
            r = float(rise_per_layer[k])
            item.setText(f"{r:5.1f} °C")
            c = lut[int(np.clip(r / max(scale, 1e-9), 0, 1) * 255)]
            item.setBackground(QBrush(QColor(int(c[0]), int(c[1]), int(c[2]))))
            item.setForeground(QBrush(QColor("#ffffff" if r / max(scale, 1e-9) > 0.6 else INK)))
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

    def _changed(self):
        """Emit the newly selected layer's lattice index."""
        rows = self.selectionModel().selectedRows()
        if rows:
            self.layerSelected.emit(self._nz - 1 - rows[0].row())
