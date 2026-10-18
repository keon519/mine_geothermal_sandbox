"""Shared test setup: the grids every contract test runs against, a tiny mine, and MODFLOW skip
markers.

Tests that need MODFLOW are skipped (not failed) when mf6 or libmf6 can't be found, so the rest
of the suite still runs on a machine without them.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mine_sandbox import RadialDriftGrid, StructuredDriftGrid  # noqa: E402
from mine_sandbox.mf6 import find_libmf6, find_mf6  # noqa: E402


def _have(finder):
    """True if a MODFLOW finder locates its file."""
    try:
        finder()
        return True
    except FileNotFoundError:
        return False


needs_mf6 = pytest.mark.skipif(not _have(find_mf6), reason="mf6 executable not found")
needs_libmf6 = pytest.mark.skipif(not _have(find_libmf6), reason="libmf6 not found")

from mine_sandbox.mines import ConditionsSpec, MineGrid, MineGridSpec, MineSpec, RoomAndPillar  # noqa: E402

# A deliberately small mine for tests: 3 entries x 3 pillars, 6 m cells.
TINY_MINE = MineSpec(name="tiny", layout=RoomAndPillar(n_entries=3, n_pillars_along=3, entry_width_m=6,
                                                        crosscut_width_m=6, pillar_length_m=12,
                                                        pillar_width_m=12),
                     conditions=ConditionsSpec(T_rock_C=13.0),
                     grid=MineGridSpec(plan_cell_m=6, margin_m=40, margin_cells=4, layers_above=4,
                                       layers_below=4, extent_above_m=40, extent_below_m=40))

# Grids that discretise a single drift from a DriftSpec.
DRIFT_GRIDS = {
    "radial": RadialDriftGrid(),
    "slab2d": StructuredDriftGrid.plane_2d(n_x=6, n_side=6),
    "block3d": StructuredDriftGrid.block_3d(n_x=6, n_side=6, n_core=5),
}
# Every grid the package ships. The generic contract tests run against all of them,
# so a new grid only has to be added here to be held to the same rules.
ALL_GRIDS = {**DRIFT_GRIDS, "mine": MineGrid(TINY_MINE)}


@pytest.fixture
def ws(tmp_path):
    """A fresh temporary folder for MODFLOW's input and output files."""
    return str(tmp_path)
