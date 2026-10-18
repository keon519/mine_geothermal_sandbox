"""Contract tests: every Grid must produce a CellGraph downstream code can trust."""
import numpy as np
import pytest

from conftest import ALL_GRIDS, DRIFT_GRIDS
from mine_sandbox import CircularSection, DriftSpec, RectangularSection
from mine_sandbox.grids import StructuredDriftGrid

DRIFTS = [DriftSpec(500.0, CircularSection(4.0)), DriftSpec(300.0, RectangularSection(6.0, 2.0))]


@pytest.mark.parametrize("grid", ALL_GRIDS.values(), ids=ALL_GRIDS.keys())
class TestAnyGridContract:
    """What every consumer (assembly, engines, results, GUI) relies on."""

    def test_validates(self, grid):
        grid.build(DRIFTS[0]).validate()

    def test_has_a_lattice_for_slice_views(self, grid):
        g = grid.build(DRIFTS[0])
        lat = g.lattice
        assert lat is not None
        for k in range(lat.shape[2]):
            assert lat.layer(k).shape == lat.shape[:2]
        assert lat.voxel_of(g.inlet_cell)

    def test_rock_profile_above_the_route(self, grid):
        g = grid.build(DRIFTS[0])
        d, nodes = g.wall_profile()
        assert len(nodes) >= 3 and np.all(np.diff(d) > 0)
        assert not np.any(g.is_drift[nodes[1:]])

    def test_route_is_connected_through_water(self, grid):
        g = grid.build(DRIFTS[0])
        pairs = {tuple(sorted(p)) for p in zip(g.conn_a.tolist(), g.conn_b.tolist())}
        path = g.path_cells.tolist()
        assert all(tuple(sorted(p)) in pairs for p in zip(path[:-1], path[1:]))

    def test_far_field_is_rock_on_the_outer_boundary(self, grid):
        g = grid.build(DRIFTS[0])
        assert len(g.far_field_cells) > 0
        assert not np.any(g.is_drift[g.far_field_cells])
        shape = np.array(g.lattice.shape)
        for n in g.far_field_cells[:: max(1, len(g.far_field_cells) // 200)]:
            ijk = np.array(g.lattice.voxel_of(int(n)))
            assert np.any((ijk == 0) | (ijk == shape - 1)), f"far-field cell {n} is inside the block"


@pytest.mark.parametrize("grid", DRIFT_GRIDS.values(), ids=DRIFT_GRIDS.keys())
@pytest.mark.parametrize("drift", DRIFTS, ids=["round", "rect"])
class TestDriftGridContract:
    """Extra rules for grids built from a single DriftSpec."""

    def test_validates(self, grid, drift):
        grid.build(drift).validate()

    def test_drift_runs_inlet_to_outlet(self, grid, drift):
        g = grid.build(drift)
        x = g.centroids[g.drift_cells, 0]
        assert np.all(np.diff(x) > 0) and x[0] < x[-1] <= drift.length_m

    def test_drift_volume_is_the_tunnel(self, grid, drift):
        g = grid.build(drift)
        assert np.isclose(g.volumes[g.drift_cells].sum(), drift.section.area_m2 * drift.length_m)

    def test_wall_area(self, grid, drift):
        """Area of drift-rock faces = perimeter x length (for the shape the grid represents)."""
        g = grid.build(drift)
        d = g.is_drift
        wall = d[g.conn_a] ^ d[g.conn_b]
        s = drift.section
        if isinstance(grid, StructuredDriftGrid):
            per = 2 * s.width_m + (2 * s.height_m if grid.three_d else 0.0)  # slab: top+bottom only
        else:
            per = 2 * np.pi * s.equivalent_radius_m
        assert np.isclose(g.face_area[wall].sum(), per * drift.length_m)

    def test_far_field_surrounds(self, grid, drift):
        g = grid.build(drift)
        assert len(g.far_field_cells) > 0
        dist = np.hypot(g.centroids[g.far_field_cells, 1], g.centroids[g.far_field_cells, 2])
        assert np.min(dist) > 2 * max(drift.section.width_m, drift.section.height_m)

    def test_views_are_available(self, grid, drift):
        g = grid.build(drift)
        d, nodes = g.wall_profile()
        assert len(nodes) >= 3 and np.all(np.diff(d) > 0)
        assert set(g.drift_cells).issubset(set(g.section_plane()))


def test_structured_total_volume():
    drift = DriftSpec(200.0, RectangularSection(6.0, 2.0))
    grid = StructuredDriftGrid.block_3d(n_x=4, n_side=5, extent_m=30.0)
    assert np.isclose(grid.build(drift).volumes.sum(), 200.0 * 60.0 * 60.0)
