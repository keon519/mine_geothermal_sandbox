"""Scenario -> ThermalProblem: property assignment and the wall film."""
import numpy as np
import pytest

from mine_sandbox import (ROCKS, WATER, GeologyNotSupportedError, Layer, LayeredGeology,
                          RadialDriftGrid, StructuredDriftGrid, assemble, make_scenario)

SEAM = LayeredGeology(ROCKS["shale"], (Layer(-1.0, 1.0, ROCKS["coal"]),
                                        Layer(4.0, 1e9, ROCKS["sandstone"])))


def test_layered_geology_rejected_by_axisymmetric_grid():
    with pytest.raises(GeologyNotSupportedError):
        assemble(make_scenario(rock=SEAM, grid=RadialDriftGrid()))


def test_layered_geology_placed_by_elevation():
    p = assemble(make_scenario(rock=SEAM, width_m=6, height_m=2, diameter_m=None,
                               grid=StructuredDriftGrid.block_3d(n_x=3, n_side=5, n_core=5)))
    z, k = p.graph.centroids[:, 2], p.props.k_solid
    rock = ~p.graph.is_drift
    assert np.all(k[rock & (np.abs(z) < 1.0)] == ROCKS["coal"].k)
    assert np.all(k[rock & (z > 4.0)] == ROCKS["sandstone"].k)
    assert np.all(k[rock & (z > 1.0) & (z < 4.0)] == ROCKS["shale"].k)


def test_wall_film_only_on_drift_rock_faces():
    sc = make_scenario(h_film_W_m2K=500.0)
    raw = sc.grid.build(sc.drift)
    g = assemble(sc).graph
    film = WATER.k / 500.0
    d = g.is_drift
    wall = d[g.conn_a] ^ d[g.conn_b]
    drift_side = np.where(d[g.conn_a], g.dist_a, g.dist_b)
    assert np.allclose(drift_side[wall], film)
    assert np.array_equal(g.dist_a[~wall], raw.dist_a[~wall])
    assert np.array_equal(g.dist_b[~wall], raw.dist_b[~wall])


def test_porosity_and_hydraulic_contrast():
    p = assemble(make_scenario())
    d = p.graph.is_drift
    assert np.all(p.props.porosity[d] == 1.0) and np.all(p.props.porosity[~d] == 0.01)
    assert p.props.hydraulic_k_m_d[~d].max() < 1e-6 * p.props.hydraulic_k_m_d[d].min()
