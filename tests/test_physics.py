"""MODFLOW results: regression, energy conservation, and independent benchmarks."""
import json
import os

import numpy as np
import pytest

from conftest import ALL_GRIDS, needs_mf6
from mine_sandbox import ROCKS, StructuredDriftGrid, make_scenario, run_fixed_inlet
from validation.benchmarks import planar_flux, radial_flux, section_flux

GOLD = json.load(open(os.path.join(os.path.dirname(__file__), "data", "gold_radial_v1.json")))


@needs_mf6
@pytest.mark.parametrize("name,kw", [("baseline", {}), ("coal", {"rock": "coal"}),
                                     ("L2000", {"length_m": 2000}), ("flow90", {"flow_m3_h": 90}),
                                     ("D6", {"diameter_m": 6})])
def test_matches_pre_refactor_results(ws, name, kw):
    """The radial grid must reproduce the original single-file model exactly."""
    r = run_fixed_inlet(make_scenario(workspace=ws, name=name, **kw))
    for day, (T_out, heat) in GOLD[name].items():
        i = r.at_day(float(day))
        assert r.T_out_C[i] == pytest.approx(T_out, abs=1e-6)
        assert r.heat_removed_W[i] == pytest.approx(heat, rel=1e-6)


@needs_mf6
@pytest.mark.parametrize("grid", ALL_GRIDS.values(), ids=ALL_GRIDS.keys())
def test_energy_is_conserved(ws, grid):
    """Heat that left the water == heat now stored in water + rock."""
    r = run_fixed_inlet(make_scenario(workspace=ws, grid=grid, sim_days=10))
    dt = np.diff(np.r_[0.0, r.times_d]) * 86400.0
    assert np.sum(r.heat_removed_W * dt) == pytest.approx(r.stored_heat_J(), rel=1e-4)


# Wall held ~22 C above the rock: very high flow and negligible film resistance
HOT_WALL = dict(flow_m3_h=36000.0, h_film_W_m2K=1e7, sim_days=30, T_in_C=35.0)
SS = ROCKS["sandstone"]


def _flux_per_m(r, day):
    """Heat removed per metre of drift on a given day, and how far the water sits above rock
    temperature on average.
    """
    i = r.at_day(day)
    _, Tw = r.along_drift(i)
    return r.heat_removed_W[i] / r.scenario.drift.length_m, Tw.mean() - r.scenario.ground.T_rock_C


@needs_mf6
def test_radial_grid_vs_reference(ws):
    q, dT = _flux_per_m(run_fixed_inlet(make_scenario(workspace=ws, **HOT_WALL)), 30)
    ref = radial_flux(SS.k, SS.volumetric_heat_capacity, dT, 2.0, [30])[0]
    assert q == pytest.approx(ref, rel=0.02)


@needs_mf6
def test_slab_grid_vs_exact_solution(ws):
    sc = make_scenario(workspace=ws, width_m=6, height_m=2, diameter_m=None,
                       grid=StructuredDriftGrid.plane_2d(), **HOT_WALL)
    q, dT = _flux_per_m(run_fixed_inlet(sc), 30)
    assert q == pytest.approx(planar_flux(SS.k, SS.volumetric_heat_capacity, dT, 6.0, 30), rel=0.02)


@needs_mf6
def test_block_grid_vs_reference(ws):
    sc = make_scenario(workspace=ws, width_m=6, height_m=2, diameter_m=None, length_m=100,
                       grid=StructuredDriftGrid.block_3d(n_x=4), **HOT_WALL)
    q, dT = _flux_per_m(run_fixed_inlet(sc), 30)
    ref = section_flux(SS.k, SS.volumetric_heat_capacity, dT, 6.0, 2.0, [30])[0]
    assert q == pytest.approx(ref, rel=0.04)
