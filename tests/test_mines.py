"""Mine files, layouts, the generator and MineGrid."""
import glob
import json
import os

import numpy as np
import pytest

from conftest import TINY_MINE
from mine_sandbox import assemble
from mine_sandbox.mines import (Composite, MineFileError, MineGeometryError, MineGrid, Polylines,
                                PumpSpec, Rectangle, RoomAndPillar, TracedDrift, generate_mine,
                                load_mine, mine_from_dict, mine_statistics, mine_to_dict, save_mine,
                                scenario_from_mine)
from mine_sandbox.mines.grid import bridge_diagonals

EXAMPLES = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "examples", "mines", "*.json")))


# ------------------------------------------------------------------ layouts
def test_room_and_pillar_open_area_matches_its_definition():
    rp = RoomAndPillar(n_entries=4, n_pillars_along=5, entry_width_m=6, crosscut_width_m=6,
                       pillar_length_m=18, pillar_width_m=12)
    xs = np.arange(-10, rp.length_m + 10, 0.5) + 0.25
    ys = np.arange(-10, rp.width_m + 10, 0.5) + 0.25
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    area = rp.is_open(X, Y).sum() * 0.25
    assert area == pytest.approx(rp.nominal_open_area_m2, rel=0.02)


def test_rotation_keeps_area():
    a = RoomAndPillar(n_entries=3, n_pillars_along=4)
    b = RoomAndPillar(n_entries=3, n_pillars_along=4, origin=(100, 50), rotation_deg=90)
    xmin, ymin, xmax, ymax = b.bounds()
    X, Y = np.meshgrid(np.arange(xmin, xmax, 0.5) + 0.25, np.arange(ymin, ymax, 0.5) + 0.25, indexing="ij")
    assert b.is_open(X, Y).sum() * 0.25 == pytest.approx(a.nominal_open_area_m2, rel=0.03)


def test_polyline_width_and_composite_holes():
    p = Polylines(drifts=(TracedDrift(points=((0, 0), (100, 0)), width_m=5),))
    assert p.is_open(np.array([50.0]), np.array([2.4]))[0]
    assert not p.is_open(np.array([50.0]), np.array([2.6]))[0]
    c = Composite(parts=(p,), holes=(Rectangle(origin=(40, -5), length_m=20, width_m=10),))
    assert not c.is_open(np.array([50.0]), np.array([0.0]))[0]
    assert c.is_open(np.array([10.0]), np.array([0.0]))[0]


def test_bridging_keeps_diagonal_tunnels_continuous():
    o = np.zeros((4, 4), bool)
    o[0, 0] = o[1, 1] = o[2, 2] = True              # corner-to-corner only
    b = bridge_diagonals(o)
    from scipy import ndimage
    assert ndimage.label(b)[1] == 1                 # one 4-connected tunnel
    assert b.sum() == o.sum() + 2                   # one bridge per corner contact


# ------------------------------------------------------------------ files
@pytest.mark.parametrize("path", EXAMPLES, ids=[os.path.basename(p) for p in EXAMPLES])
def test_example_files_load_and_build(path):
    m = load_mine(path)
    p = assemble(scenario_from_mine(m, sim_days=1))
    st = mine_statistics(p.graph)
    assert st["water_cells"] > 50 and st["route_length_m"] > 100
    assert mine_to_dict(mine_from_dict(json.loads(json.dumps(mine_to_dict(m))))) == mine_to_dict(m)


def test_save_and_load_round_trip(tmp_path):
    m = generate_mine(seed=5, size="small")
    for ext in (".json", ".yaml"):
        path = str(tmp_path / f"m{ext}")
        save_mine(m, path)
        assert mine_to_dict(load_mine(path)) == mine_to_dict(m)


@pytest.mark.parametrize("patch,message", [
    ({"seam": {"thickness_m": 2, "thicknes_m": 3}}, "unknown key"),
    ({"seam": {"thickness_m": "two"}}, "expected a number"),
    ({"layout": {"type": "spiral"}}, "unknown layout type"),
    ({"seam": {"rock": "cheese"}}, "Unknown rock"),
    ({"layout": {"type": "room_and_pillar", "n_entries": 0}}, "n_entries"),
    ({"grid": {"bridge_diagonals": "yes"}}, "true or false"),
    ({"strata": {"beds": [{"z_bottom_m": 5, "z_top_m": 1, "rock": "shale"}]}}, "above"),
    ({"conditions": {"T_rock_C": None}}, "seam.depth_m"),
])
def test_bad_files_say_what_is_wrong(patch, message):
    d = mine_to_dict(TINY_MINE)
    d.update(patch)
    with pytest.raises(MineFileError, match=message):
        mine_from_dict(d)


def test_rock_temperature_from_depth():
    d = mine_to_dict(TINY_MINE)
    d["seam"]["depth_m"] = 400
    d["conditions"]["T_rock_C"] = None
    m = mine_from_dict(d)
    assert m.T_rock_C == pytest.approx(12 + 25 * 0.4)


# ------------------------------------------------------------------ generator
def test_generator_is_reproducible_and_varied():
    a, b, c = generate_mine(3), generate_mine(3), generate_mine(4)
    assert mine_to_dict(a) == mine_to_dict(b)
    assert mine_to_dict(a)["layout"] != mine_to_dict(c)["layout"]


@pytest.mark.parametrize("seed", range(8))
def test_generated_mines_are_connected(seed):
    p = assemble(scenario_from_mine(generate_mine(seed, "small"), sim_days=1))
    assert len(p.graph.path_cells) > 20            # pumps found a route through the workings


# ------------------------------------------------------------------ MineGrid
def test_pumps_follow_the_file_and_must_connect():
    from dataclasses import replace
    m = replace(TINY_MINE, pumps=PumpSpec(inlet_xy=(3, 3), outlet_xy=(57, 39)))
    g = MineGrid(m).build(None)
    assert np.allclose(g.centroids[g.inlet_cell, :2], (3, 3))
    assert np.allclose(g.centroids[g.outlet_cell, :2], (57, 39))
    sealed = replace(m, layout=Composite(parts=(m.layout,), holes=(Rectangle(origin=(20, -1), length_m=8, width_m=50),)))
    with pytest.raises(MineGeometryError, match="not connected"):
        MineGrid(sealed).build(None)


def test_layered_rock_lands_in_the_right_layers():
    m = generate_mine(1, "small")
    p = assemble(scenario_from_mine(m, sim_days=1))
    g, rocks = p.graph, p.scenario.ground.geology.rocks_at(p.graph.centroids)
    seam = np.abs(g.centroids[:, 2]) < m.seam.thickness_m / 2
    assert {rocks[i].name for i in np.nonzero(seam & ~g.is_drift)[0]} == {"coal"}
    assert np.all(g.is_drift <= seam)               # water only in the seam layer
