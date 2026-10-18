"""3-D data layer, ParaView export, and the interactive 3-D view (needs pyvista)."""
import os
import shutil
import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from conftest import ALL_GRIDS, TINY_MINE  # noqa: E402
from mine_sandbox import CoupledStepper, ConstantInlet, FiniteVolumeModel, assemble  # noqa: E402
from mine_sandbox.mines import scenario_from_mine  # noqa: E402
from mine_sandbox.viz3d import (Not3DError, VolumeData, export_paraview_series,  # noqa: E402
                                supports_3d)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def warmed():
    """The tiny mine after 20 days at a fixed 35 °C inlet."""
    p = assemble(scenario_from_mine(TINY_MINE, sim_days=20))
    st = CoupledStepper(FiniteVolumeModel(p), ConstantInlet(35.0))
    while st.step() is not None:
        pass
    return p, st.model.temperatures()


@pytest.mark.parametrize("name", ALL_GRIDS)
def test_only_real_3d_grids_are_accepted(name):
    g = ALL_GRIDS[name].build(None if name == "mine" else __import__("mine_sandbox").DriftSpec())
    if name in ("radial", "slab2d"):
        assert not supports_3d(g)
        with pytest.raises(Not3DError):
            VolumeData(g, 13.0)
    else:
        assert supports_3d(g)


def test_grids_carry_the_field_exactly(warmed):
    p, T = warmed
    vd = VolumeData(p.graph, 13.0)
    vd.update(T)
    lat = p.graph.lattice
    assert vd.rect.n_cells == np.prod(lat.shape)
    # every rect cell holds the temperature of the node occupying that voxel
    assert np.allclose(vd.rect.cell_data["T"], T[lat.voxel_node.ravel(order="F")])
    assert np.allclose(vd.rect.cell_data["dT"], vd.rect.cell_data["T"] - 13.0)
    # water flag matches the graph
    assert vd.rect.cell_data["water"].sum() == p.graph.is_drift[lat.voxel_node].sum()
    # the uniform volume samples stay within the field's range
    assert T.min() - 1e-4 <= vd.image.point_data["T"].min() and vd.image.point_data["T"].max() <= T.max() + 1e-4


def test_vertical_exaggeration_scales_only_z(warmed):
    p, _ = warmed
    a, b = VolumeData(p.graph, 13.0), VolumeData(p.graph, 13.0, z_scale=4.0)
    assert np.allclose(a.bounds[:4], b.bounds[:4])
    assert np.isclose(b.bounds[5] - b.bounds[4], 4 * (a.bounds[5] - a.bounds[4]))
    assert b.true_bounds == a.bounds


def test_cuts_and_slices_show_the_heat(warmed):
    p, T = warmed
    vd = VolumeData(p.graph, 13.0)
    vd.update(T)
    x0, x1, y0, y1 = vd.bounds[:4]
    cut = vd.cut((0, 1, 0.2), ((x0 + x1) / 2, (y0 + y1) / 2, 0))
    assert cut.n_cells > 0 and cut["T"].max() > 25 and cut["T"].min() >= 13 - 1e-4
    k = p.graph.lattice.voxel_of(p.graph.inlet_cell)[2]
    sheet = vd.axis_slice("z", k)
    vd.fill_sheet(sheet, T)
    assert np.isclose(sheet["T"].max(), T[p.graph.lattice.layer(k)].max())


def test_paraview_series(warmed, tmp_path):
    p, T = warmed
    vd = VolumeData(p.graph, 13.0)
    frames = [(1.0, np.full(p.graph.n_cells, 13.0)), (20.0, T)]
    pvd = export_paraview_series(vd, frames, str(tmp_path), "tiny")
    entries = ET.parse(pvd).getroot().findall(".//DataSet")
    assert [float(e.get("timestep")) for e in entries] == [1.0, 20.0]
    last = pv.read(os.path.join(tmp_path, entries[-1].get("file")))
    assert set(last.array_names) >= {"T", "dT", "water"}
    assert np.isclose(last["T"].max(), T.max(), atol=1e-4)


def _display_runner():
    """How to get an OpenGL display: none needed, run under xvfb-run, or None if neither is possible."""
    if os.environ.get("DISPLAY"):
        return []
    if shutil.which("xvfb-run"):
        return ["xvfb-run", "-a", "-s", "-screen 0 1600x1000x24"]
    return None


@pytest.mark.skipif(_display_runner() is None, reason="needs a display (or xvfb-run) for OpenGL")
def test_interactive_3d_view_end_to_end(tmp_path):
    """Run in its own process on a real (or virtual) display: build a mine, open the 3-D
    tab, cut it, show slices, change exaggeration and colours, keep simulating, save images."""
    pytest.importorskip("pyvistaqt")
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {ROOT!r}); sys.path.insert(0, {os.path.join(ROOT, 'tests')!r})
        from PySide6.QtWidgets import QApplication
        app = QApplication(sys.argv[:1])
        from conftest import TINY_MINE
        from mine_sandbox.gui.main_window import MainWindow
        from mine_sandbox.gui.app import close_and_flush
        import numpy as np, imageio.v3 as iio
        w = MainWindow(threaded=False); w.resize(1400, 900); w.show()
        w.settings.set_mine(TINY_MINE); w.settings.request_build(); w.run_steps(30); app.processEvents()
        w.tabs.setCurrentWidget(w.volume); app.processEvents()
        v = w.volume
        assert v.plotter is not None and v.data is not None, v.placeholder.text()
        v.chk_cut.setChecked(True); v.chk_slices.setChecked(True); app.processEvents()
        v.exag.setValue(5.0); v.cmap.setCurrentIndex(v.cmap.findData("turbo")); app.processEvents()
        w.run_steps(10); app.processEvents()
        v.sz.setValue(0); app.processEvents()
        assert w.k == 0, "3-D slider should move the maps' layer"
        out = {str(tmp_path / 'v.png')!r}
        v.save_image(out)
        img = iio.imread(out)
        assert img.std() > 10, "3-D image looks blank"
        close_and_flush(app, w)
        print("OK")
    """)
    env = dict(os.environ, QT_QPA_PLATFORM="xcb")
    r = subprocess.run(_display_runner() + [sys.executable, "-c", script], capture_output=True, text=True,
                       timeout=240, env=env)
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout[-2000:] + r.stderr[-3000:]
