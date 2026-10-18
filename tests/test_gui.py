"""Headless GUI tests (Qt's offscreen platform). Skipped if PySide6/pyqtgraph are missing."""
import os
import time

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from conftest import TINY_MINE  # noqa: E402

EXAMPLE = os.path.join(os.path.dirname(__file__), "..", "examples", "mines", "single_panel.json")


@pytest.fixture(scope="module")
def app():
    """One QApplication for the whole test module, on Qt's offscreen platform."""
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app):
    """An inline (synchronous) main window, closed cleanly after each test."""
    from mine_sandbox.gui.app import close_and_flush
    from mine_sandbox.gui.main_window import MainWindow
    w = MainWindow(threaded=False)
    w.resize(1400, 900)
    w.show()
    yield w
    close_and_flush(app, w)


def test_mine_file_builds_and_plays(window, app):
    window.settings.load_mine_file(EXAMPLE)
    window.settings.request_build()
    assert window.info is not None, window.banner.text()
    window.run_steps(40)
    app.processEvents()
    assert window.history.n_steps == 40
    img = window.plan.img.image
    assert img is not None and np.nanmax(img) > np.nanmin(img) + 5      # the workings show up hot
    assert "day 20.0" in window.time_label.text()


def test_layer_selection_and_clicks_move_the_slices(window, app):
    window.settings.set_mine(TINY_MINE)
    window.settings.request_build()
    window.run_steps(10)
    lat = window.info.problem.graph.lattice
    window.layers.select_layer(0)
    assert window.k == 0 and "layer" in window.plan.plot.getPlotItem().titleLabel.text
    node = int(lat.layer(lat.shape[2] - 1)[2, 3])
    window._clicked(node, Qt.LeftButton, "section")              # clicking a height picks that layer
    assert window.k == lat.shape[2] - 1
    window._clicked(node, Qt.RightButton, "plan")                # right click pins a probe
    assert list(window.probes.values()) == [node]
    window._clicked(node, Qt.RightButton, "plan")                # and again removes it
    assert not window.probes


def test_replay_and_export(window, app, tmp_path):
    window.settings.set_mine(TINY_MINE)
    window.settings.request_build()
    window._toggle_probe(int(window.info.problem.graph.wall_profile()[1][1]))
    window.run_steps(60)
    window._seek_time(10.0)
    assert not window.follow_live and "replay" in window.time_label.text()
    window._go_live()
    path = str(tmp_path / "run.csv")
    window._export_csv(path)
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    assert data.shape == (60, 6)
    assert os.path.exists(path.replace(".csv", "_probes.csv"))


def test_bad_settings_show_an_error_instead_of_crashing(window):
    window.settings.source.setCurrentIndex(1)                   # "Mine file" with no mine loaded
    window.settings.request_build()
    assert window.info is None and window.banner.isVisible()
    assert "open or generate a mine" in window.banner.text()


def test_threaded_live_run(app):
    """The real configuration: the simulation runs in its own process while the window draws."""
    from mine_sandbox.gui.app import close_and_flush
    from mine_sandbox.gui.main_window import MainWindow
    w = MainWindow(threaded=True)
    try:
        w.show()
        w.settings.set_mine(TINY_MINE)
        w.speed.setCurrentIndex(w.speed.findData(30))
        w.settings.request_build()
        t0 = time.time()
        while w.info is None and time.time() - t0 < 30:
            app.processEvents()
        assert w.info is not None
        w._toggle_play()
        t0 = time.time()
        while time.time() - t0 < 1.5:
            app.processEvents()
            time.sleep(0.01)
        w._toggle_play()                                        # pause
        t0 = time.time()
        while w.playing and time.time() - t0 < 5:
            app.processEvents()
        days = w._last_tick.t_d
        assert 15 < days < 70                                   # ~30 simulated days per second
    finally:
        close_and_flush(app, w)


def test_3d_tab_explains_itself_without_a_display(window, app):
    window.settings.set_mine(TINY_MINE)
    window.settings.request_build()
    window.tabs.setCurrentWidget(window.volume)
    app.processEvents()
    assert window.volume.plotter is None
    assert "display" in window.volume.placeholder.text() or "PyVista" in window.volume.placeholder.text()


def test_paraview_export_from_the_window(window, app, tmp_path):
    pytest.importorskip("pyvista")
    window.settings.set_mine(TINY_MINE)
    window.settings.request_build()
    window.run_steps(12)
    pvd = window._export_paraview(str(tmp_path))
    assert pvd and os.path.exists(pvd)
    assert len([f for f in os.listdir(tmp_path) if f.endswith(".vtr")]) == window.history.n_frames


def _spin(app, cond, timeout=60.0):
    """Keep the Qt event loop running until a condition holds or the timeout passes; returns the
    condition.
    """
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.01)
    return cond()


def test_progress_bar_follows_build_and_run(window, app):
    window.settings.set_mine(TINY_MINE)
    window.settings.request_build()
    assert window.progress.isVisible() and "day 0" in window.progress.format()
    assert "built in" in window.timing.text()
    assert "compute:" in window.settings.stats.text() and "built in" in window.settings.stats.text()
    window.run_steps(20)
    app.processEvents()
    assert window.progress.value() > 0 and "ms/step" in window.timing.text()
    assert window.speed.findData(float("inf")) >= 0                       # "max" speed is offered


def test_process_window_end_to_end(app):
    """The default window: devices listed, staged build with progress, a run at full speed."""
    from mine_sandbox.gui.app import close_and_flush
    from mine_sandbox.gui.main_window import MainWindow
    from mine_sandbox.live import MAX_SPEED
    w = MainWindow()
    assert w.backend == "process"
    try:
        w.show()
        assert _spin(app, lambda: w.worker.devices is not None)
        labels = [w.settings.device.itemText(i) for i in range(w.settings.device.count())]
        assert labels[0].startswith("Auto") and "CPU" in labels
        progress = []
        w.worker.progress.connect(progress.append)
        w.settings.set_mine(TINY_MINE)
        w.settings.days.setValue(30)
        w.speed.setCurrentIndex(w.speed.findData(MAX_SPEED))
        w.settings.request_build()
        assert w.progress.isVisible()
        assert _spin(app, lambda: w.info is not None), w.banner.text()
        assert progress and progress[-1].finished
        w._toggle_play()
        assert _spin(app, lambda: w._last_tick is not None and w._last_tick.done)
        _spin(app, lambda: w.history.n_steps == 60, timeout=5)
        assert w.history.n_steps == 60 and "Run complete" in w.banner.text()
        assert w.progress.value() == 1000
        pid = w.worker.pid
    finally:
        close_and_flush(app, w)
    import os
    import signal
    try:
        os.kill(pid, 0)                                                     # the process must be gone
        alive = True
    except (OSError, ProcessLookupError):
        alive = False
    if alive:                                                               # zombie until reaped is fine
        os.kill(pid, signal.SIGKILL)


def test_cancelling_a_build_restarts_the_process(app):
    from mine_sandbox.gui.app import close_and_flush
    from mine_sandbox.gui.main_window import MainWindow
    from mine_sandbox.mines import generate_mine
    w = MainWindow()
    try:
        w.show()
        assert _spin(app, lambda: w.worker.devices is not None)
        first = w.worker.pid
        w.settings.set_mine(generate_mine(seed=1, size="large"))
        w.settings.request_build()
        assert w.cancel_btn.isVisibleTo(w)
        w._cancel_build()
        assert "cancelled" in w.status.text() and not w.progress.isVisible()
        assert w.worker.pid != first
        w.settings.set_mine(TINY_MINE)
        w.settings.request_build()
        assert _spin(app, lambda: w.info is not None), w.banner.text()
    finally:
        close_and_flush(app, w)
