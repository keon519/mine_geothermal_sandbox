"""Entry point for the live simulator.

    python -m mine_sandbox.gui                          # open the app
    python -m mine_sandbox.gui --mine examples/mines/single_panel.json
    python -m mine_sandbox.gui --mine m.json --screenshot out.png --days 60
    python -m mine_sandbox.gui --device cpu             # never use the GPU
    python -m mine_sandbox.gui --backend thread         # old in-process worker (debugging)

The simulation runs in a separate process by default (see live/process.py).
"""
from __future__ import annotations

import argparse
import multiprocessing
import sys


def close_and_flush(app, win) -> None:
    """Close a window (its plots are disposed in closeEvent) and drain pending events."""
    win.close()
    app.processEvents()


def main(argv=None) -> int:
    """Parse the command line, open the main window, and run the Qt event loop.

    With `--screenshot` it instead builds the model, simulates the requested number of days
    in-process, saves a picture and exits, which is handy for documentation and for checking the
    GUI on a machine without a display.
    """
    ap = argparse.ArgumentParser(description="Mine-water heat sink — live simulator")
    ap.add_argument("--mine", help="mine file to load (.json / .yaml)")
    ap.add_argument("--build", action="store_true", help="build the model immediately")
    ap.add_argument("--probe", type=int, action="append", default=[], help="cell id to probe (repeatable)")
    ap.add_argument("--screenshot", help="render headlessly, save a PNG, and exit")
    ap.add_argument("--days", type=float, default=60.0, help="days to simulate before --screenshot")
    ap.add_argument("--layer", type=int, help="layer index to show")
    ap.add_argument("--device", help="compute device to select: auto, cpu, cuda:0, ...")
    ap.add_argument("--backend", choices=("process", "thread"), default="process",
                    help="where the simulation runs while the window is open (default: its own process)")
    args = ap.parse_args(argv)
    multiprocessing.freeze_support()

    from PySide6.QtWidgets import QApplication
    from .main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])
    win = MainWindow(backend="inline" if args.screenshot else args.backend)
    if args.mine:
        win.settings.load_mine_file(args.mine)
    if args.device:
        i = win.settings.device.findData(args.device)
        if i < 0:                        # a GPU not detected yet: keep the request
            win.settings.device.addItem(args.device, args.device)
            i = win.settings.device.count() - 1
        win.settings.device.setCurrentIndex(i)
    if args.build or args.screenshot:
        win.settings.request_build()
    if args.screenshot:
        if win.info is None:
            print(win.banner.text())
            return 1
        if args.layer is not None:
            win.layers.select_layer(args.layer)
        for node in args.probe:
            win._toggle_probe(node)
        steps = int(round(args.days / win.info.problem.scenario.numerics.dt_days))
        win.resize(1600, 1000)
        win.show()
        win.run_steps(steps)
        app.processEvents()
        win._save_png(args.screenshot)
        close_and_flush(app, win)
        return 0
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
