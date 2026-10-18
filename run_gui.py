"""Open the live simulator:  python run_gui.py [--mine examples/mines/single_panel.json --build]"""
import multiprocessing
import sys

if __name__ == "__main__":          # required: the simulation process re-imports this file on Windows
    multiprocessing.freeze_support()
    from mine_sandbox.gui.app import main
    sys.exit(main())
