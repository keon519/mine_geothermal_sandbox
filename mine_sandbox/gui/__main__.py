"""Lets the live simulator run with `python -m mine_sandbox.gui`."""
import sys

from .app import main

if __name__ == "__main__":
    sys.exit(main())
