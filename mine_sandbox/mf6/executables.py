"""Locate the MODFLOW 6 executable and shared library."""
from __future__ import annotations

import os
import shutil

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SEARCH_DIRS = (os.path.join(_PROJECT_ROOT, "bin"), os.path.expanduser("~/bin"), os.getcwd())


def find_mf6() -> str:
    """mf6 executable: env MF6_EXE, then PATH, then ./bin, ~/bin, cwd."""
    exe = os.environ.get("MF6_EXE")
    if exe and os.path.exists(exe):
        return exe
    for name in ("mf6", "mf6.exe"):
        found = shutil.which(name)
        if found:
            return found
    for d in _SEARCH_DIRS:
        for name in ("mf6", "mf6.exe"):
            if os.path.exists(os.path.join(d, name)):
                return os.path.join(d, name)
    raise FileNotFoundError("MODFLOW 6 executable not found. Run `get-modflow :python` "
                            "or put mf6 in ./bin, or set MF6_EXE.")


def find_libmf6() -> str:
    """libmf6 shared library: env MF6_LIB, then ./bin, ~/bin, cwd."""
    lib = os.environ.get("MF6_LIB")
    if lib and os.path.exists(lib):
        return lib
    for d in _SEARCH_DIRS:
        for name in ("libmf6.so", "libmf6.dll", "libmf6.dylib"):
            if os.path.exists(os.path.join(d, name)):
                return os.path.join(d, name)
    raise FileNotFoundError("libmf6 not found. It ships in the same zip as mf6 "
                            "(github.com/MODFLOW-ORG/executables); put it in ./bin or set MF6_LIB, "
                            "or use Mf6RestartSession, which only needs the executable.")
