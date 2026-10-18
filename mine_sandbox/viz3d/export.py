"""Write a run as a ParaView time series: one .vtr per frame plus a .pvd index.

Open the .pvd in ParaView (File > Open) and press play; every frame has cell
data T (°C), dT (rise above rock) and water (1 in the workings), on the
model's real cell edges.
"""
from __future__ import annotations

import os
from xml.sax.saxutils import quoteattr

from .volume_data import VolumeData


def export_paraview_series(volume: VolumeData, frames, folder: str, name: str = "mine") -> str:
    """Write every frame as a .vtr file plus a .pvd index that ties them into a time series.

    `frames` yields (t_d, field). Returns the path of the .pvd file.
    """
    os.makedirs(folder, exist_ok=True)
    entries = []
    for n, (t_d, field) in enumerate(frames):
        fname = f"{name}_{n:05d}.vtr"
        volume.snapshot(field, t_d).save(os.path.join(folder, fname))
        entries.append(f'    <DataSet timestep="{t_d:.6g}" part="0" file={quoteattr(fname)}/>')
    pvd = os.path.join(folder, f"{name}.pvd")
    with open(pvd, "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n  <Collection>\n')
        f.write("\n".join(entries))
        f.write("\n  </Collection>\n</VTKFile>\n")
    return pvd


def history_frames(history):
    """(t_d, field) for every stored frame of a live History."""
    for i in range(history.n_frames):
        yield history.frame(i)


def result_frames(result, every: int = 1):
    """(t_d, field) from a RunResult, every `every`-th step."""
    for i in range(0, len(result.times_d), every):
        yield float(result.times_d[i]), result.temperatures[i]
