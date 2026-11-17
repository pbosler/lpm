#!/usr/bin/env python3
"""Compare dfs_bve_gauss_vort runs against a reference run, using the grid
vorticity snapshots in the .m files (zeta_grid, zeta_grid_times, grid_nlat,
grid_nlon).

DFS grids are nested when nlon is multiplied by an integer k: colatitude
i*pi/(nlat-1) and longitude 2*pi*j/nlon of a run are the reference's points
(k*i, k*j). Errors are measured on the run's grid, at the snapshot times
both files share, with Clenshaw-Curtis (in z) x trapezoid (in longitude)
quadrature, the same rule as DFSGrid::quadrature_weight:
    l2   = ||zeta - zeta_ref||_2 / ||zeta_ref||_2
    linf = max |zeta - zeta_ref| / max |zeta_ref|

usage: gauss_vort_compare.py REF.m RUN.m [RUN.m ...] [--csv out.csv]
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_rh54 import read_matlab_vectors  # noqa: E402


def clenshaw_curtis_weights(n):
    """weights for int_{-1}^{1} f(z) dz at z_i = cos(i pi / n), i = 0..n"""
    theta = np.pi * np.arange(n + 1) / n
    w = np.zeros(n + 1)
    v = np.ones(n - 1)
    if n % 2 == 0:
        w[0] = w[n] = 1.0 / (n**2 - 1)
        for k in range(1, n // 2):
            v -= 2 * np.cos(2 * k * theta[1:-1]) / (4 * k**2 - 1)
        v -= np.cos(n * theta[1:-1]) / (n**2 - 1)
    else:
        w[0] = w[n] = 1.0 / n**2
        for k in range(1, (n - 1) // 2 + 1):
            v -= 2 * np.cos(2 * k * theta[1:-1]) / (4 * k**2 - 1)
    w[1:-1] = 2 * v / n
    return w


def grid_weights(nlat, nlon):
    """nlat x nlon quadrature weights on the unit sphere"""
    return np.outer(clenshaw_curtis_weights(nlat - 1), np.full(nlon, 2 * np.pi / nlon))


def load(path):
    v = read_matlab_vectors(path)
    nlat, nlon = int(v["grid_nlat"][0]), int(v["grid_nlon"][0])
    times = v["zeta_grid_times"]
    zeta = v["zeta_grid"].reshape(len(times), nlat, nlon)
    return times, zeta, nlat, nlon


def compare(ref_path, run_path):
    rt, rz, rnlat, rnlon = load(ref_path)
    t, z, nlat, nlon = load(run_path)
    if rnlon % nlon != 0:
        raise ValueError(f"reference nlon {rnlon} is not a multiple of run nlon {nlon}")
    k = rnlon // nlon
    assert rnlat - 1 == k * (nlat - 1)
    w = grid_weights(nlat, nlon)
    rows = []
    for i, ti in enumerate(t):
        j = np.flatnonzero(np.isclose(rt, ti, rtol=0, atol=1e-9))
        if len(j) == 0:
            continue
        zr = rz[j[0], ::k, ::k]
        diff = z[i] - zr
        rows.append(dict(t=ti,
                         l2=np.sqrt(np.sum(w * diff**2) / np.sum(w * zr**2)),
                         linf=np.max(np.abs(diff)) / np.max(np.abs(zr))))
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("ref")
    p.add_argument("runs", nargs="+")
    p.add_argument("--csv", help="also write rows to this file")
    a = p.parse_args()
    out = []
    for run in a.runs:
        rows = compare(a.ref, run)
        name = Path(run).name.split("_cubed_sphere")[0]
        if not rows:
            print(f"{name}: no snapshot times in common with the reference", file=sys.stderr)
        for r in rows:
            out.append(dict(run=name, **r))
            print(f"{name}  t={r['t']:.4g}  l2={r['l2']:.3e}  linf={r['linf']:.3e}")
    if a.csv:
        with open(a.csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["run", "t", "l2", "linf"])
            wr.writeheader()
            wr.writerows(out)
