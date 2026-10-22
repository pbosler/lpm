#!/usr/bin/env python3
"""Publication plots for the DFS-BVE Rossby-Haurwitz 54 sweep (rh54_trials.sh).

Reads the data written by scripts/dfs_bve/rh54_trials.sh:

  <outdir>/timing_summary.csv   one row per run; exit_status != 0 means the
                                run crashed or timed out (124/137)
  <outdir>/failed_runs.log      human-readable record of failed runs (unused)
  <outdir>/<run_id>_<seed><depth>dt<dt>_nlon<nlon>_<remesh>_.m
                                MATLAB-style vectors written at the END of a
                                run by examples/dfs_bve_rh54*.cpp: time,
                                zeta_l1err, zeta_l2err, zeta_linferr, ...
                                (errors are relative; index 0 is a placeholder)

with <run_id> = dfs_bve_rh54_{rk2|rk3|rk4}_g{g}_eps{eps}_d{d}_nlon{nlon}_tf{tf}_nsteps{n}_nthr{T}.

Runs without a .m file (crashed, timed out, still running), runs with a
non-zero exit status in timing_summary.csv, and runs whose error at t_eval is
NaN/inf are skipped and counted in the summary printed at the end.

Figures (one set per norm):
  convergence_<norm>.png            error vs dt, one panel per solver, color =
                                    GMLS order, marker/linestyle = tree depth
  convergence_<norm>_by_depth.png   same data, rows = tree depth, cols = solver
  error_vs_gmls_order_<norm>.png    error vs GMLS order at fixed nsteps
  error_vs_nlon_<norm>.png          error vs nlon at fixed nsteps

Example:
  python3 scripts/dfs_bve/plot_rh54.py --outdir /data/pabosle/dfs-bve-paper \
      --figdir figs --norms l2 linf
"""
import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_OUTDIR = "/data/pabosle/dfs-bve-paper"

RK_ORDER = {"rk2": 2, "rk3": 3, "rk4": 4}
RK_LABEL = {"rk2": "RK2", "rk3": "RK3", "rk4": "RK4"}
NORM_KEY = {"l1": "zeta_l1err", "l2": "zeta_l2err", "linf": "zeta_linferr"}
NORM_LABEL = {
    "l1": r"$\ell_1$",
    "l2": r"$\ell_2$",
    "linf": r"$\ell_\infty$",
}

RUN_RE = re.compile(
    r"dfs_bve_rh54_(?P<rk>rk\d)_g(?P<g>\d+)_eps(?P<eps>[0-9.eE+-]+)"
    r"_d(?P<d>\d+)_nlon(?P<nlon>\d+)_tf(?P<tf>[0-9.eE+-]+)"
    r"_nsteps(?P<nsteps>\d+)_nthr(?P<nthr>\d+)"
)
VEC_RE = re.compile(r"^\s*(\w+)\s*=\s*\[(.*)\]\s*;\s*$")


class Run:
    __slots__ = ("rk", "g", "eps", "d", "nlon", "tf", "nsteps", "path", "mtime")

    def __init__(self, m, path):
        self.rk = m["rk"]
        self.g = int(m["g"])
        self.eps = float(m["eps"])
        self.d = int(m["d"])
        self.nlon = int(m["nlon"])
        self.tf = float(m["tf"])
        self.nsteps = int(m["nsteps"])
        self.path = path
        self.mtime = path.stat().st_mtime

    @property
    def key(self):
        return (self.rk, self.g, self.d, self.nlon, self.nsteps)

    @property
    def dt(self):
        return self.tf / self.nsteps


def read_matlab_vectors(path):
    """Parse 'name = [a,b,c];' lines into numpy arrays (nan/inf allowed)."""
    out = {}
    with open(path) as f:
        for line in f:
            m = VEC_RE.match(line)
            if not m:
                continue
            vals = [v for v in m.group(2).split(",") if v.strip()]
            out[m.group(1)] = np.array([float(v) for v in vals])
    return out


def failed_run_ids(outdir):
    """Run ids (basename of output_root) with a non-zero exit status."""
    bad = set()
    csv_path = outdir / "timing_summary.csv"
    if not csv_path.exists():
        return bad
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                status = int(row["exit_status"])
            except (KeyError, ValueError):
                continue
            if status != 0:
                bad.add(Path(row["output_root"]).name)
    return bad


def discover_runs(outdir):
    """Find one .m file per run; returns (runs, n_failed, n_dupe)."""
    bad = failed_run_ids(outdir)
    by_key = {}
    n_failed = n_dupe = 0
    for path in sorted(outdir.glob("dfs_bve_rh54_*.m")):
        m = RUN_RE.match(path.name)
        if not m:
            continue
        if m.group(0) in bad:
            n_failed += 1
            continue
        run = Run(m, path)
        old = by_key.get(run.key)
        if old is not None:
            n_dupe += 1
            if old.mtime >= run.mtime:
                continue
        by_key[run.key] = run
    return list(by_key.values()), n_failed, n_dupe


def error_at(run, norm, t_eval):
    """Error at the stored time closest to t_eval; None if unusable."""
    data = read_matlab_vectors(run.path)
    t = data.get("time")
    e = data.get(NORM_KEY[norm])
    if t is None or e is None or len(t) != len(e) or len(t) < 2:
        return None
    # index 0 is the (unfilled) initial slot; search the stepped entries only
    i = 1 + int(np.argmin(np.abs(t[1:] - t_eval)))
    if abs(t[i] - t_eval) > 0.5 * run.dt + 1e-12:
        return None  # run did not reach t_eval
    val = e[i]
    return float(val) if np.isfinite(val) and val > 0 else None


def load_errors(runs, norm, t_eval):
    """dict: run.key -> (run, error); unusable runs are dropped."""
    out = {}
    n_bad = 0
    for run in runs:
        err = error_at(run, norm, t_eval)
        if err is None:
            n_bad += 1
        else:
            out[run.key] = (run, err)
    return out, n_bad


# ---------------------------------------------------------------------------
# plotting helpers
# ---------------------------------------------------------------------------

MARKERS = ["o", "s", "^", "D", "v", "P"]
LINESTYLES = ["-", "--", ":", "-."]


def style_maps(values_g, values_d):
    cmap = plt.get_cmap("viridis")
    gs = sorted(values_g)
    ds = sorted(values_d)
    gcol = {g: cmap(0.9 * i / max(len(gs) - 1, 1)) for i, g in enumerate(gs)}
    dmark = {d: MARKERS[i % len(MARKERS)] for i, d in enumerate(ds)}
    dls = {d: LINESTYLES[i % len(LINESTYLES)] for i, d in enumerate(ds)}
    return gcol, dmark, dls


def pick_nlon(runs, requested):
    """Requested nlon if given, else the largest nlon present."""
    nlons = sorted({r.nlon for r in runs})
    if requested is not None:
        if requested not in nlons:
            sys.exit(f"nlon={requested} not found in data (available: {nlons})")
        return requested
    return nlons[-1]


def add_slope(ax, order, x0, y0, xs):
    """Dashed grey reference line y = y0 (x/x0)^order, labelled."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.plot(xs, y0 * (xs / x0) ** order, color="0.5", lw=1, ls=(0, (4, 3)),
            zorder=0)
    ax.text(xs[-1], y0, f" $O(\\Delta t^{order})$", color="0.4", fontsize=8,
            va="bottom", ha="right")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def group(items, keyfn):
    out = defaultdict(list)
    for it in items:
        out[keyfn(it)].append(it)
    return out


def legend_handles(gcol, dmark, dls, g_label="GMLS order"):
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=gcol[g], lw=2, label=f"{g_label} {g}")
          for g in sorted(gcol)]
    hs += [Line2D([], [], color="k", marker=dmark[d], ls=dls[d], lw=1.2,
                  ms=5, label=f"depth {d}") for d in sorted(dmark)]
    return hs


def plot_convergence(errs, norm, nlon, t_eval, figdir, ext):
    items = [(r, e) for (r, e) in errs.values() if r.nlon == nlon]
    if not items:
        print(f"[convergence] no data for nlon={nlon}; skipped")
        return
    rks = sorted({r.rk for r, _ in items})
    gcol, dmark, dls = style_maps({r.g for r, _ in items},
                                  {r.d for r, _ in items})
    ylabel = f"{NORM_LABEL[norm]} relative vorticity error at $t={t_eval:g}$"

    def draw(ax, sub, color_by_g=True):
        for (g, d), pts in sorted(group(sub, lambda p: (p[0].g, p[0].d)).items()):
            pts.sort(key=lambda p: p[0].dt)
            ax.loglog([p[0].dt for p in pts], [p[1] for p in pts],
                      color=gcol[g], marker=dmark[d], ls=dls[d], ms=4, lw=1.2)

    pending = []  # reference slopes are added after all panels are drawn

    def decorate(ax, rk, sub):
        dts = np.array(sorted({r.dt for r, _ in sub}))
        es = np.array([e for _, e in sub])
        if len(dts) > 1:
            xs = np.array([dts.min(), dts.max()])
            # anchor the reference slope at the median error at the largest dt
            big = [e for r, e in sub if r.dt == xs[1]]
            pending.append((ax, RK_ORDER[rk], xs[1], float(np.median(big)), xs))
        ax.set_title(RK_LABEL[rk])
        ax.set_xlabel(r"time step $\Delta t$")
        ax.grid(True, which="both", alpha=0.25)

    # (1) one panel per solver
    fig, axes = plt.subplots(1, len(rks), figsize=(4.2 * len(rks), 4.2),
                             sharey=True, squeeze=False)
    for ax, rk in zip(axes[0], rks):
        sub = [p for p in items if p[0].rk == rk]
        draw(ax, sub)
        decorate(ax, rk, sub)
    axes[0][0].set_ylabel(ylabel)
    for args in pending:
        add_slope(*args)
    pending.clear()
    fig.suptitle(f"RH54 convergence, nlon = {nlon}", y=1.0)
    fig.legend(handles=legend_handles(gcol, dmark, dls), loc="lower center",
               ncol=min(len(gcol) + len(dmark), 7), frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, figdir / f"convergence_{norm}", ext)

    # (2) rows = depth, cols = solver
    ds = sorted({r.d for r, _ in items})
    fig, axes = plt.subplots(len(ds), len(rks),
                             figsize=(4.0 * len(rks), 3.2 * len(ds)),
                             sharex=True, sharey=True, squeeze=False)
    for i, d in enumerate(ds):
        for j, rk in enumerate(rks):
            ax = axes[i][j]
            sub = [p for p in items if p[0].rk == rk and p[0].d == d]
            draw(ax, sub)
            decorate(ax, rk, sub)
            ax.set_title(f"{RK_LABEL[rk]}, depth {d}")
            if i < len(ds) - 1:
                ax.set_xlabel("")
        axes[i][0].set_ylabel(ylabel)
    for args in pending:
        add_slope(*args)
    fig.suptitle(f"RH54 convergence, nlon = {nlon}", y=1.0)
    hs = [h for h in legend_handles(gcol, dmark, dls) if h.get_label().startswith("GMLS")]
    fig.legend(handles=hs, loc="lower center", ncol=len(hs), frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, figdir / f"convergence_{norm}_by_depth", ext)


def plot_vs_gmls_order(errs, norm, nlon, nsteps, t_eval, figdir, ext):
    items = [(r, e) for (r, e) in errs.values()
             if r.nlon == nlon and r.nsteps == nsteps]
    if not items:
        print(f"[gmls order] no data for nlon={nlon}, nsteps={nsteps}; skipped")
        return
    rks = sorted({r.rk for r, _ in items})
    ds = sorted({r.d for r, _ in items})
    _, dmark, _ = style_maps(set(), ds)
    dcol = {d: plt.get_cmap("viridis")(0.9 * i / max(len(ds) - 1, 1))
            for i, d in enumerate(ds)}
    fig, axes = plt.subplots(1, len(rks), figsize=(4.2 * len(rks), 4.0),
                             sharey=True, squeeze=False)
    for ax, rk in zip(axes[0], rks):
        for d in ds:
            pts = sorted([(r.g, e) for r, e in items if r.rk == rk and r.d == d])
            if pts:
                ax.semilogy(*zip(*pts), color=dcol[d], marker=dmark[d], lw=1.2,
                            ms=5, label=f"depth {d}")
        ax.set_title(RK_LABEL[rk])
        ax.set_xlabel("GMLS order")
        ax.set_xticks(sorted({r.g for r, _ in items}))
        ax.grid(True, which="both", alpha=0.25)
    axes[0][0].set_ylabel(f"{NORM_LABEL[norm]} relative vorticity error at $t={t_eval:g}$")
    axes[0][-1].legend(frameon=False)
    fig.suptitle(f"RH54 error vs GMLS order, nsteps = {nsteps}, nlon = {nlon}", y=1.0)
    fig.tight_layout()
    save(fig, figdir / f"error_vs_gmls_order_{norm}", ext)


def plot_vs_nlon(errs, norm, nsteps, t_eval, figdir, ext):
    items = [(r, e) for (r, e) in errs.values() if r.nsteps == nsteps]
    if not items:
        print(f"[nlon] no data for nsteps={nsteps}; skipped")
        return
    rks = sorted({r.rk for r, _ in items})
    gcol, dmark, dls = style_maps({r.g for r, _ in items},
                                  {r.d for r, _ in items})
    fig, axes = plt.subplots(1, len(rks), figsize=(4.2 * len(rks), 4.2),
                             sharey=True, squeeze=False)
    for ax, rk in zip(axes[0], rks):
        sub = [p for p in items if p[0].rk == rk]
        for (g, d), pts in sorted(group(sub, lambda p: (p[0].g, p[0].d)).items()):
            pts.sort(key=lambda p: p[0].nlon)
            ax.semilogy([p[0].nlon for p in pts], [p[1] for p in pts],
                        color=gcol[g], marker=dmark[d], ls=dls[d], ms=4, lw=1.2)
        ax.set_title(RK_LABEL[rk])
        ax.set_xlabel("nlon")
        ax.set_xticks(sorted({r.nlon for r, _ in items}))
        ax.grid(True, which="both", alpha=0.25)
    axes[0][0].set_ylabel(f"{NORM_LABEL[norm]} relative vorticity error at $t={t_eval:g}$")
    fig.suptitle(f"RH54 error vs DFS grid resolution, nsteps = {nsteps}", y=1.0)
    fig.legend(handles=legend_handles(gcol, dmark, dls), loc="lower center",
               ncol=min(len(gcol) + len(dmark), 7), frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, figdir / f"error_vs_nlon_{norm}", ext)


def save(fig, stem, exts):
    for e in exts:
        p = stem.with_suffix("." + e)
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print("wrote", p)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR,
                    help="rh54_trials.sh OUTDIR (default: %(default)s)")
    ap.add_argument("--figdir", default="rh54_figs", help="where to write figures")
    ap.add_argument("--norms", nargs="+", default=["l2", "linf"],
                    choices=sorted(NORM_KEY), help="error norms to plot")
    ap.add_argument("--t-eval", type=float, default=2.0, help="error evaluation time")
    ap.add_argument("--nlon", type=int, default=None,
                    help="nlon held fixed in the convergence and GMLS-order plots "
                         "(default: largest available)")
    ap.add_argument("--nsteps", type=int, default=2000,
                    help="nsteps held fixed in the GMLS-order and nlon plots")
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"])
    args = ap.parse_args()

    outdir = Path(args.outdir)
    if not outdir.is_dir():
        sys.exit(f"output directory not found: {outdir}")
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    runs, n_failed, n_dupe = discover_runs(outdir)
    if not runs:
        sys.exit(f"no completed runs found in {outdir}")
    nlon = pick_nlon(runs, args.nlon)
    print(f"found {len(runs)} completed runs ({n_failed} failed runs skipped, "
          f"{n_dupe} duplicate runs resolved to newest); fixed nlon = {nlon}")

    for norm in args.norms:
        errs, n_bad = load_errors(runs, norm, args.t_eval)
        print(f"[{norm}] {len(errs)} usable runs, {n_bad} dropped "
              f"(did not reach t={args.t_eval:g}, or NaN/inf/zero error)")
        plot_convergence(errs, norm, nlon, args.t_eval, figdir, args.formats)
        plot_vs_gmls_order(errs, norm, nlon, args.nsteps, args.t_eval, figdir,
                           args.formats)
        plot_vs_nlon(errs, norm, args.nsteps, args.t_eval, figdir, args.formats)


if __name__ == "__main__":
    main()
