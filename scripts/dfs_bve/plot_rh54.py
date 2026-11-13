#!/usr/bin/env python3
"""Publication plots for the DFS-BVE Rossby-Haurwitz 54 sweep (rh54_trials.sh).

Reads the data written by scripts/dfs_bve/rh54_trials.sh:

  <outdir>/timing_summary*.csv  one row per run (rh54_sweep.sh writes one file
                                per machine); exit_status != 0 means the run
                                crashed or timed out (124/137)
  <outdir>/failed_runs.log      human-readable record of failed runs (unused)
  <outdir>/<run_id>_<seed><depth>dt<dt>_nlon<nlon>_<remesh>_.m
                                MATLAB-style vectors written at the END of a
                                run by examples/dfs_bve_rh54*.cpp: time,
                                zeta_l1err, zeta_l2err, zeta_linferr, ...
                                (errors are relative; index 0 is a placeholder)

with <run_id> = dfs_bve_rh54_{rk2|rk3|rk4}_g{g}_eps{eps}_d{d}_nlon{nlon}_tf{tf}_nsteps{n}_nthr{T}.

Runs without a .m file (crashed, timed out, still running), runs with a
non-zero exit status in timing_summary.csv, and runs whose error at t_eval is
NaN/inf are skipped and counted in the summary printed at the end. With
--exclude, runs that rh54_classify_failures.py --include-ok classed as blow-ups
(ok_blowup or blowup) are skipped too: they finish, but their errors are not
meaningful. With --stale RK@TIME (as in rh54_plan.py), timing rows of RK written
before TIME are ignored and an RK run is plotted only if a newer row succeeded,
so .m files left from the stale runs are not used.

Figures (one set per norm):
  convergence_<norm>.png            error vs dt, one panel per solver, color =
                                    GMLS order, marker/linestyle = tree depth
  convergence_<norm>_by_depth.png   same data, rows = tree depth, cols = solver
  error_vs_gmls_order_<norm>.png    spatial error floor (--floor-rk, default
                                    RK4, error at its finest dt) at fixed
                                    nlon, vs GMLS order and vs tree depth;
                                    open markers = not converged in dt (the
                                    error still changes between the two
                                    finest dt)
  error_vs_nlon_<norm>.png          same spatial error floor vs nlon
  work_precision_<norm>.png         error vs wall-clock time (elapsed_seconds
                                    in timing_summary*.csv) at fixed nlon, one
                                    panel per solver, color = GMLS order,
                                    marker = tree depth; points = nsteps
  work_precision_<norm>_by_depth.png  same, one panel per depth, color = solver
  conservation.png                  relative change of kinetic energy and
                                    enstrophy, particle vs DFS grid quadrature
                                    (grid needs runs made with -gi), for one
                                    solver/GMLS order (--cons-rk, --cons-g)

Example:
  python3 scripts/dfs_bve/plot_rh54.py --outdir /gpfs/pabosle/lpm-dfs-bve-paper \
      --figdir figs --norms l2 linf
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rh54_plan  # noqa: E402  (--stale handling)

DEFAULT_OUTDIR = os.environ.get("OUTDIR", "/gpfs/pabosle/lpm-dfs-bve-paper")

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
    __slots__ = ("rk", "g", "eps", "d", "nlon", "tf", "nsteps", "path", "mtime",
                 "wall")

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
        self.wall = None  # elapsed seconds from timing_summary*.csv, if recorded

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


def run_status(outdir, stale):
    """(bad, good, wall) over every timing_summary*.csv (one per machine):
    bad/good are run ids (basename of output_root); bad have a non-zero exit
    status and no successful record. wall maps a good run id to the elapsed
    wall-clock seconds of its newest successful row. Stale rows are ignored."""
    bad, good = set(), set()
    wall = {}  # run id -> (timestamp, elapsed seconds)
    for csv_path in sorted(outdir.glob("timing_summary*.csv")):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    status = int(row["exit_status"])
                except (KeyError, ValueError):
                    continue
                name = Path(row["output_root"]).name
                m = RUN_RE.match(name)
                if m and rh54_plan.is_stale(stale, m["rk"], row.get("timestamp", "")):
                    continue
                (good if status == 0 else bad).add(name)
                if status == 0:
                    ts = row.get("timestamp", "")
                    try:
                        sec = float(row["elapsed_seconds"])
                    except (KeyError, ValueError):
                        continue
                    if name not in wall or ts >= wall[name][0]:
                        wall[name] = (ts, sec)
    return bad - good, good, {k: v[1] for k, v in wall.items()}


def blowup_keys(tsv_path):
    """Run keys classed as blow-ups in rh54_classify_failures.py output."""
    keys = set()
    with open(tsv_path, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["class"] in ("ok_blowup", "blowup"):
                keys.add((row["rk"], int(row["g"]), int(row["d"]),
                          int(row["nlon"]), int(row["nsteps"])))
    return keys


def discover_runs(outdir, exclude=frozenset(), stale=None):
    """Find one .m file per run, skipping keys in exclude and stale runs;
    returns (runs, n_failed, n_stale, n_dupe, n_excluded). n_stale counts
    files of a --stale solver with no successful post-cutoff run (pre-cutoff
    output, e.g. a different nthr in the name, so never overwritten)."""
    stale = stale or {}
    bad, good, wall = run_status(outdir, stale)
    by_key = {}
    n_failed = n_stale = n_dupe = n_excluded = 0
    for path in sorted(outdir.glob("dfs_bve_rh54_*.m")):
        m = RUN_RE.match(path.name)
        if not m:
            continue
        if m.group(0) in bad:
            n_failed += 1
            continue
        if m["rk"] in stale and m.group(0) not in good:
            n_stale += 1
            continue
        run = Run(m, path)
        run.wall = wall.get(m.group(0))
        if run.key in exclude:
            n_excluded += 1
            continue
        old = by_key.get(run.key)
        if old is not None:
            n_dupe += 1
            if old.mtime >= run.mtime:
                continue
        by_key[run.key] = run
    return list(by_key.values()), n_failed, n_stale, n_dupe, n_excluded


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


def add_slope(ax, order, x0, y0, xs, fontsize=8):
    """Dashed grey reference line y = y0 (x/x0)^order, labelled."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.plot(xs, y0 * (xs / x0) ** order, color="0.5", lw=1, ls=(0, (4, 3)),
            zorder=0)
    # label at the line's midpoint (log scale), just below-right of it, which
    # stays clear of the data at the largest dt
    xm = float(np.sqrt(xs[0] * xs[-1]))
    ax.annotate(f"$O(\\Delta t^{order})$", (xm, y0 * (xm / x0) ** order),
                xytext=(4, -4), textcoords="offset points", color="0.4",
                fontsize=fontsize, va="top", ha="left")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def group(items, keyfn):
    out = defaultdict(list)
    for it in items:
        out[keyfn(it)].append(it)
    return out


def legend_handles(gcol, dmark, dls, g_label="GMLS order", ms=5, lw=1.2):
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=gcol[g], lw=2 * lw / 1.2, label=f"{g_label} {g}")
          for g in sorted(gcol)]
    hs += [Line2D([], [], color="k", marker=dmark[d], ls=dls[d], lw=lw,
                  ms=ms, label=f"depth {d}") for d in sorted(dmark)]
    return hs


PAPER_RC = {"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 14,
            "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 13}


def plot_convergence(errs, norm, nlon, t_eval, figdir, ext):
    items = [(r, e) for (r, e) in errs.values() if r.nlon == nlon]
    if not items:
        print(f"[convergence] no data for nlon={nlon}; skipped")
        return
    rks = sorted({r.rk for r, _ in items})
    gcol, dmark, dls = style_maps({r.g for r, _ in items},
                                  {r.d for r, _ in items})
    ylabel = f"{NORM_LABEL[norm]} relative vorticity error at $t={t_eval:g}$"

    def draw(ax, sub, ms=4, lw=1.2):
        for (g, d), pts in sorted(group(sub, lambda p: (p[0].g, p[0].d)).items()):
            pts.sort(key=lambda p: p[0].dt)
            ax.loglog([p[0].dt for p in pts], [p[1] for p in pts],
                      color=gcol[g], marker=dmark[d], ls=dls[d], ms=ms, lw=lw)

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

    # (1) one panel per solver; paper figure: no title (nlon goes in the
    # caption), larger fonts and markers
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, len(rks), figsize=(4.8 * len(rks), 4.8),
                                 sharey=True, squeeze=False)
        for ax, rk in zip(axes[0], rks):
            sub = [p for p in items if p[0].rk == rk]
            draw(ax, sub, ms=7, lw=1.6)
            decorate(ax, rk, sub)
        axes[0][0].set_ylabel(ylabel)
        for args in pending:
            add_slope(*args, fontsize=12)
        pending.clear()
        fig.legend(handles=legend_handles(gcol, dmark, dls, ms=7, lw=1.6),
                   loc="lower center", ncol=min(len(gcol) + len(dmark), 7),
                   frameon=False)
        fig.tight_layout(rect=(0, 0.11, 1, 1))
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


FLOOR_MAX_ORDER = 1.0  # observed order between the two finest dt below this = on the floor


def spatial_floor(errs, rk):
    """Spatial error floor of solver rk: dict (g, d, nlon) -> (err, nsteps, on_floor).

    err is the error at the finest dt (largest nsteps) of that configuration.
    on_floor is False while the error still changes with dt at an observed
    order >= FLOOR_MAX_ORDER between the two finest dt. err is then not
    converged in dt: usually time-limited (an upper bound on the spatial
    error), but it can also be a cancellation dip below the floor, or a
    configuration whose error is noisy in dt.
    """
    out = {}
    sub = [(r, e) for r, e in errs.values() if r.rk == rk]
    for key, pts in group(sub, lambda p: (p[0].g, p[0].d, p[0].nlon)).items():
        pts.sort(key=lambda p: p[0].nsteps)
        (r1, e1) = pts[-1]
        on_floor = True
        if len(pts) > 1:
            (r0, e0) = pts[-2]
            order = np.log(e0 / e1) / np.log(r1.nsteps / r0.nsteps)
            on_floor = order < FLOOR_MAX_ORDER
        out[key] = (e1, r1.nsteps, on_floor)
    return out


def plot_floor_points(ax, xs, ys, flags, color, marker, ls, label=None):
    """Filled markers on the floor, open ones not converged in dt; segments
    touching an open marker are faded so they don't read as a trend."""
    ax.semilogy([], [], color=color, ls=ls, lw=1.2, label=label)
    for i in range(len(xs) - 1):
        alpha = 1.0 if flags[i] and flags[i + 1] else 0.3
        ax.semilogy(xs[i:i + 2], ys[i:i + 2], color=color, ls=ls, lw=1.2, alpha=alpha)
    for x, y, f in zip(xs, ys, flags):
        ax.semilogy([x], [y], ls="none", marker=marker, ms=6, color=color,
                    mfc=color if f else "white", mew=1.3)


def floor_legend_handles():
    from matplotlib.lines import Line2D
    return [Line2D([], [], ls="none", marker="o", color="k", ms=6, label="on spatial floor"),
            Line2D([], [], ls="none", marker="o", color="k", mfc="white", ms=6, mew=1.3,
                   label="not converged in $\\Delta t$")]


def print_floor_table(floor, rk, nlon, norm):
    gs = sorted({g for g, _, n in floor if n == nlon})
    ds = sorted({d for _, d, n in floor if n == nlon})
    print(f"[{norm}] {RK_LABEL[rk]} spatial floor, nlon = {nlon} "
          f"(error at finest dt [nsteps]; * = not converged in dt)")
    print("          " + "".join(f"{'depth ' + str(d):>20s}" for d in ds))
    for g in gs:
        cells = []
        for d in ds:
            v = floor.get((g, d, nlon))
            cells.append(f"{'-':>20s}" if v is None else
                         f"{v[0]:9.2e} [{v[1]:4d}]{' ' if v[2] else '*'}".rjust(20))
        print(f"  GMLS {g}: " + "".join(cells))


def plot_vs_gmls_order(errs, norm, nlon, rk, t_eval, figdir, ext):
    """Spatial error floor at fixed nlon: vs GMLS order (color = depth) and
    vs tree depth (color = GMLS order)."""
    floor = {k: v for k, v in spatial_floor(errs, rk).items() if k[2] == nlon}
    if not floor:
        print(f"[gmls order] no {rk} data for nlon={nlon}; skipped")
        return
    print_floor_table(floor, rk, nlon, norm)
    gs = sorted({g for g, _, _ in floor})
    ds = sorted({d for _, d, _ in floor})
    gcol, dmark, _ = style_maps(gs, ds)
    dcol = {d: plt.get_cmap("viridis")(0.9 * i / max(len(ds) - 1, 1))
            for i, d in enumerate(ds)}
    gmark = {g: MARKERS[i % len(MARKERS)] for i, g in enumerate(gs)}
    fig, (ax_g, ax_d) = plt.subplots(1, 2, figsize=(8.8, 4.0), sharey=True)
    for d in ds:
        pts = [(g, *floor[(g, d, nlon)]) for g in gs if (g, d, nlon) in floor]
        if pts:
            plot_floor_points(ax_g, [p[0] for p in pts], [p[1] for p in pts],
                              [p[3] for p in pts], dcol[d], dmark[d], "-",
                              label=f"depth {d}")
    for g in gs:
        pts = [(d, *floor[(g, d, nlon)]) for d in ds if (g, d, nlon) in floor]
        if pts:
            plot_floor_points(ax_d, [p[0] for p in pts], [p[1] for p in pts],
                              [p[3] for p in pts], gcol[g], gmark[g], "-",
                              label=f"GMLS order {g}")
    ax_g.set_xlabel("GMLS order")
    ax_g.set_xticks(gs)
    ax_d.set_xlabel("tree depth")
    ax_d.set_xticks(ds)
    ax_g.set_ylabel(f"{NORM_LABEL[norm]} relative vorticity error at $t={t_eval:g}$")
    for ax in (ax_g, ax_d):
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
    fig.legend(handles=floor_legend_handles(), loc="lower center", ncol=2, frameon=False)
    fig.suptitle(f"RH54 spatial error ({RK_LABEL[rk]} at finest $\\Delta t$), "
                 f"nlon = {nlon}", y=1.0)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, figdir / f"error_vs_gmls_order_{norm}", ext)


def plot_vs_nlon(errs, norm, rk, t_eval, figdir, ext):
    """Spatial error floor vs nlon, color = GMLS order, marker/linestyle = depth."""
    floor = spatial_floor(errs, rk)
    if not floor:
        print(f"[nlon] no {rk} data; skipped")
        return
    gcol, dmark, dls = style_maps({g for g, _, _ in floor}, {d for _, d, _ in floor})
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for (g, d), keys in sorted(group(floor, lambda k: (k[0], k[1])).items()):
        keys.sort(key=lambda k: k[2])
        plot_floor_points(ax, [k[2] for k in keys], [floor[k][0] for k in keys],
                          [floor[k][2] for k in keys], gcol[g], dmark[d], dls[d])
    ax.set_xlabel("nlon")
    ax.set_xticks(sorted({n for _, _, n in floor}))
    ax.set_ylabel(f"{NORM_LABEL[norm]} relative vorticity error at $t={t_eval:g}$")
    ax.grid(True, which="both", alpha=0.25)
    hs = legend_handles(gcol, dmark, dls) + floor_legend_handles()
    fig.legend(handles=hs, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
    fig.suptitle(f"RH54 spatial error vs DFS grid resolution "
                 f"({RK_LABEL[rk]} at finest $\\Delta t$)", y=1.0)
    fig.tight_layout()
    save(fig, figdir / f"error_vs_nlon_{norm}", ext)


def n_particles(d):
    """Lagrangian particles of the depth-d cubed sphere: 6*4^d faces plus
    6*4^d + 2 vertices (both carry vorticity)."""
    return 12 * 4**d + 2


def plot_work_precision(errs, norm, nlon, t_eval, figdir, ext):
    """Error vs wall-clock time (timing_summary*.csv elapsed_seconds) at fixed
    nlon; each curve runs over nsteps. Wall time is for the whole run (setup,
    time stepping, output) on a shared node, so treat it as indicative."""
    items = [(r, e) for (r, e) in errs.values() if r.nlon == nlon]
    n_untimed = sum(r.wall is None for r, _ in items)
    items = [(r, e) for (r, e) in items if r.wall is not None and r.wall > 0]
    if not items:
        print(f"[work-precision] no timed runs for nlon={nlon}; skipped")
        return
    print(f"[work-precision] {len(items)} timed runs at nlon={nlon} "
          f"({n_untimed} without a timing row skipped)")
    rks = sorted({r.rk for r, _ in items})
    gs = sorted({r.g for r, _ in items})
    ds = sorted({r.d for r, _ in items})
    gcol, dmark, dls = style_maps(gs, ds)
    ylabel = f"{NORM_LABEL[norm]} relative vorticity error at $t={t_eval:g}$"
    xlabel = "wall-clock time (s)"
    from matplotlib.lines import Line2D

    def depth_handles(ms, lw):
        return [Line2D([], [], color="k", marker=dmark[d], ls=dls[d], ms=ms, lw=lw,
                       label=f"depth {d} ({n_particles(d):,} particles)")
                for d in ds]

    def curves(sub, keyfn):
        for k, pts in sorted(group(sub, keyfn).items()):
            pts.sort(key=lambda p: p[0].nsteps)
            yield k, [p[0].wall for p in pts], [p[1] for p in pts]

    # (1) one panel per solver, color = GMLS order, marker/linestyle = depth;
    # paper style like convergence_<norm>
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, len(rks), figsize=(4.8 * len(rks), 4.8),
                                 sharex=True, sharey=True, squeeze=False)
        for ax, rk in zip(axes[0], rks):
            sub = [p for p in items if p[0].rk == rk]
            for (g, d), xs, ys in curves(sub, lambda p: (p[0].g, p[0].d)):
                ax.loglog(xs, ys, color=gcol[g], marker=dmark[d], ls=dls[d],
                          ms=7, lw=1.6)
            ax.set_title(RK_LABEL[rk])
            ax.set_xlabel(xlabel)
            ax.grid(True, which="both", alpha=0.25)
        axes[0][0].set_ylabel(ylabel)
        hs = [Line2D([], [], color=gcol[g], lw=2.7, label=f"GMLS order {g}")
              for g in gs] + depth_handles(7, 1.6)
        fig.legend(handles=hs, loc="lower center", ncol=3, frameon=False)
        fig.tight_layout(rect=(0, 0.16, 1, 1))
        save(fig, figdir / f"work_precision_{norm}", ext)

    # (2) one panel per depth, color = solver, marker = GMLS order: which RK
    # order is cheapest at a given accuracy; paper style, no title (nlon goes
    # in the caption)
    rcol = {rk: plt.get_cmap("plasma")(0.8 * i / max(len(rks) - 1, 1))
            for i, rk in enumerate(rks)}
    gmark = {g: MARKERS[i % len(MARKERS)] for i, g in enumerate(gs)}
    gls = {g: LINESTYLES[i % len(LINESTYLES)] for i, g in enumerate(gs)}
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, len(ds), figsize=(4.8 * len(ds), 4.8),
                                 sharex=True, sharey=True, squeeze=False)
        for ax, d in zip(axes[0], ds):
            sub = [p for p in items if p[0].d == d]
            for (rk, g), xs, ys in curves(sub, lambda p: (p[0].rk, p[0].g)):
                ax.loglog(xs, ys, color=rcol[rk], marker=gmark[g], ls=gls[g],
                          ms=7, lw=1.6)
            ax.set_title(f"depth {d} ({n_particles(d):,} particles)")
            ax.set_xlabel(xlabel)
            ax.grid(True, which="both", alpha=0.25)
        axes[0][0].set_ylabel(ylabel)
        hs = [Line2D([], [], color=rcol[rk], lw=2.7, label=RK_LABEL[rk]) for rk in rks]
        hs += [Line2D([], [], color="k", marker=gmark[g], ls=gls[g], ms=7, lw=1.6,
                      label=f"GMLS order {g}") for g in gs]
        fig.legend(handles=hs, loc="lower center", ncol=len(hs), frameon=False)
        fig.tight_layout(rect=(0, 0.09, 1, 1))
        save(fig, figdir / f"work_precision_{norm}_by_depth", ext)


CONSERVED = [("total_kinetic_energy", "kinetic energy"),
             ("total_enstrophy", "enstrophy")]


def relative_change(q):
    """(q(t) - q(0)) / q(0), or None if q is unusable."""
    if q is None or len(q) < 2 or not np.isfinite(q).all() or q[0] == 0:
        return None
    return (q - q[0]) / q[0]


def plot_conservation(runs, rk, g, nlon, nsteps, figdir, ext):
    """Relative change of the conserved integrals, particles vs DFS grid.

    conservation: max_t |Q(t) - Q(0)|/Q(0) vs tree depth, solid = particle
    quadrature (panel areas), dashed = DFS grid quadrature (written with -gi).
    conservation_timeseries (supplement): Q(t) vs t, color = tree depth; top
    row (Q(t) - Q(0))/Q(0) for the particles in %, bottom row |Q(t) - Q(0)|/Q(0)
    for the DFS grid on a log axis (it is too small to see on the % axis).
    """
    cands = [r for r in runs if r.rk == rk and r.nlon == nlon]
    if g is None and cands:
        g = max(r.g for r in cands)
    cands = [r for r in cands if r.g == g]
    if not cands:
        print(f"[conservation] no {rk} runs with g={g}, nlon={nlon}; skipped")
        return
    if nsteps not in {r.nsteps for r in cands}:
        fallback = min(r.nsteps for r in cands)
        print(f"[conservation] no {rk} g={g} nlon={nlon} runs at nsteps={nsteps}; "
              f"using nsteps={fallback}")
        nsteps = fallback
    sel = sorted((r for r in cands if r.nsteps == nsteps), key=lambda r: r.d)
    ds = [r.d for r in sel]
    dcol = {d: plt.get_cmap("viridis")(0.9 * i / max(len(ds) - 1, 1))
            for i, d in enumerate(ds)}
    title = (f"RH54 conservation, {RK_LABEL[rk]}, GMLS order {g}, "
             f"nlon = {nlon}, nsteps = {nsteps}")

    # series[key][src] = [(depth, t, rel)], src in ("particles", "DFS grid")
    series = {key: {"particles": [], "DFS grid": []} for key, _ in CONSERVED}
    for run in sel:
        data = read_matlab_vectors(run.path)
        t = data.get("time")
        for key, _ in CONSERVED:
            for src, suffix in (("particles", ""), ("DFS grid", "_grid")):
                rel = relative_change(data.get(key + suffix))
                if rel is None or t is None or len(rel) != len(t):
                    continue
                series[key][src].append((run.d, t, rel))
    if not any(series[key]["DFS grid"] for key, _ in CONSERVED):
        print("[conservation] no *_grid integrals in these runs (rerun with -gi); "
              "plotting particle integrals only")
    from matplotlib.lines import Line2D

    # main figure: max relative change vs depth; paper figure: no title
    # (solver, GMLS order, nlon, nsteps go in the caption), larger fonts and markers
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, len(CONSERVED), figsize=(5.4 * len(CONSERVED), 4.6),
                                 squeeze=False)
        for j, (key, label) in enumerate(CONSERVED):
            ax = axes[0][j]
            for src, ls, mk in (("particles", "-", "o"), ("DFS grid", "--", "s")):
                pts = [(d, float(np.max(np.abs(rel[1:])))) for d, _, rel in series[key][src]]
                pts = [(d, v) for d, v in pts if v > 0]
                if pts:
                    ax.semilogy(*zip(*pts), color="k", ls=ls, marker=mk, ms=8, lw=1.6)
            ax.set_title(label)
            ax.set_xlabel("tree depth")
            ax.set_xticks(ds)
            ax.set_ylabel("max relative change")
            ax.grid(True, which="both", alpha=0.25)
        hs = [Line2D([], [], color="k", ls="-", marker="o", ms=8, lw=1.6, label="particles"),
              Line2D([], [], color="k", ls="--", marker="s", ms=8, lw=1.6, label="DFS grid")]
        fig.legend(handles=hs, loc="lower center", ncol=len(hs), frameon=False)
        fig.tight_layout(rect=(0, 0.09, 1, 1))
        save(fig, figdir / "conservation", ext)

    # supplement: time series
    fig, axes = plt.subplots(2, len(CONSERVED), figsize=(4.6 * len(CONSERVED), 7.4),
                             squeeze=False)
    for j, (key, label) in enumerate(CONSERVED):
        ax_p, ax_g = axes[0][j], axes[1][j]
        for d, t, rel in series[key]["particles"]:
            ax_p.plot(t, 100 * rel, color=dcol[d], lw=1.4)
        for d, t, rel in series[key]["DFS grid"]:
            ok = np.abs(rel) > 0
            ax_g.semilogy(t[ok], np.abs(rel[ok]), color=dcol[d], lw=1.2)
        ax_p.axhline(0, color="0.6", lw=0.8, zorder=0)
        ax_p.set_title(f"{label}, particles")
        ax_p.set_ylabel(f"relative change in {label} (%)")
        ax_g.set_title(f"{label}, DFS grid")
        ax_g.set_ylabel(f"|relative change in {label}|")
        for ax in (ax_p, ax_g):
            ax.set_xlabel("time $t$")
            ax.grid(True, which="both", alpha=0.25)
    hs = [Line2D([], [], color=dcol[d], lw=2, label=f"depth {d}") for d in ds]
    fig.legend(handles=hs, loc="lower center", ncol=len(hs), frameon=False)
    fig.suptitle(title, y=1.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, figdir / "conservation_timeseries", ext)


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
                    help="nsteps of the runs shown in the conservation figures")
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"])
    ap.add_argument("--exclude", default=None, metavar="TSV",
                    help="rh54_classify_failures.py --include-ok output; runs "
                         "classed as blow-ups are skipped")
    ap.add_argument("--stale", action="append", default=[], metavar="RK@TIME",
                    help="ignore timing rows of RK written before TIME (repeatable)")
    ap.add_argument("--cons-rk", default="rk4", choices=sorted(RK_ORDER),
                    help="solver shown in the conservation figure")
    ap.add_argument("--floor-rk", default="rk4", choices=sorted(RK_ORDER),
                    help="solver used for the spatial-error figures (error at "
                         "its finest dt)")
    ap.add_argument("--cons-g", type=int, default=None,
                    help="GMLS order shown in the conservation figure "
                         "(default: largest available)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    if not outdir.is_dir():
        sys.exit(f"output directory not found: {outdir}")
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    exclude = blowup_keys(args.exclude) if args.exclude else frozenset()
    runs, n_failed, n_stale, n_dupe, n_excluded = discover_runs(
        outdir, exclude, rh54_plan.parse_stale(args.stale))
    if not runs:
        sys.exit(f"no completed runs found in {outdir}")
    nlon = pick_nlon(runs, args.nlon)
    print(f"found {len(runs)} completed runs ({n_failed} failed runs skipped, "
          f"{n_stale} stale pre-cutoff runs skipped, "
          f"{n_excluded} blow-ups excluded, "
          f"{n_dupe} duplicate runs resolved to newest); fixed nlon = {nlon}")

    for norm in args.norms:
        errs, n_bad = load_errors(runs, norm, args.t_eval)
        print(f"[{norm}] {len(errs)} usable runs, {n_bad} dropped "
              f"(did not reach t={args.t_eval:g}, or NaN/inf/zero error)")
        plot_convergence(errs, norm, nlon, args.t_eval, figdir, args.formats)
        plot_vs_gmls_order(errs, norm, nlon, args.floor_rk, args.t_eval, figdir,
                           args.formats)
        plot_vs_nlon(errs, norm, args.floor_rk, args.t_eval, figdir, args.formats)
        plot_work_precision(errs, norm, nlon, args.t_eval, figdir, args.formats)

    plot_conservation(runs, args.cons_rk, args.cons_g, nlon, args.nsteps, figdir,
                      args.formats)


if __name__ == "__main__":
    main()
