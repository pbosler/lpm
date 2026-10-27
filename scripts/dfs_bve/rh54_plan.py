#!/usr/bin/env python3
"""Plan (and shard) the RH54 sweep that rh54_trials.sh runs serially.

Prints one tab-separated line per run to stdout:

    rk  g  eps  d  nlon  nsteps  est_seconds

and a summary to stderr. rh54_sweep.sh consumes the lines. The run set is the
same cross product as rh54_trials.sh; the run ids it produces are identical, so
plot_rh54.py reads the output unchanged.

Selecting work (all filters are ANDed):
  --rk/--g/--depth/--nlon/--nsteps     restrict a sweep axis
  --min-cost/--max-cost SECONDS        keep runs whose estimated cost is in range
                                       (cheap runs on workstations, the rest on HPC)
  --shard I/N                          take shard I (1-based) of N. Runs are dealt
                                       out largest-first to the least-loaded shard
                                       (LPT), so shards have similar total cost.
  --shard-weights W1,W2,..             relative machine speeds, one per shard

The shard assignment is computed over the full filtered set *before* completed
runs are removed, so it does not shift as runs finish: every machine can re-run
this at any time and gets a stable, disjoint share.

Completed runs (exit_status 0 in any <outdir>/timing_summary*.csv, thread count
ignored) are skipped. Runs that failed or timed out are skipped too unless
--rerun-failed is given, since a timeout would just repeat.

The cost model, STARTUP + nsteps * STEP_COST[nlon] * DEPTH_FACTOR[d] *
G_FACTOR[g] * RK_FACTOR[rk], was fit against the 678 successful HPC rows in
timing_summary*.csv as of 2026-10-27 (Huber loss in log-space). It is a crude
estimate: thread contention when the four shards share a node inflates elapsed
time by factors of 2-3x, so per-group RMS log-residual is ~0.8 (model vs
measured within 2x for the middle 50% of runs). nlon=60 and nlon=100 are the
most lightly sampled (6 and 3 (rk,g,d) groups with >=3 nsteps points) and the
non-monotonic shape there could still shift as the sweep fills in. Use
--calibrate to compare the model with recorded elapsed_seconds.
"""
import argparse
import csv
import glob
import os
import re
import sys

RKS = ["rk4", "rk2", "rk3"]
GS = [2, 4, 6, 8]
DEPTHS = [4, 5, 6]
NLONS = [20, 40, 60, 80, 100, 120]
NSTEPS = [40, 80, 125, 250, 500, 1000, 2000, 4000]
TFINAL = "2.0"

# seconds per RK4 step at g=2, depth 4, fit from HPC timing_summary*.csv (2026-10-27)
STEP_COST = {20: 4.0, 40: 8.0, 60: 1.5, 80: 12.0, 100: 0.5, 120: 15.5}
DEPTH_FACTOR = {4: 1.0, 5: 1.0, 6: 1.1}
G_FACTOR = {2: 1.0, 4: 0.75, 6: 1.25, 8: 2.0}
RK_FACTOR = {"rk4": 1.0, "rk3": 0.80, "rk2": 0.65}
STARTUP = 500.0


def eps_for(g):
    e2 = os.environ.get("EPS_MULTIPLIER_G2", "3.0")
    e = os.environ.get("EPS_MULTIPLIER", "2.0")
    return e2 if g == 2 else e


def parse_timeout(s):
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd]?)", s)
    if not m:
        sys.exit("bad --timeout %r" % s)
    return float(m.group(1)) * {"": 1, "s": 1, "m": 60, "h": 3600, "d": 86400}[m.group(2)]


def est(rk, g, d, nlon, nsteps, cap):
    t = nsteps * STEP_COST[nlon] * DEPTH_FACTOR[d] * G_FACTOR[g] * RK_FACTOR[rk]
    return min(t + STARTUP, cap)


RUN_RE = re.compile(
    r"dfs_bve_rh54_(rk\d)_g(\d+)_eps[^_]+_d(\d+)_nlon(\d+)_tf[^_]+_nsteps(\d+)_nthr\d+")


def recorded(outdir):
    """(done, failed) sets of (rk,g,d,nlon,nsteps) from every timing CSV."""
    done, failed = set(), set()
    for path in glob.glob(os.path.join(outdir, "timing_summary*.csv")):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                m = RUN_RE.search(row.get("output_root", ""))
                if not m:
                    continue
                key = (m[1], int(m[2]), int(m[3]), int(m[4]), int(m[5]))
                try:
                    ok = int(row["exit_status"]) == 0
                except (KeyError, ValueError):
                    continue
                (done if ok else failed).add(key)
    return done, failed - done


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=os.environ.get("OUTDIR", "/data/pabosle/dfs-bve-paper"))
    ap.add_argument("--rk", nargs="+", choices=RKS, default=RKS)
    ap.add_argument("--g", nargs="+", type=int, choices=GS, default=GS)
    ap.add_argument("--depth", nargs="+", type=int, choices=DEPTHS, default=DEPTHS)
    ap.add_argument("--nlon", nargs="+", type=int, choices=NLONS, default=NLONS)
    ap.add_argument("--nsteps", nargs="+", type=int, choices=NSTEPS, default=NSTEPS)
    ap.add_argument("--min-cost", type=float, default=0.0)
    ap.add_argument("--max-cost", type=float, default=float("inf"))
    ap.add_argument("--shard", default="1/1", help="I/N (1-based)")
    ap.add_argument("--shard-weights", default=None)
    ap.add_argument("--timeout", default=os.environ.get("RUN_TIMEOUT", "3h"))
    ap.add_argument("--rerun-failed", action="store_true")
    ap.add_argument("--calibrate", action="store_true",
                    help="compare modelled and recorded elapsed_seconds, then exit")
    a = ap.parse_args()
    cap = parse_timeout(a.timeout)

    if a.calibrate:
        for path in sorted(glob.glob(os.path.join(a.outdir, "timing_summary*.csv"))):
            with open(path, newline="") as f:
                for r in csv.DictReader(f):
                    m = RUN_RE.search(r["output_root"])
                    if m and r["exit_status"] == "0":
                        rk, g, d, nl, ns = m[1], int(m[2]), int(m[3]), int(m[4]), int(m[5])
                        e = est(rk, g, d, nl, ns, 1e12)
                        print("%s g%d d%d nlon%d n%d model %.0f actual %.0f ratio %.2f" % (
                            rk, g, d, nl, ns, e, float(r["elapsed_seconds"]),
                            float(r["elapsed_seconds"]) / e))
        return

    n_shards = int(a.shard.split("/")[1])
    shard = int(a.shard.split("/")[0])
    if not 1 <= shard <= n_shards:
        sys.exit("--shard must be I/N with 1 <= I <= N")
    w = ([float(x) for x in a.shard_weights.split(",")] if a.shard_weights
         else [1.0] * n_shards)
    if len(w) != n_shards:
        sys.exit("--shard-weights needs one value per shard")

    runs = []
    for rk in a.rk:
        for g in a.g:
            for d in a.depth:
                for nlon in a.nlon:
                    for ns in a.nsteps:
                        c = est(rk, g, d, nlon, ns, cap)
                        if a.min_cost <= c <= a.max_cost:
                            runs.append((c, rk, g, d, nlon, ns))

    # LPT: biggest first onto the shard with the smallest load/weight.
    runs.sort(key=lambda r: (-r[0], r[1:]))
    load = [0.0] * n_shards
    mine = []
    for r in runs:
        i = min(range(n_shards), key=lambda k: (load[k] / w[k], k))
        load[i] += r[0]
        if i == shard - 1:
            mine.append(r)

    done, failed = recorded(a.outdir)
    todo, n_done, n_failed = [], 0, 0
    for r in mine:
        key = r[1:]
        if key in done:
            n_done += 1
        elif key in failed and not a.rerun_failed:
            n_failed += 1
        else:
            todo.append(r)
    todo.sort(key=lambda r: (r[0], r[1:]))   # cheap first: results arrive early

    for c, rk, g, d, nlon, ns in todo:
        print("\t".join(map(str, (rk, g, eps_for(g), d, nlon, ns, "%.0f" % c))))

    tot = sum(r[0] for r in todo)
    print("shard %d/%d: %d runs in shard, %d done, %d skipped (failed before; "
          "--rerun-failed to retry), %d to run, est %.1f days (cap %ds/run)" % (
              shard, n_shards, len(mine), n_done, n_failed, len(todo), tot / 86400, cap),
          file=sys.stderr)
    print("all shards, est days: " + " ".join("%.1f" % (x / 86400) for x in load),
          file=sys.stderr)


if __name__ == "__main__":
    main()
