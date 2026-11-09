#!/usr/bin/env python3
"""Classify RH54 sweep runs that never succeeded as blow-ups, timeouts or crashes.

Reads every <outdir>/timing_summary*.csv (as rh54_plan.py does), keeping only
runs in the current rh54_plan.py cross product (--all-history keeps legacy
axes such as g=2). A run that has exit_status 0 in any row is done and ignored
unless --include-ok is given; for every other run the most recent failed row is
classified from its log:

    blowup   max |rel. vort passive| grew by more than --factor over its
             initial value, or went non-finite. The RH54 vorticity is
             materially conserved, so its range should not grow at all.
    timeout  exit 124 (timeout) or 137 (killed after --kill-after), no blow-up
    crash    any other nonzero exit, no blow-up
    no_log   log file not found

With --include-ok, completed runs are scanned too and classed ok, or
ok_blowup when the same growth test fires (the run finished, but its errors
are not meaningful and it should be dropped from convergence plots).

Prints one tab-separated line per failed run to stdout:

    class  rk  g  d  nlon  nsteps  exit  elapsed_s  cr  t_last  t_onset  vort_growth

(t_onset is the first logged time at which the growth exceeded --factor; "-"
when not applicable.) Counts by rk and class go to stderr.

    python3 rh54_classify_failures.py --outdir /gpfs/pabosle/lpm-dfs-bve-paper > rh54_failures.tsv
"""
import argparse
import collections
import csv
import glob
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rh54_plan  # noqa: E402  (sweep axes)

RUN_RE = re.compile(
    r"dfs_bve_rh54_(rk\d)_g(\d+)_eps[^_]+_d(\d+)_nlon(\d+)_tf[^_]+_nsteps(\d+)_nthr\d+")
RANGE_RE = re.compile(
    r"t = (\S+), rel\. vort passive range : \((\S+), (\S+)\)")
CR_RE = re.compile(r"dt = \S+, cr = (\S+)")
TIMEOUT_EXITS = {124, 137}


def in_plan(key):
    rk, g, d, nlon, nsteps = key
    return (rk in rh54_plan.RKS and g in rh54_plan.GS and d in rh54_plan.DEPTHS and
            nlon in rh54_plan.NLONS and nsteps in rh54_plan.NSTEPS)


def latest_rows(outdir):
    """({key: row} failed, {key: row} done), keeping the latest row of each."""
    done, failed = {}, {}
    for path in glob.glob(os.path.join(outdir, "timing_summary*.csv")):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                m = RUN_RE.search(row.get("output_root", ""))
                if not m:
                    continue
                key = (m.group(1), int(m.group(2)), int(m.group(3)),
                       int(m.group(4)), int(m.group(5)))
                try:
                    status = int(row["exit_status"])
                except (KeyError, ValueError):
                    continue
                rows = done if status == 0 else failed
                if key not in rows or row["timestamp"] > rows[key]["timestamp"]:
                    rows[key] = row
    return {k: r for k, r in failed.items() if k not in done}, done


def find_log(row, outdir):
    path = row.get("logfile", "")
    if path and os.path.isfile(path):
        return path
    # logs moved with the outdir (e.g. rsync from another machine)
    alt = os.path.join(outdir, "logs", os.path.basename(path))
    return alt if path and os.path.isfile(alt) else None


def scan_log(path, factor):
    """(cr, t_last, t_onset, growth) from one run log; None where not found."""
    cr = t_last = t_onset = None
    zeta0 = None
    zmax = 0.0
    with open(path, errors="replace") as f:
        for line in f:
            if cr is None:
                m = CR_RE.search(line)
                if m:
                    cr = m.group(1)
                    continue
            m = RANGE_RE.search(line)
            if not m:
                continue
            try:
                t, lo, hi = float(m.group(1)), float(m.group(2)), float(m.group(3))
            except ValueError:
                continue
            t_last = t
            z = max(abs(lo), abs(hi))
            if not math.isfinite(z):
                z = math.inf
            if zeta0 is None:
                zeta0 = z
            zmax = max(zmax, z)
            if t_onset is None and zeta0 > 0 and z > factor * zeta0:
                t_onset = t
    growth = zmax / zeta0 if zeta0 else None
    return cr, t_last, t_onset, growth


def fmt(x, spec="%g"):
    return "-" if x is None else (x if isinstance(x, str) else spec % x)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=os.environ.get("OUTDIR", "/gpfs/pabosle/lpm-dfs-bve-paper"))
    ap.add_argument("--factor", type=float, default=2.0,
                    help="vorticity growth over its initial max that counts as a blow-up")
    ap.add_argument("--include-ok", action="store_true",
                    help="also scan completed runs for blow-ups")
    ap.add_argument("--all-history", action="store_true",
                    help="keep runs outside the current rh54_plan.py axes")
    a = ap.parse_args()

    failed, done = latest_rows(a.outdir)
    runs = dict(failed)
    if a.include_ok:
        runs.update(done)
    if not a.all_history:
        runs = {k: r for k, r in runs.items() if in_plan(k)}
    counts = collections.Counter()
    print("\t".join(("class", "rk", "g", "d", "nlon", "nsteps", "exit", "elapsed_s",
                     "cr", "t_last", "t_onset", "vort_growth")))
    for key in sorted(runs):
        row = runs[key]
        status = int(row["exit_status"])
        log = find_log(row, a.outdir)
        cr = t_last = t_onset = growth = None
        if log is None:
            cls = "no_log"
        else:
            cr, t_last, t_onset, growth = scan_log(log, a.factor)
            if status == 0:
                cls = "ok" if t_onset is None else "ok_blowup"
            elif t_onset is not None:
                cls = "blowup"
            elif status in TIMEOUT_EXITS:
                cls = "timeout"
            else:
                cls = "crash"
        counts[(key[0], cls)] += 1
        print("\t".join((cls,) + tuple(map(str, key)) + (
            str(status), fmt(row.get("elapsed_seconds") or None), fmt(cr),
            fmt(t_last), fmt(t_onset), fmt(growth, "%.3g"))))

    classes = ["blowup", "timeout", "crash", "no_log"]
    if a.include_ok:
        classes = ["ok", "ok_blowup"] + classes
    rks = sorted({k[0] for k in counts})
    print("%d runs in %s" % (len(runs), a.outdir), file=sys.stderr)
    print("rk\t" + "\t".join(classes), file=sys.stderr)
    for rk in rks:
        print(rk + "\t" + "\t".join(str(counts[(rk, c)]) for c in classes), file=sys.stderr)


if __name__ == "__main__":
    main()
