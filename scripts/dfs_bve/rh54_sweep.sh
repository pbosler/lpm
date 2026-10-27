#!/usr/bin/env bash
# Run one slice of the RH54 sweep (the same runs as rh54_trials.sh, split up).
#
#   OMP_NUM_THREADS=24 ./rh54_sweep.sh --shard 2/4
#   OMP_NUM_THREADS=24 ./rh54_sweep.sh --rk rk4 --g 4 6 --max-cost 3000
#   DRY_RUN=1 ./rh54_sweep.sh --shard 1/4          # print the plan only
#
# All arguments go to rh54_plan.py (see its --help): sweep-axis filters,
# --min-cost/--max-cost, --shard I/N, --shard-weights, --rerun-failed.
# Completed runs are skipped, so re-running a slice resumes it.
#
# Output layout is identical to rh54_trials.sh (run ids, ${OUTDIR}/<run_id>.m,
# logs/), so plot_rh54.py works on the result. To avoid machines writing the
# same CSV, timing goes to ${OUTDIR}/timing_summary.${TAG}.csv and failures to
# ${OUTDIR}/failed_runs.${TAG}.log (TAG defaults to <host>.s<I>of<N> with --shard).
# If machines use separate OUTDIRs, rsync them into one before plotting; the
# planner and plot_rh54.py both read every timing_summary*.csv.
#
# Environment: OMP_NUM_THREADS (required), OUTDIR, BUILD_DIR, TAG, RUN_TIMEOUT,
# EPS_MULTIPLIER, EPS_MULTIPLIER_G2, DRY_RUN.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUILD_DIR="${BUILD_DIR:-${HOME}/lpm/build-dfs-bve-paper}"
# OUTDIR="${OUTDIR:-/data/pabosle/dfs-bve-paper}"
OUTDIR="${OUTDIR:-/gpfs/pabosle/lpm-dfs-bve-paper}"

# Default TAG is <host>, or <host>.s<I>of<N> when --shard I/N is given, so
# shards sharing a node write separate timing and failure files.
SHARD_I=""
SHARD_N=""
if [[ -z "${TAG:-}" ]]; then
  TAG="$(hostname -s)"
  prev=""
  for arg in "$@"; do
    if [[ "${prev}" == "--shard" && "${arg}" =~ ^([0-9]+)/([0-9]+)$ ]]; then
      SHARD_I="${BASH_REMATCH[1]}"
      SHARD_N="${BASH_REMATCH[2]}"
      TAG="${TAG}.s${SHARD_I}of${SHARD_N}"
    fi
    prev="${arg}"
  done
fi
RUN_TIMEOUT="${RUN_TIMEOUT:-3h}"
TFINAL="2.0"
export OUTDIR RUN_TIMEOUT

plan="$(python3 "${here}/rh54_plan.py" --outdir "${OUTDIR}" --timeout "${RUN_TIMEOUT}" "$@")"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '%s\n' "${plan}"
  exit 0
fi

: "${OMP_NUM_THREADS:?ERROR: OMP_NUM_THREADS is not set. Please export OMP_NUM_THREADS before running.}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_PLACES="${OMP_PLACES:-cores}"

# cee-compute002 (16 NUMA nodes, 28 cores each; hyperthread siblings are
# cpu+448): give each shard its own NUMA node, using OMP_NUM_THREADS physical
# cores on it and node-local memory, so shards on the node do not share cores.
# Shard I (1-based) uses node CEE_NODES[I-1]; nodes 4 5 7 11 had >200 GB free on
# 2026-10-22, override with e.g. CEE_NODES="1 2 5 6". RH54_NO_PIN=1 disables.
PIN_CMD=()
if [[ "$(hostname -s)" == "cee-compute002" && "${RH54_NO_PIN:-0}" != "1" ]]; then
  if [[ -z "${SHARD_I}" ]]; then
    echo "WARNING: cee-compute002: no --shard I/N given, running unpinned." >&2
  else
    read -r -a cee_nodes <<< "${CEE_NODES:-4 5 7 11}"
    if (( SHARD_I > ${#cee_nodes[@]} )); then
      echo "ERROR: shard ${SHARD_I} has no NUMA node in CEE_NODES (${cee_nodes[*]})." >&2
      exit 1
    fi
    if (( OMP_NUM_THREADS > 28 )); then
      echo "ERROR: OMP_NUM_THREADS=${OMP_NUM_THREADS} exceeds the 28 physical cores per node." >&2
      exit 1
    fi
    node="${cee_nodes[$((SHARD_I - 1))]}"
    first=$((node * 28))
    last=$((first + OMP_NUM_THREADS - 1))
    export OMP_PROC_BIND=close OMP_PLACES=cores
    PIN_CMD=(numactl --physcpubind="${first}-${last}" --membind="${node}")
    echo "cee-compute002: shard ${SHARD_I} pinned to cpus ${first}-${last}, NUMA node ${node}"
  fi
fi

declare -A EXE=(
  [rk4]="${BUILD_DIR}/examples/dfs_bve_rh54"
  [rk2]="${BUILD_DIR}/examples/dfs_bve_rh54_rk2"
  [rk3]="${BUILD_DIR}/examples/dfs_bve_rh54_rk3"
)

LOGDIR="${OUTDIR}/logs"
TIMING_FILE="${OUTDIR}/timing_summary.${TAG}.csv"
FAILURE_FILE="${OUTDIR}/failed_runs.${TAG}.log"
mkdir -p "${OUTDIR}" "${LOGDIR}"

if [[ -z "${plan}" ]]; then
  echo "Nothing to run."
  exit 0
fi

while IFS=$'\t' read -r rk _ _ _ _ _ _; do
  [[ -x "${EXE[${rk}]}" ]] || { echo "ERROR: executable not found: ${EXE[${rk}]}" >&2; exit 1; }
done <<< "${plan}"

TIMEFORMAT="%R"
# Serialize appends when flock exists, in case two processes share a TAG.
locked() {
  if command -v flock > /dev/null; then
    ( flock 9; "${@:2}" ) 9>> "$1.lock"
  else
    "${@:2}"
  fi
}

append_line() { printf '%s\n' "$2" >> "$1"; }

write_header() {
  [[ -s "${TIMING_FILE}" ]] ||
    echo "timestamp,executable,rk,gmls_order,eps_multiplier,tree_depth,nlon,tfinal,target_nsteps,dt,omp_num_threads,output_root,logfile,exit_status,elapsed_seconds" > "${TIMING_FILE}"
}
locked "${TIMING_FILE}" write_header

n_total=$(printf '%s\n' "${plan}" | wc -l)
n=0
while IFS=$'\t' read -r rk g eps d nlon nsteps est; do
  n=$((n + 1))
  exe="${EXE[${rk}]}"
  dt=$(awk -v tf="${TFINAL}" -v n="${nsteps}" 'BEGIN {printf "%.17g", tf/n}')
  run_id="dfs_bve_rh54_${rk}_g${g}_eps${eps}_d${d}_nlon${nlon}_tf${TFINAL}_nsteps${nsteps}_nthr${OMP_NUM_THREADS}"
  output_root="${OUTDIR}/${run_id}"
  logfile="${LOGDIR}/${run_id}.log"

  echo "[${n}/${n_total}] ${run_id} (est ${est}s, timeout ${RUN_TIMEOUT})"

  set +e
  elapsed_seconds=$(
    {
      time timeout --kill-after=30s "${RUN_TIMEOUT}" ${PIN_CMD[@]+"${PIN_CMD[@]}"} "${exe}" \
        -tf "${TFINAL}" -dt "${dt}" -g "${g}" -eps "${eps}" -d "${d}" \
        -nlon "${nlon}" -of "${nsteps}" -o "${output_root}" \
        > "${logfile}" 2>&1
    } 2>&1
  )
  exit_status=$?
  set -e

  timestamp=$(date +"%Y-%m-%dT%H:%M:%S%z")
  row=$(printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" \
    "${timestamp}" "${exe}" "${rk}" "${g}" "${eps}" "${d}" "${nlon}" "${TFINAL}" \
    "${nsteps}" "${dt}" "${OMP_NUM_THREADS}" "${output_root}" "${logfile}" \
    "${exit_status}" "${elapsed_seconds}")
  locked "${TIMING_FILE}" append_line "${TIMING_FILE}" "${row}"

  if [[ "${exit_status}" -eq 0 ]]; then
    echo "  completed in ${elapsed_seconds} s"
  else
    echo "  FAILED (exit ${exit_status}) after ${elapsed_seconds} s; continuing. See ${logfile}" >&2
    entry="${timestamp} ${run_id} exit_status=${exit_status} elapsed=${elapsed_seconds}s logfile=${logfile}"$'\n'"$(tail -n 3 "${logfile}" | cut -c1-300 | sed 's/^/    /')"
    locked "${FAILURE_FILE}" append_line "${FAILURE_FILE}" "${entry}"
  fi
done <<< "${plan}"

echo "Slice finished: ${n} runs. Timing: ${TIMING_FILE}"
