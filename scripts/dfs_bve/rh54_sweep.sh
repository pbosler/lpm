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
# ${OUTDIR}/failed_runs.${TAG}.log (TAG defaults to the short hostname).
# If machines use separate OUTDIRs, rsync them into one before plotting; the
# planner and plot_rh54.py both read every timing_summary*.csv.
#
# Environment: OMP_NUM_THREADS (required), OUTDIR, BUILD_DIR, TAG, RUN_TIMEOUT,
# EPS_MULTIPLIER, EPS_MULTIPLIER_G2, DRY_RUN.

set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BUILD_DIR="${BUILD_DIR:-${HOME}/lpm/build-dfs-bve-paper}"
OUTDIR="${OUTDIR:-/data/pabosle/dfs-bve-paper}"
TAG="${TAG:-$(hostname -s)}"
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
if [[ ! -s "${TIMING_FILE}" ]]; then
  echo "timestamp,executable,rk,gmls_order,eps_multiplier,tree_depth,nlon,tfinal,target_nsteps,dt,omp_num_threads,output_root,logfile,exit_status,elapsed_seconds" > "${TIMING_FILE}"
fi

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
      time timeout --kill-after=30s "${RUN_TIMEOUT}" "${exe}" \
        -tf "${TFINAL}" -dt "${dt}" -g "${g}" -eps "${eps}" -d "${d}" \
        -nlon "${nlon}" -of "${nsteps}" -o "${output_root}" \
        > "${logfile}" 2>&1
    } 2>&1
  )
  exit_status=$?
  set -e

  timestamp=$(date +"%Y-%m-%dT%H:%M:%S%z")
  printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
    "${timestamp}" "${exe}" "${rk}" "${g}" "${eps}" "${d}" "${nlon}" "${TFINAL}" \
    "${nsteps}" "${dt}" "${OMP_NUM_THREADS}" "${output_root}" "${logfile}" \
    "${exit_status}" "${elapsed_seconds}" >> "${TIMING_FILE}"

  if [[ "${exit_status}" -eq 0 ]]; then
    echo "  completed in ${elapsed_seconds} s"
  else
    echo "  FAILED (exit ${exit_status}) after ${elapsed_seconds} s; continuing. See ${logfile}" >&2
    {
      echo "${timestamp} ${run_id} exit_status=${exit_status} elapsed=${elapsed_seconds}s logfile=${logfile}"
      tail -n 3 "${logfile}" | cut -c1-300 | sed 's/^/    /'
    } >> "${FAILURE_FILE}"
  fi
done <<< "${plan}"

echo "Slice finished: ${n} runs. Timing: ${TIMING_FILE}"
