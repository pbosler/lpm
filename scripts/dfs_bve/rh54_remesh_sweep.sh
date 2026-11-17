#!/usr/bin/env bash
# RH54 remeshing sweep: no remeshing vs. remeshing every N steps vs.
# FTLE-triggered remeshing, at fixed RK4 / GMLS order / nlon / dt.
#
#   OMP_NUM_THREADS=28 ./rh54_remesh_sweep.sh                    # everything
#   OMP_NUM_THREADS=28 ./rh54_remesh_sweep.sh --shard 2/5
#   OMP_NUM_THREADS=28 ./rh54_remesh_sweep.sh --depth 5 --modes ftle --ftle 1 3
#   DRY_RUN=1 ./rh54_remesh_sweep.sh --shard 1/5                 # print the plan only
#
# Options (each takes one or more values; defaults in brackets):
#   --depth D..        mesh depths                       [5 6]
#   --nsteps N..       time steps to tfinal = 2          [1000]
#   --modes M..        none | interval | ftle            [none interval ftle]
#   --rm N..           remesh intervals, in time steps   [20]
#   --ftle T..         FTLE remesh tolerances            [1 2 4]
#   --g G              GMLS order                        [6]
#   --rmeps E          GMLS radius multiplier for the    [EPS_MULTIPLIER]
#                      remesh only (-rmeps); RK stages
#                      keep EPS_MULTIPLIER
#   --nlon N           DFS grid longitudes               [80]
#   --shard I/N        take every N-th run starting at I (1-based)
#
# A run is done when its .m file exists (the driver writes it last), so
# re-running a slice resumes it; delete a .m file to redo that run.
#
# Choosing the FTLE tolerances: max FTLE is log of the largest stretch since
# the last remesh. In the no-remesh RH54 runs (rk4 g6 nlon80, d5 and d6) it
# first exceeds 1, 2, 4, 6 at t ~ 0.07, 0.21, 0.48, 0.75-1.0, so tolerances
# 1, 2, 4 remesh very roughly every 0.07, 0.2, 0.5 time units. For comparison,
# --rm 20 at nsteps 1000 remeshes every 0.04 time units (49 remeshes).
#
# Remesh neighborhoods: at eps 2.0, d6 g6 rm20 amplified rough error ~30x per
# remesh near the equatorial saddles (g4 was worse); eps 3.0 for the remesh
# removed it (rh54_remesh_diag, 2026-11-15), so pass --rmeps 3.0 for the
# study. Remeshed run ids end in _rmeps<E> when it differs from EPS_MULTIPLIER.
#
# Output: ${OUTDIR}/<run_id>_<driver suffix>.m (with remesh_times), logs in
# ${OUTDIR}/logs, timing in ${OUTDIR}/timing_summary.${TAG}.csv, failures in
# ${OUTDIR}/failed_runs.${TAG}.log. OUTDIR is separate from the no-remesh
# sweep's so plot_rh54.py does not pick these runs up.
#
# Environment: OMP_NUM_THREADS (required), OUTDIR, BUILD_DIR, TAG, RUN_TIMEOUT,
# EPS_MULTIPLIER, OUTPUT_FREQUENCY (VTK output every this many steps; default
# nsteps, i.e. first and last step only), CEE_NODES, RH54_NO_PIN, DRY_RUN.

set -euo pipefail

BUILD_DIR="${BUILD_DIR:-${HOME}/lpm/build-dfs-bve-paper}"
OUTDIR="${OUTDIR:-/gpfs/pabosle/lpm-dfs-bve-paper/rh54_remesh}"
RUN_TIMEOUT="${RUN_TIMEOUT:-8h}"
EPS="${EPS_MULTIPLIER:-2.0}"
TFINAL="2.0"
EXE="${BUILD_DIR}/examples/dfs_bve_rh54"

DEPTHS=(5 6)
NSTEPS_LIST=(1000)
MODES=(none interval ftle)
RM_LIST=(20)
FTLE_LIST=(1 2 4)
G=6
RMEPS=""
NLON=80
SHARD_I=""
SHARD_N=""

# collect the values that follow a flag, up to the next --flag
take_values() {
  local -n out=$1
  shift
  out=()
  while [[ $# -gt 0 && "$1" != --* ]]; do
    out+=("$1")
    shift
  done
  [[ ${#out[@]} -gt 0 ]] || { echo "ERROR: missing value(s)" >&2; exit 1; }
}

while [[ $# -gt 0 ]]; do
  flag="$1"
  shift
  vals=()
  case "${flag}" in
    --depth)  take_values DEPTHS "$@";      shift ${#DEPTHS[@]} ;;
    --nsteps) take_values NSTEPS_LIST "$@"; shift ${#NSTEPS_LIST[@]} ;;
    --modes)  take_values MODES "$@";       shift ${#MODES[@]} ;;
    --rm)     take_values RM_LIST "$@";     shift ${#RM_LIST[@]} ;;
    --ftle)   take_values FTLE_LIST "$@";   shift ${#FTLE_LIST[@]} ;;
    --g)      take_values vals "$@"; G="${vals[0]}";    shift ${#vals[@]} ;;
    --nlon)   take_values vals "$@"; NLON="${vals[0]}"; shift ${#vals[@]} ;;
    --rmeps)  take_values vals "$@"; RMEPS="${vals[0]}"; shift ${#vals[@]} ;;
    --shard)
      take_values vals "$@"; shift ${#vals[@]}
      [[ "${vals[0]}" =~ ^([0-9]+)/([0-9]+)$ ]] || { echo "ERROR: --shard I/N" >&2; exit 1; }
      SHARD_I="${BASH_REMATCH[1]}"
      SHARD_N="${BASH_REMATCH[2]}"
      (( SHARD_I >= 1 && SHARD_I <= SHARD_N )) || { echo "ERROR: --shard I/N needs 1 <= I <= N" >&2; exit 1; }
      ;;
    -h|--help) sed -n '2,/^$/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "ERROR: unknown option ${flag}" >&2; exit 1 ;;
  esac
done

for m in "${MODES[@]}"; do
  [[ "${m}" == none || "${m}" == interval || "${m}" == ftle ]] ||
    { echo "ERROR: unknown mode ${m} (none | interval | ftle)" >&2; exit 1; }
done

# the remesh uses the per-step neighborhoods unless --rmeps is given
RMEPS="${RMEPS:-${EPS}}"
RMEPS_TAG=""
if awk -v a="${RMEPS}" -v b="${EPS}" 'BEGIN {exit !(a + 0 != b + 0)}'; then
  RMEPS_TAG="_rmeps${RMEPS}"
fi

if [[ -z "${TAG:-}" ]]; then
  TAG="$(hostname -s)"
  [[ -n "${SHARD_I}" ]] && TAG="${TAG}.s${SHARD_I}of${SHARD_N}"
fi

# Plan: one line per run, "d nsteps mode param", where param is the remesh
# interval (interval), the FTLE tolerance (ftle), or - (none). Most expensive
# (deepest, most steps) first, so --shard deals heavy runs out evenly.
plan=()
for d in $(printf '%s\n' "${DEPTHS[@]}" | sort -rn); do
  for nsteps in $(printf '%s\n' "${NSTEPS_LIST[@]}" | sort -rn); do
    for mode in "${MODES[@]}"; do
      case "${mode}" in
        none)     plan+=("${d} ${nsteps} none -") ;;
        interval) for rm in "${RM_LIST[@]}"; do plan+=("${d} ${nsteps} interval ${rm}"); done ;;
        ftle)     for tol in "${FTLE_LIST[@]}"; do plan+=("${d} ${nsteps} ftle ${tol}"); done ;;
      esac
    done
  done
done

run_id_of() {  # d nsteps mode param
  local tag
  case "$3" in
    none)     tag="no_rm" ;;
    interval) tag="rm${4}${RMEPS_TAG}" ;;
    ftle)     tag="ftle${4}${RMEPS_TAG}" ;;
  esac
  echo "dfs_bve_rh54_rk4_g${G}_eps${EPS}_d$1_nlon${NLON}_tf${TFINAL}_nsteps$2_${tag}"
}

# match up to the driver suffix, so ..._rm20 does not match ..._rm20_rmeps3.0
is_done() { compgen -G "${OUTDIR}/$1_cubed_sphere*.m" > /dev/null; }

todo=()
idx=0
for entry in "${plan[@]}"; do
  idx=$((idx + 1))
  if [[ -n "${SHARD_I}" ]] && (( (idx - 1) % SHARD_N != SHARD_I - 1 )); then
    continue
  fi
  read -r d nsteps mode param <<< "${entry}"
  run_id="$(run_id_of "${d}" "${nsteps}" "${mode}" "${param}")"
  if is_done "${run_id}"; then
    echo "done:  ${run_id}" >&2
  else
    todo+=("${entry}")
  fi
done

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  for entry in "${todo[@]+"${todo[@]}"}"; do
    read -r d nsteps mode param <<< "${entry}"
    echo "run:   $(run_id_of "${d}" "${nsteps}" "${mode}" "${param}")"
  done
  echo "${#todo[@]} of ${#plan[@]} runs to do in this slice." >&2
  exit 0
fi

if [[ ${#todo[@]} -eq 0 ]]; then
  echo "Nothing to run."
  exit 0
fi

: "${OMP_NUM_THREADS:?ERROR: OMP_NUM_THREADS is not set. Please export OMP_NUM_THREADS before running.}"
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_PLACES="${OMP_PLACES:-cores}"
[[ -x "${EXE}" ]] || { echo "ERROR: executable not found: ${EXE}" >&2; exit 1; }

# cee-compute002 (16 NUMA nodes, 28 cores each): pin each shard to its own
# NUMA node, as rh54_sweep.sh does. Shard I uses node CEE_NODES[I-1].
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

LOGDIR="${OUTDIR}/logs"
TIMING_FILE="${OUTDIR}/timing_summary.${TAG}.csv"
FAILURE_FILE="${OUTDIR}/failed_runs.${TAG}.log"
mkdir -p "${OUTDIR}" "${LOGDIR}"

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
    echo "timestamp,executable,gmls_order,eps_multiplier,remesh_eps,tree_depth,nlon,tfinal,target_nsteps,dt,remesh_mode,remesh_param,omp_num_threads,output_root,logfile,exit_status,elapsed_seconds" > "${TIMING_FILE}"
}
locked "${TIMING_FILE}" write_header

n_total=${#todo[@]}
n=0
for entry in "${todo[@]}"; do
  n=$((n + 1))
  read -r d nsteps mode param <<< "${entry}"
  dt=$(awk -v tf="${TFINAL}" -v n="${nsteps}" 'BEGIN {printf "%.17g", tf/n}')
  run_id="$(run_id_of "${d}" "${nsteps}" "${mode}" "${param}")"
  output_root="${OUTDIR}/${run_id}"
  logfile="${LOGDIR}/${run_id}.log"
  case "${mode}" in
    none)     remesh_args=() ;;
    interval) remesh_args=(-rm "${param}" -rmeps "${RMEPS}") ;;
    ftle)     remesh_args=(-rt ftle -ftle "${param}" -rmeps "${RMEPS}") ;;
  esac

  echo "[${n}/${n_total}] ${run_id} (timeout ${RUN_TIMEOUT})"

  set +e
  elapsed_seconds=$(
    {
      time timeout --kill-after=30s "${RUN_TIMEOUT}" ${PIN_CMD[@]+"${PIN_CMD[@]}"} "${EXE}" \
        -tf "${TFINAL}" -dt "${dt}" -g "${G}" -eps "${EPS}" -d "${d}" \
        -nlon "${NLON}" -of "${OUTPUT_FREQUENCY:-${nsteps}}" -o "${output_root}" -gi \
        ${remesh_args[@]+"${remesh_args[@]}"} \
        > "${logfile}" 2>&1
    } 2>&1
  )
  exit_status=$?
  set -e

  timestamp=$(date +"%Y-%m-%dT%H:%M:%S%z")
  [[ "${mode}" == none ]] && rmeps_col="-" || rmeps_col="${RMEPS}"
  row=$(printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s" \
    "${timestamp}" "${EXE}" "${G}" "${EPS}" "${rmeps_col}" "${d}" "${NLON}" "${TFINAL}" \
    "${nsteps}" "${dt}" "${mode}" "${param}" "${OMP_NUM_THREADS}" \
    "${output_root}" "${logfile}" "${exit_status}" "${elapsed_seconds}")
  locked "${TIMING_FILE}" append_line "${TIMING_FILE}" "${row}"

  if [[ "${exit_status}" -eq 0 ]]; then
    echo "  completed in ${elapsed_seconds} s"
  else
    echo "  FAILED (exit ${exit_status}) after ${elapsed_seconds} s; continuing. See ${logfile}" >&2
    entry_txt="${timestamp} ${run_id} exit_status=${exit_status} elapsed=${elapsed_seconds}s logfile=${logfile}"$'\n'"$(tail -n 3 "${logfile}" | cut -c1-300 | sed 's/^/    /')"
    locked "${FAILURE_FILE}" append_line "${FAILURE_FILE}" "${entry_txt}"
  fi
done

echo "Slice finished: ${n} runs. Timing: ${TIMING_FILE}"
