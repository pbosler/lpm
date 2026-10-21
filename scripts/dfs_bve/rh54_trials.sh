#!/usr/bin/env bash

set -euo pipefail

# ---------------------------------------------------------------------------
# Build directory and executable paths
# ---------------------------------------------------------------------------

buildDir="${HOME}/lpm/build-dfs-bve-paper"

EXE_RK4="${buildDir}/examples/dfs_bve_rh54"
EXE_RK2="${buildDir}/examples/dfs_bve_rh54_rk2"
EXE_RK3="${buildDir}/examples/dfs_bve_rh54_rk3"

# ---------------------------------------------------------------------------
# Threading
# ---------------------------------------------------------------------------

: "${OMP_NUM_THREADS:?ERROR: OMP_NUM_THREADS is not set. Please export OMP_NUM_THREADS before running.}"

# Optional OpenMP settings
export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
export OMP_PROC_BIND="${OMP_PROC_BIND:-spread}"
export OMP_PLACES="${OMP_PLACES:-cores}"

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------

OUTDIR="/data/pabosle/dfs-bve-paper"
LOGDIR="${OUTDIR}/logs"

mkdir -p "${OUTDIR}"
mkdir -p "${LOGDIR}"

TIMING_FILE="${OUTDIR}/timing_summary.csv"

# ---------------------------------------------------------------------------
# Executables and labels
# ---------------------------------------------------------------------------

EXECUTABLES=(
  "${EXE_RK4}:rk4"
  "${EXE_RK2}:rk2"
  "${EXE_RK3}:rk3"
)

# ---------------------------------------------------------------------------
# Parameter sweeps
# ---------------------------------------------------------------------------

GMLS_ORDERS=(2 4 6 8)
TREE_DEPTHS=(4 5 6)
NLONS=(20 40 60 80 100 120)
NSTEPS_LIST=(40 80 125 250 500 1000 2000 4000)

TFINAL="2.0"

# Bash time format:
#   %R = elapsed wall-clock time in seconds
TIMEFORMAT="%R"

# ---------------------------------------------------------------------------
# Check executables before starting
# ---------------------------------------------------------------------------

for exe_label in "${EXECUTABLES[@]}"; do
  exe="${exe_label%%:*}"

  if [[ ! -x "${exe}" ]]; then
    echo "ERROR: executable not found or not executable: ${exe}" >&2
    exit 1
  fi
done

# ---------------------------------------------------------------------------
# Write timing CSV header
# ---------------------------------------------------------------------------

cat > "${TIMING_FILE}" << EOF
timestamp,executable,rk,gmls_order,tree_depth,nlon,tfinal,target_nsteps,dt,omp_num_threads,output_root,logfile,exit_status,elapsed_seconds
EOF

# ---------------------------------------------------------------------------
# Run sweep
# ---------------------------------------------------------------------------

total_runs=0
successful_runs=0
failed_runs=0

for exe_label in "${EXECUTABLES[@]}"; do
  exe="${exe_label%%:*}"
  rk_label="${exe_label##*:}"

  for g in "${GMLS_ORDERS[@]}"; do
    for d in "${TREE_DEPTHS[@]}"; do
      for nlon in "${NLONS[@]}"; do
        for nsteps in "${NSTEPS_LIST[@]}"; do

          # Use the default use_dt=true path and pass dt directly.
          # This avoids relying on the executable's bool parser for -udt/-un.
          dt=$(awk -v tf="${TFINAL}" -v n="${nsteps}" 'BEGIN {printf "%.17g", tf/n}')

          run_id="dfs_bve_rh54_${rk_label}_g${g}_d${d}_nlon${nlon}_tf${TFINAL}_nsteps${nsteps}_nthr${OMP_NUM_THREADS}"

          output_root="${OUTDIR}/${run_id}"
          logfile="${LOGDIR}/${run_id}.log"

          echo "================================================================"
          echo "Running ${run_id}"
          echo "Executable:          ${exe}"
          echo "OMP_NUM_THREADS:     ${OMP_NUM_THREADS}"
          echo "tfinal:              ${TFINAL}"
          echo "target nsteps:       ${nsteps}"
          echo "dt:                  ${dt}"
          echo "GMLS order:          ${g}"
          echo "tree depth:          ${d}"
          echo "nlon:                ${nlon}"
          echo "Output root:         ${output_root}"
          echo "Log file:            ${logfile}"
          echo "================================================================"

          set +e

          elapsed_seconds=$(
            {
              time "${exe}" \
                -tf "${TFINAL}" \
                -dt "${dt}" \
                -g "${g}" \
                -d "${d}" \
                -nlon "${nlon}" \
                -of "${nsteps}" \
                -o "${output_root}" \
                > "${logfile}" 2>&1
            } 2>&1
          )

          exit_status=$?

          set -e

          timestamp=$(date +"%Y-%m-%dT%H:%M:%S%z")

          printf "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
            "${timestamp}" \
            "${exe}" \
            "${rk_label}" \
            "${g}" \
            "${d}" \
            "${nlon}" \
            "${TFINAL}" \
            "${nsteps}" \
            "${dt}" \
            "${OMP_NUM_THREADS}" \
            "${output_root}" \
            "${logfile}" \
            "${exit_status}" \
            "${elapsed_seconds}" \
            >> "${TIMING_FILE}"

          total_runs=$((total_runs + 1))

          if [[ "${exit_status}" -eq 0 ]]; then
            successful_runs=$((successful_runs + 1))
            echo "Completed ${run_id} in ${elapsed_seconds} seconds."
          else
            failed_runs=$((failed_runs + 1))
            echo "ERROR: ${run_id} failed with exit status ${exit_status} after ${elapsed_seconds} seconds." >&2
            echo "See log file: ${logfile}" >&2
            echo "Timing record written to: ${TIMING_FILE}" >&2
            exit "${exit_status}"
          fi

        done
      done
    done
  done
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo "================================================================"
echo "All runs completed."
echo "Total runs:       ${total_runs}"
echo "Successful runs:  ${successful_runs}"
echo "Failed runs:      ${failed_runs}"
echo "OMP_NUM_THREADS:  ${OMP_NUM_THREADS}"
echo "Output directory: ${OUTDIR}"
echo "Log directory:    ${LOGDIR}"
echo "Timing file:      ${TIMING_FILE}"
echo "================================================================"