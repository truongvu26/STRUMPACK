#!/bin/bash
#SBATCH -A m4872_g
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -N 4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH -t 00:30:00
#SBATCH -J hodlr150_panel_perf
#SBATCH -o profile_results/panel_perf_%j.out
set -euo pipefail
module load PrgEnv-gnu cudatoolkit/13.2 craype-accel-nvidia80
module unload cray-libsci darshan
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPICH_GPU_SUPPORT_ENABLED=0 BPACK_GPU_AWARE_MPI=0 BPACK_CHECK=''
export BPACK_GPU_HEAP_FRACTION=0.85 HODLR_PROFILE_RANKS="${HODLR_PANEL_RANKS:-32}"
grid=${HODLR_PANEL_GRID:-150}; pernode=$((HODLR_PROFILE_RANKS/SLURM_JOB_NUM_NODES))
cd "$SLURM_SUBMIT_DIR"
started=$(date +%s)
out="$PWD/profile_results/$SLURM_JOB_ID"; mkdir -p "$out"
cp "$HODLR_PROFILE_SNAPSHOT/manifest.txt" "$out/manifest.txt"
cp "${BASH_SOURCE[0]}" "$out/job.sbatch"
scontrol show job "$SLURM_JOB_ID" > "$out/slurm_job.txt"
srun -n "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 -c1 --gpu-bind=none bash -c 'hostname; nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv' > "$out/devices.txt"
read -r -a cases <<< "${HODLR_PANEL_CASES:-baseline panels}"
if [[ " ${cases[*]} " == *' traced '* ]]; then
  export HODLR_NSYS_TRACE_RANKS="${HODLR_NSYS_TRACE_RANKS:-0,$((HODLR_PROFILE_RANKS-1))}"
  printf '%s\n' "$HODLR_NSYS_TRACE_RANKS" > "$out/traced_ranks.txt"
  printf '%s\n' "${HODLR_NSYS_CPU_SAMPLING:-0}" > "$out/nsys_cpu_sampling.txt"
fi
printf '%s\n' "${cases[@]}" > "$out/cases.txt"
printf '%s\n' "$grid" "$HODLR_PROFILE_RANKS" > "$out/layout.txt"; printf '0.85\n' > "$out/heap_fraction.txt"
printf 'GPU host panels with 67108864 byte per-rank staging; original small-image cap 268435456\n' > "$out/panel_settings.txt"
for variant in "${cases[@]}"; do
  binary=baseline; extra=(); mode=1; rect=1; cap=268435456; trace=0
  case "$variant" in
    baseline|panels|diagnostic|traced) ;;
    cpu) binary=panels; mode=0; rect=0; cap=0 ;;
    *) echo "Unknown panel-profile case: $variant" >&2; exit 1 ;;
  esac
  if [[ "$variant" == panels || "$variant" == diagnostic || "$variant" == traced ]]; then
    binary=panels; extra=(--hodlr_gpu_host_panels 1 --hodlr_gpu_panel_max_bytes 67108864)
  fi
  [[ "$variant" == diagnostic || "$variant" == traced ]] && binary=diagnostic
  [[ "$variant" == traced ]] && trace=1
  args=("$HODLR_PROFILE_SNAPSHOT/$binary" "$grid" 3 --sp_compression HODLR
    --sp_compression_min_sep_size 1000 --sp_compression_min_front_size 1000
    --hodlr_leaf_size 128 --hodlr_rel_tol 1e-5 --hodlr_butterfly_levels 0
    --sp_Krylov_solver refinement --sp_rel_tol 1e-10 --hodlr_use_gpu "$mode" --hodlr_gpu_rectangular "$rect"
    --hodlr_gpu_front_max_bytes "$cap" "${extra[@]}")
  [[ -z ${HODLR_PANEL_MAXIT:-} ]] || args+=(--sp_maxit "$HODLR_PANEL_MAXIT")
  printf '%q ' "${args[@]}" > "$out/$variant.command"; printf '\n' >> "$out/$variant.command"
  remaining=$((${HODLR_PANEL_ALLOCATION_SECONDS:-1800}-$(date +%s)+started)); limit=$((remaining-90))
  (( limit>0 )) || { echo 'No time remains'; exit 1; }; (( limit<1650 )) || limit=1650
  status=0
  /usr/bin/time -v -o "$out/$variant.time" timeout --kill-after=30s "${limit}s" \
    srun -n "$HODLR_PROFILE_RANKS" --ntasks-per-node="$pernode" -c8 --gpu-bind=none --cpu-bind=cores --kill-on-bad-exit=1 --wait=0 \
    --output="$out/$variant.rank%t.log" --error="$out/$variant.rank%t.log" \
    bash test/run_hodlr_stage_rank.sh "$out" "$variant" "$trace" "${args[@]}" || status=$?
  printf '%s\n' "$status" > "$out/$variant.exit_code"
  if [[ "$trace" == 1 && "$status" == 0 ]]; then
    report_status=0
    bash test/process_hodlr_stage_traces.sh "$out" "$HODLR_PROFILE_RANKS" || report_status=$?
    printf '%s\n' "$report_status" > "$out/$variant.report_exit_code"
  fi
  python3 test/analyze_hodlr_stages.py "$out"
  (( status==0 )) || exit "$status"
done
python3 test/analyze_hodlr_stages.py "$out" --require-complete
sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,MaxRSS,ExitCode -P > "$out/accounting.txt"
