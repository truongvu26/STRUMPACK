#!/bin/bash
#SBATCH -A m4872_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -N 4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH -t 02:00:00
#SBATCH -J hodlr_150_profile
#SBATCH -o profile_results/stage_%j.out
set -euo pipefail
profile_batch_started=$(date +%s)
module load PrgEnv-gnu cudatoolkit/13.2 craype-accel-nvidia80
module unload cray-libsci darshan
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPICH_GPU_SUPPORT_ENABLED=0 BPACK_GPU_AWARE_MPI=0 BPACK_CHECK=''
cd "$SLURM_SUBMIT_DIR"
out="$PWD/profile_results/$SLURM_JOB_ID"; mkdir -p "$out"
export HODLR_PROFILE_RANKS="${HODLR_PROFILE_RANKS:-32}"
n=${HODLR_PROFILE_GRID:-150}; pernode=$((HODLR_PROFILE_RANKS / SLURM_JOB_NUM_NODES))
export LD_LIBRARY_PATH="$HODLR_PROFILE_SNAPSHOT/lib:${LD_LIBRARY_PATH:-}"
cp "$HODLR_PROFILE_SNAPSHOT/manifest.txt" "$out/manifest.txt"
cp "${BASH_SOURCE[0]}" "$out/job.sbatch"
LD_PRELOAD="$HODLR_PROFILE_SNAPSHOT/lib/libbutterflypack.so.5.0.0" ldd "$HODLR_PROFILE_SNAPSHOT/diagnostic" > "$out/libraries.txt"
scontrol show job "$SLURM_JOB_ID" > "$out/slurm_job.txt"
srun -n "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 -c 1 --gpu-bind=none \
  bash -c 'hostname; nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv' > "$out/devices.txt"
read -r -a cases <<< "${HODLR_PROFILE_CASES:-gpu cpu diagnostic traced}"
printf '%s\n' "${cases[@]}" > "$out/cases.txt"
printf '%s\n' "$n" "$HODLR_PROFILE_RANKS" > "$out/layout.txt"
printf '%s\n' "${BPACK_GPU_HEAP_FRACTION:-0.85}" > "$out/heap_fraction.txt"
for variant in "${cases[@]}"; do
  # Short backfill allocations prioritize the requested GPU run. A CPU control
  # is optional and must not start without its measured 150-cubed time budget.
  if [[ "$variant" == cpu && -n "${HODLR_PROFILE_ALLOCATION_SECONDS:-}" ]]; then
    remaining=$((HODLR_PROFILE_ALLOCATION_SECONDS - $(date +%s) + profile_batch_started))
    if (( remaining < ${HODLR_PROFILE_CPU_SLACK:-1000} )); then
      python3 - "$out" "$remaining" <<'PYCODE'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]); plan=(r/'cases.txt').read_text().split()
(r/'planned_cases.txt').write_text('\n'.join(plan)+'\n')
(r/'cases.txt').write_text('\n'.join(x for x in plan if x!='cpu')+'\n')
(r/'skipped_cpu.json').write_text(json.dumps(dict(status='NOT_RUN',reason='Insufficient allocation time for the CPU control',remaining_seconds=int(sys.argv[2])),indent=2)+'\n')
PYCODE
      echo "CPU control not started: only $remaining allocation seconds remain"
      continue
    fi
  fi
  mode=1; rect=1; cap=268435456; binary=baseline; trace=0
  [[ "$variant" == cpu ]] && { mode=0; rect=0; cap=0; }
  [[ "$variant" == diagnostic || "$variant" == traced ]] && binary=diagnostic
  [[ "$variant" == traced ]] && trace=1
  args=("$HODLR_PROFILE_SNAPSHOT/$binary" "$n" 3 --sp_compression HODLR
    --sp_compression_min_sep_size 1000 --sp_compression_min_front_size 1000
    --hodlr_leaf_size 128 --hodlr_rel_tol 1e-5 --hodlr_butterfly_levels 0
    --sp_Krylov_solver refinement --sp_rel_tol 1e-10 --hodlr_use_gpu "$mode"
    --hodlr_gpu_rectangular "$rect" --hodlr_gpu_front_max_bytes "$cap")
  printf '%q ' "${args[@]}" > "$out/$variant.command"; printf '\n' >> "$out/$variant.command"
  status=0
  /usr/bin/time -v -o "$out/$variant.time" timeout --kill-after=60s "${HODLR_PROFILE_TIMEOUT:-3600}s" \
    srun --kill-on-bad-exit=1 --wait=60 -n "$HODLR_PROFILE_RANKS" --ntasks-per-node="$pernode" -c 8 \
    --cpu-bind=cores --gpu-bind=none --output="$out/$variant.rank%t.log" --error="$out/$variant.rank%t.log" \
    bash test/run_hodlr_stage_rank.sh "$out" "$variant" "$trace" "${args[@]}" || status=$?
  printf '%s\n' "$status" > "$out/$variant.exit_code"
  if [[ "$trace" == 1 && "$status" == 0 ]]; then
    report_status=0
    bash test/process_hodlr_stage_traces.sh "$out" "$HODLR_PROFILE_RANKS" || report_status=$?
    printf '%s\n' "$report_status" > "$out/$variant.report_exit_code"
  fi
  python3 test/analyze_hodlr_stages.py "$out"
  (( status == 0 )) || exit "$status"
  echo "$variant completed"
done
python3 test/analyze_hodlr_stages.py "$out" --require-complete
sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,MaxRSS,MaxVMSize,ExitCode -P > "$out/accounting.txt"
