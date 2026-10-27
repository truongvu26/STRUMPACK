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
#SBATCH -J hodlr_large_poisson
#SBATCH -o profile_results/scale_%j.out
set -euo pipefail
module load PrgEnv-gnu cudatoolkit/13.2 craype-accel-nvidia80
module unload cray-libsci darshan
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPICH_GPU_SUPPORT_ENABLED=0 BPACK_GPU_AWARE_MPI=0 BPACK_CHECK=''
cd "$SLURM_SUBMIT_DIR"
out="$PWD/profile_results/$SLURM_JOB_ID"; mkdir -p "$out"
build="${HODLR_EVALUATOR_BUILD:-$PWD/build_optimized_hodlr}"
exe="$build/examples/sparse/testPoisson3dMPIDist"
ranks=$((SLURM_JOB_NUM_NODES * 8))
read -r -a grids <<< "${HODLR_SCALE_GRIDS:-100 150 200}"
repeats="${HODLR_SCALE_REPEATS:-3}"
[[ "$repeats" =~ ^[1-3]$ ]] || { echo 'Repetitions must be 1, 2 or 3.' >&2; exit 1; }
for n in "${grids[@]}"; do
  case "$n" in 100|150|200) ;; *) echo "Unsupported grid $n" >&2; exit 1;; esac
done
python3 - "$out/plan.json" "$repeats" "${grids[@]}" <<'PYCODE'
import json,sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps(dict(repetitions=int(sys.argv[2]),grids=list(map(int,sys.argv[3:]))),indent=2)+'\n')
PYCODE
cp "${BASH_SOURCE[0]}" "$out/job.sbatch"
git rev-parse HEAD > "$out/source_commit.txt"
git diff --binary HEAD > "$out/source.patch"
sha256sum "$exe" > "$out/executable.sha256"
cp "$build/CMakeCache.txt" "$build/StrumpackConfig.h" "$out/"
scontrol show job "$SLURM_JOB_ID" > "$out/slurm_job.txt"
ldd "$exe" > "$out/libraries.txt"
srun -n "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 -c 1 --gpu-bind=none \
  bash -c 'hostname; nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv' > "$out/devices.txt"
# Each case is a fresh MPI job step. Capture status and elapsed/RSS externally
# without enabling ButterflyPACK's costly native entry/product error checks.
for n in "${grids[@]}"; do
  for (( repeat=1; repeat<=repeats; repeat++ )); do
    if (( repeat % 2 )); then variants=(cpu hybrid evaluator); else variants=(evaluator hybrid cpu); fi
    for variant in "${variants[@]}"; do
      mode=1; cap=0
      [[ "$variant" == cpu ]] && mode=0
      [[ "$variant" == evaluator ]] && cap=8388608
      name="n${n}_${variant}_r${repeat}"
      args=("$exe" "$n" 3 --sp_compression HODLR
        --sp_compression_min_sep_size 1000 --sp_compression_min_front_size 1000
        --hodlr_leaf_size 128 --hodlr_rel_tol 1e-5 --hodlr_butterfly_levels 0
        --sp_Krylov_solver refinement --sp_rel_tol 1e-10 --hodlr_use_gpu "$mode"
        --hodlr_gpu_front_max_bytes "$cap")
      printf '%q ' srun -n "$ranks" --ntasks-per-node=8 -c 8 --cpu-bind=cores --gpu-bind=none \
        bash test/run_hodlr_scale_rank.sh "${args[@]}" > "$out/$name.command"
      printf '\n' >> "$out/$name.command"
      status=0
      /usr/bin/time -v -o "$out/$name.time" \
        srun -n "$ranks" --ntasks-per-node=8 -c 8 --cpu-bind=cores --gpu-bind=none \
        --output="$out/$name.rank%t.log" --error="$out/$name.rank%t.log" \
        bash test/run_hodlr_scale_rank.sh "${args[@]}" || status=$?
      printf '%s\n' "$status" > "$out/$name.exit_code"
      python3 test/analyze_hodlr_gpu_scale.py "$out" --ranks "$ranks"
      if (( status )); then exit "$status"; fi
      echo "$name completed"
    done
  done
done
python3 test/analyze_hodlr_gpu_scale.py "$out" --ranks "$ranks" --require-complete
sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,MaxRSS,MaxVMSize,ExitCode -P > "$out/accounting.txt"
