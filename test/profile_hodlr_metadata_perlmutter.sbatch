#!/bin/bash
#SBATCH -A m4872_g
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -t 00:15:00
#SBATCH -J hodlr_metadata_ab
#SBATCH -o profile_results/job_%j.out
set -euo pipefail
module load PrgEnv-gnu cudatoolkit/13.2 craype-accel-nvidia80
module unload cray-libsci darshan
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPICH_GPU_SUPPORT_ENABLED=0 BPACK_GPU_AWARE_MPI=0 BPACK_CHECK=''
cd "$SLURM_SUBMIT_DIR"
out="$PWD/profile_results/$SLURM_JOB_ID"
mkdir -p "$out"
baseline="${HODLR_BASELINE_EXE:?Set HODLR_BASELINE_EXE to the original hybrid Release binary}"
optimized="$PWD/build_optimized_hodlr/examples/sparse/testPoisson3dMPIDist"
bpack="${HODLR_OPTIMIZED_BPACK_SOURCE:?Set HODLR_OPTIMIZED_BPACK_SOURCE}"
cp "${BASH_SOURCE[0]}" "$out/job.sbatch"
git rev-parse HEAD > "$out/source_commit.txt"
git -C "$bpack" rev-parse HEAD > "$out/bpack_commit.txt"
sha256sum "$baseline" "$optimized" "$bpack/build_gpu_regressions/metadata_lifetime_old" \
  "$bpack/build_gpu_regressions/metadata_lifetime_new" > "$out/executables.sha256"
cp build_optimized_hodlr/CMakeCache.txt build_optimized_hodlr/StrumpackConfig.h "$out/"
python3 - "$baseline" "$optimized" "$bpack" <<'PYCODE'
from pathlib import Path
import re,subprocess,sys
paths=[]
for exe in sys.argv[1:3]:
 text=subprocess.check_output(['ldd',exe],universal_newlines=True)
 found=re.search(r'libbutterflypack\.so\S* => (\S+)',text)
 if not found:raise SystemExit('ButterflyPACK library missing from executable dependencies')
 paths.append(str(Path(found.group(1)).resolve()))
if paths[0]==paths[1]:raise SystemExit('A/B executables resolve the same ButterflyPACK library')
if not paths[1].startswith(str(Path(sys.argv[3]).resolve())):
 raise SystemExit('Optimized executable resolves an unexpected ButterflyPACK library')
print('Verified distinct original and optimized ButterflyPACK dependencies.')
PYCODE
{
  date -u
  hostname
  module list 2>&1
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  nsys --version
  ldd "$baseline"
  ldd "$optimized"
} > "$out/environment.txt"
# The same allocation and rank/GPU binding are used for every A/B trial.
for repeat in 1 2 3; do
  if (( repeat % 2 )); then variants=(old_cpu old_gpu new_gpu); else variants=(new_gpu old_gpu old_cpu); fi
  for variant in "${variants[@]}"; do
    exe="$baseline"; mode=1
    if [[ "$variant" == old_cpu ]]; then mode=0; fi
    if [[ "$variant" == new_gpu ]]; then exe="$optimized"; fi
    name="${variant}_r${repeat}"
    args=("$exe" 60 3 --sp_compression HODLR
      --sp_compression_min_sep_size 1000 --sp_compression_min_front_size 1000
      --hodlr_leaf_size 128 --hodlr_butterfly_levels 0 --hodlr_rel_tol 1e-4
      --sp_rel_tol 1e-10 --sp_Krylov_solver refinement --hodlr_use_gpu "$mode")
    printf '%q ' srun -n 4 -c 8 --cpu-bind=cores --gpus-per-task=1 --gpu-bind=single:1 \
      "${args[@]}" > "$out/$name.command"
    printf '\n' >> "$out/$name.command"
    timeout 240s srun -n 4 -c 8 --cpu-bind=cores --gpus-per-task=1 --gpu-bind=single:1 \
      --output="$out/$name.rank%t.log" --error="$out/$name.rank%t.log" "${args[@]}"
    echo "$name completed"
  done
done
# The old header must reproduce the pinned-metadata regression; the fixed one must pass.
if timeout 30s srun -n 1 -c 2 --gpus-per-task=1 --gpu-bind=single:1 \
    "$bpack/build_gpu_regressions/metadata_lifetime_old" > "$out/metadata_old.log" 2>&1; then
  echo 'Old metadata unexpectedly passed the regression.' >&2; exit 1
fi
rg -q 'small image pins more than 1 MiB' "$out/metadata_old.log"
timeout 30s srun -n 1 -c 2 --gpus-per-task=1 --gpu-bind=single:1 \
  "$bpack/build_gpu_regressions/metadata_lifetime_new" > "$out/metadata_new.log" 2>&1
rg -q 'Metadata lifetime: PASS' "$out/metadata_new.log"
# Validate the existing equal-share mechanism on a fixed full solver group.
timeout 90s srun -n 4 -c 8 --cpu-bind=cores --gpu-bind=none \
  --output="$out/shared_sparse.rank%t.log" --error="$out/shared_sparse.rank%t.log" \
  bash test/run_hodlr_shared_gpu_rank.sh "$optimized" 40 3 --sp_compression HODLR \
  --sp_compression_min_sep_size 1000 --sp_compression_min_front_size 1000 \
  --hodlr_leaf_size 128 --hodlr_butterfly_levels 0 --hodlr_rel_tol 1e-4 \
  --sp_rel_tol 1e-10 --sp_Krylov_solver refinement --hodlr_use_gpu 1 --hodlr_verbose
timeout 90s srun -n 4 -c 4 --gpu-bind=none bash test/run_hodlr_shared_gpu_rank.sh \
  build_optimized_hodlr/test/test_HODLR_gpu_interface_mpi --hodlr_use_gpu 1 --hodlr_verbose \
  > "$out/shared_api.log" 2>&1
rg -q 'HODLR hybrid interface: PASS' "$out/shared_api.log"
# Retain a CUDA trace to verify the measured pinned-allocation cost after the fix.
timeout 240s srun -n 4 -c 8 --cpu-bind=cores --gpus-per-task=1 --gpu-bind=single:1 \
  --output="$out/traced_gpu.rank%t.log" --error="$out/traced_gpu.rank%t.log" \
  bash test/run_hodlr_nsys_rank.sh "$out" "$optimized" 60 3 --sp_compression HODLR \
  --sp_compression_min_sep_size 1000 --sp_compression_min_front_size 1000 \
  --hodlr_leaf_size 128 --hodlr_butterfly_levels 0 --hodlr_rel_tol 1e-4 \
  --sp_rel_tol 1e-10 --sp_Krylov_solver refinement --hodlr_use_gpu 1 --hodlr_verbose
python3 test/analyze_hodlr_metadata_profile.py "$out"
