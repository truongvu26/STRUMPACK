#!/bin/bash
#SBATCH -A m4872_g
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -t 00:15:00
#SBATCH -J hodlr_gpu_front
#SBATCH -o profile_results/job_%j.out
set -euo pipefail
module load PrgEnv-gnu cudatoolkit/13.2 craype-accel-nvidia80
module unload cray-libsci darshan
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPICH_GPU_SUPPORT_ENABLED=0 BPACK_GPU_AWARE_MPI=0
cd "$SLURM_SUBMIT_DIR"
out="$PWD/profile_results/$SLURM_JOB_ID"; mkdir -p "$out"
build="${HODLR_EVALUATOR_BUILD:-$PWD/build_optimized_hodlr}"
cp "${BASH_SOURCE[0]}" "$out/job.sbatch"
git rev-parse HEAD > "$out/source_commit.txt"
git diff > "$out/source.patch"
cp "$build/CMakeCache.txt" "$build/StrumpackConfig.h" "$out/"
sha256sum "$build/test/test_HODLR_gpu_front_evaluator" \
  "$build/test/test_HODLR_gpu_front_integration" \
  "$build/examples/sparse/testPoisson3dMPIDist" > "$out/executables.sha256"
launch=(srun -n 1 -c 8 --cpu-bind=cores --gpus-per-task=1 --gpu-bind=single:1)
timeout 60s "${launch[@]}" "$build/test/test_HODLR_gpu_front_evaluator" > "$out/device_contract.log" 2>&1
# A small kernel-only test makes a useful sanitizer target without the backend's
# large heaps, MAGMA warm-up or MPI-dependent allocation behavior.
timeout 120s "${launch[@]}" compute-sanitizer --tool memcheck --error-exitcode 3 \
  "$build/test/test_HODLR_gpu_front_evaluator" > "$out/device_memcheck.log" 2>&1
timeout 120s "${launch[@]}" "$build/test/test_HODLR_gpu_front_integration" > "$out/integration.log" 2>&1
for ranks in 1 4; do
  for variant in hybrid evaluator capped; do
    cap=0; [[ "$variant" == evaluator ]] && cap=8388608
    [[ "$variant" == capped ]] && cap=1
    timeout 240s srun -n "$ranks" -c 8 --cpu-bind=cores --gpus-per-task=1 --gpu-bind=single:1 \
      --output="$out/${variant}_${ranks}.rank%t.log" --error="$out/${variant}_${ranks}.rank%t.log" \
      "$build/examples/sparse/testPoisson3dMPIDist" 40 3 --sp_compression HODLR \
      --sp_compression_min_sep_size 1000 --sp_compression_min_front_size 1000 \
      --hodlr_leaf_size 128 --hodlr_rel_tol 1e-5 --hodlr_butterfly_levels 0 \
      --sp_Krylov_solver refinement --sp_rel_tol 1e-10 --hodlr_use_gpu 1 \
      --hodlr_gpu_front_max_bytes "$cap" --hodlr_verbose
  done
done
python3 test/analyze_hodlr_gpu_front.py "$out"
