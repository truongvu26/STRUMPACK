#!/bin/bash
#SBATCH -A m4872_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 00:15:00
#SBATCH -J hodlr_gpu_rect
#SBATCH -o profile_results/rect_%j.out
set -euo pipefail
module load PrgEnv-gnu cudatoolkit/13.2 craype-accel-nvidia80
module unload cray-libsci darshan
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPICH_GPU_SUPPORT_ENABLED=0 BPACK_GPU_AWARE_MPI=0
cd "$SLURM_SUBMIT_DIR"
build="${HODLR_RECTANGULAR_BUILD:-$PWD/build_rectangular_hodlr}"
out="$PWD/profile_results/$SLURM_JOB_ID"; mkdir -p "$out"
cp "${BASH_SOURCE[0]}" "$out/job.sbatch"
git rev-parse HEAD > "$out/source_commit.txt"; git diff > "$out/source.patch"
cp "$build/CMakeCache.txt" "$build/StrumpackConfig.h" "$out/"
sha256sum "$build/test/test_HODLR_gpu_rectangular" "$build/examples/sparse/testPoisson3dMPIDist" > "$out/executables.sha256"
ldd "$build/test/test_HODLR_gpu_rectangular" > "$out/libraries.txt"
bpack="$PWD/../ButterflyPACK-gpu-rectangular"
git -C "$bpack" rev-parse HEAD > "$out/butterflypack_commit.txt"
git -C "$bpack" diff HEAD > "$out/butterflypack.patch"
sha256sum "$bpack/install_hodlr_gpu/lib64/"lib*butterflypack.so* > "$out/butterflypack.sha256"
srun -n1 -c1 --gpus-per-task=1 --gpu-bind=single:1 nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv > "$out/devices.csv"
single=(srun -n1 -c8 --cpu-bind=cores --gpu-bind=none --kill-on-bad-exit=1 --wait=30)
timeout --kill-after=30s 120s "${single[@]}" bash test/run_hodlr_shared_gpu_rank.sh \
  "$build/test/test_HODLR_gpu_front_evaluator" > "$out/device_contract.log" 2>&1
timeout --kill-after=30s 120s "${single[@]}" bash test/run_hodlr_shared_gpu_rank.sh \
  compute-sanitizer --tool memcheck --error-exitcode 3 "$build/test/test_HODLR_gpu_front_evaluator" > "$out/device_memcheck.log" 2>&1
timeout --kill-after=30s 120s "${single[@]}" bash test/run_hodlr_shared_gpu_rank.sh \
  "$build/test/test_HODLR_gpu_front_integration" > "$out/square_integration.log" 2>&1
for ranks in 1 4; do
  launch=(srun -n "$ranks" -c8 --cpu-bind=cores --gpu-bind=none --kill-on-bad-exit=1 --wait=30)
  timeout --kill-after=30s 240s "${launch[@]}" --output="$out/direct_${ranks}.rank%t.log" --error="$out/direct_${ranks}.rank%t.log" \
    bash test/run_hodlr_shared_gpu_rank.sh "$build/test/test_HODLR_gpu_rectangular"
  for variant in hybrid rectangular constructed; do
    rectangular=0; cap=0
    [[ "$variant" != hybrid ]] && rectangular=1
    [[ "$variant" == constructed ]] && cap=67108864
    command=("$build/examples/sparse/testPoisson3dMPIDist" 40 3 --sp_compression HODLR
      --sp_compression_min_sep_size 1000 --sp_compression_min_front_size 1000
      --hodlr_leaf_size 128 --hodlr_rel_tol 1e-5 --hodlr_butterfly_levels 0
      --sp_Krylov_solver refinement --sp_rel_tol 1e-10 --hodlr_use_gpu 1
      --hodlr_gpu_rectangular "$rectangular" --hodlr_gpu_front_max_bytes "$cap")
    printf '%q ' "${command[@]}" > "$out/${variant}_${ranks}.command"
    timeout --kill-after=30s 240s "${launch[@]}" --output="$out/${variant}_${ranks}.rank%t.log" --error="$out/${variant}_${ranks}.rank%t.log" \
      bash test/run_hodlr_shared_gpu_rank.sh "${command[@]}"
    printf '0\n' > "$out/${variant}_${ranks}.exit"
  done
done
timeout --kill-after=30s 240s srun -n3 -c8 --cpu-bind=cores --gpu-bind=none --kill-on-bad-exit=1 --wait=30 \
  --output="$out/direct_3.rank%t.log" --error="$out/direct_3.rank%t.log" \
  bash test/run_hodlr_shared_gpu_rank.sh "$build/test/test_HODLR_gpu_rectangular"
python3 test/analyze_hodlr_gpu_rectangular.py "$out"
