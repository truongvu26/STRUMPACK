#!/bin/bash
# Release-build regression: independent recompression, replay, then sparse
# child-inclusive construction on multiple fronts, ranks and GPUs.
#SBATCH -A m4872_g
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 00:10:00
#SBATCH -J hodlr_baca_sparse
#SBATCH -o profile_results/baca_sparse_%j.out
set -euo pipefail
module load PrgEnv-gnu cudatoolkit/13.2 craype-accel-nvidia80
module unload cray-libsci darshan
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BPACK_GPU_HEAP_FRACTION=.85
export MPICH_GPU_SUPPORT_ENABLED=0 BPACK_GPU_AWARE_MPI=0 BPACK_CHECK=''
export BPACK_GPU_BACA_RESIDUAL_GUARD=1 BPACK_GPU_BACA_RESIDUAL_LOG=1
export BPACK_GPU_HODLR_SVD=qr
unset BPACK_GPU_BACA_RESIDUAL_TRACE CUDA_LAUNCH_BLOCKING
cd "$SLURM_SUBMIT_DIR"
out="$PWD/profile_results/$SLURM_JOB_ID"; mkdir -p "$out"
cp "$HODLR_PROFILE_SNAPSHOT/manifest.txt" "$out/manifest.txt"
cp "${BASH_SOURCE[0]}" "$out/job.sbatch"
if [[ ${HODLR_BACA_SKIP_COMPONENTS:-0} != 1 ]]; then
if [[ ${HODLR_BACA_SKIP_RECOMPRESSION:-0} != 1 ]]; then
timeout --kill-after=15s 90s srun -n2 -c8 --gpu-bind=none --cpu-bind=cores --wait=0 --kill-on-bad-exit=1 \
  --output="$out/recompression.rank%t.log" --error="$out/recompression.rank%t.log" bash -c '
    IFS=, read -r -a gpu <<< "${CUDA_VISIBLE_DEVICES:-$SLURM_JOB_GPUS}"
    export CUDA_VISIBLE_DEVICES=${gpu[0]}
    export LD_PRELOAD="$HODLR_PROFILE_SNAPSHOT/lib/libbutterflypack.so.5.0.0"
    exec "$HODLR_PROFILE_SNAPSHOT/recompression"
  '
fi
export HODLR_REPLAY_SNAPSHOT=$HODLR_PROFILE_SNAPSHOT
export STRUMPACK_HODLR_REPLAY_REQUIRE_GPU_ACCURACY=1 STRUMPACK_HODLR_REPLAY_TRIALS=4
for input in "$HODLR_REPLAY_INPUT" --localized-outlier; do
  case_name=capture; [[ $input != --localized-outlier ]] || case_name=localized
  timeout --kill-after=15s 60s srun -n2 -c8 --gpu-bind=none --cpu-bind=cores --wait=0 --kill-on-bad-exit=1 \
    --output="$out/$case_name.rank%t.log" --error="$out/$case_name.rank%t.log" bash -c '
      IFS=, read -r -a gpu <<< "${CUDA_VISIBLE_DEVICES:-$SLURM_JOB_GPUS}"
      export CUDA_VISIBLE_DEVICES=${gpu[0]}
      export LD_PRELOAD="$HODLR_REPLAY_SNAPSHOT/lib/libbutterflypack.so.5.0.0"
      exec "$HODLR_REPLAY_SNAPSHOT/test" "$1"
    ' bash "$input"
done
fi
export HODLR_PROFILE_RANKS=4 STRUMPACK_HODLR_VERIFY_CONSTRUCTION=1e-3
for grid in ${HODLR_BACA_SPARSE_GRIDS:-40 100}; do
  export STRUMPACK_HODLR_CHECK_DUMP="$out/g$grid"
  timeout --kill-after=15s 360s srun -n4 -c8 --gpu-bind=none --cpu-bind=cores --wait=0 --kill-on-bad-exit=1 \
    --output="$out/g$grid.rank%t.log" --error="$out/g$grid.rank%t.log" \
    bash test/run_hodlr_stage_rank.sh "$out" "g$grid" 0 "$HODLR_PROFILE_SNAPSHOT/poisson" "$grid" 3 \
    --sp_compression HODLR --sp_compression_min_sep_size 1000 --sp_compression_min_front_size 1000 \
    --hodlr_leaf_size 128 --hodlr_rel_tol 1e-5 --hodlr_butterfly_levels 0 \
    --sp_Krylov_solver refinement --sp_rel_tol 1e-10 --sp_maxit 30 --hodlr_use_gpu 1 \
    --hodlr_gpu_rectangular 1 --hodlr_gpu_front_max_bytes 268435456 --hodlr_gpu_host_panels 1
  python3 - "$out/g$grid.rank0.log" <<'PY'
import math, re, sys
text = open(sys.argv[1]).read()
error = re.search(r'\|\|x-x_exact\|\|_F/\|\|x_exact\|\|_F = (\S+)', text)
residual = re.search(r'COMPONENTWISE SCALED RESIDUAL = (\S+)', text)
if not error or not residual:
    raise SystemExit('Missing final sparse accuracy result')
error, residual = float(error.group(1)), float(residual.group(1))
if not (math.isfinite(error) and math.isfinite(residual) and error < 1e-7 and residual < 1e-6):
    raise SystemExit('Sparse accuracy failed: error=%g residual=%g' % (error, residual))
print('Sparse accuracy PASS: error=%g residual=%g' % (error, residual))
PY
done
