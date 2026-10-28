#!/bin/bash
# Check device WY against host larft, native recompression and sparse fronts.
# HODLR_WY_AFTER is an immutable snapshot containing test, panels and lib/.
#SBATCH -A m4872_g
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH -t 00:15:00
#SBATCH -J hodlr_device_wy_check
#SBATCH -o profile_results/device_wy_check_%j.out
set -euo pipefail
module load PrgEnv-gnu cudatoolkit/13.2 craype-accel-nvidia80
module unload cray-libsci darshan
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BPACK_GPU_HEAP_FRACTION=.85
export MPICH_GPU_SUPPORT_ENABLED=0 BPACK_GPU_AWARE_MPI=0 BPACK_CHECK=''
export BPACK_GPU_BACA_RESIDUAL_GUARD=0 BPACK_GPU_HODLR_SVD=qr
unset STRUMPACK_HODLR_VERIFY_CONSTRUCTION BPACK_GPU_BACA_RESIDUAL_TRACE CUDA_LAUNCH_BLOCKING
cd "$SLURM_SUBMIT_DIR"
export HODLR_PROFILE_SNAPSHOT=${HODLR_WY_AFTER:?}
out="$PWD/profile_results/$SLURM_JOB_ID"; mkdir -p "$out"
cp "$HODLR_WY_AFTER/manifest.txt" "$out/manifest.txt"
cp "${BASH_SOURCE[0]}" "$out/job.sbatch"
scontrol show job "$SLURM_JOB_ID" > "$out/slurm_job.txt"
export HODLR_RANKS_PER_GPU=2 HODLR_PROFILE_RANKS=2
timeout --kill-after=15s 180s srun -n2 -c8 --gpu-bind=none --cpu-bind=cores --wait=0 --kill-on-bad-exit=1 \
  --output="$out/native.rank%t.log" --error="$out/native.rank%t.log" \
  bash test/run_hodlr_stage_rank.sh "$out" native 0 "$HODLR_WY_AFTER/test"
export HODLR_RANKS_PER_GPU=1 HODLR_PROFILE_RANKS=1
timeout --kill-after=15s 240s srun -n1 -c8 --gpu-bind=none --cpu-bind=cores --wait=0 --kill-on-bad-exit=1 \
  --output="$out/benchmark.rank%t.log" --error="$out/benchmark.rank%t.log" \
  bash test/run_hodlr_stage_rank.sh "$out" benchmark 0 "$HODLR_WY_AFTER/test" --wy-benchmark
if [[ -n ${HODLR_WY_CAPTURE_ONE:-} && -n ${HODLR_WY_CAPTURE_TWO:-} ]]; then
  export HODLR_RANKS_PER_GPU=2 HODLR_PROFILE_RANKS=2
  timeout --kill-after=15s 90s srun -n2 -c8 --gpu-bind=none --cpu-bind=cores --wait=0 --kill-on-bad-exit=1 \
    --output="$out/capture.rank%t.log" --error="$out/capture.rank%t.log" \
    bash test/run_hodlr_stage_rank.sh "$out" capture 0 "$HODLR_WY_AFTER/test" "$HODLR_WY_CAPTURE_ONE" "$HODLR_WY_CAPTURE_TWO"
fi
export HODLR_RANKS_PER_GPU=2 HODLR_PROFILE_RANKS=${HODLR_WY_CHECK_RANKS:-8}
(( HODLR_PROFILE_RANKS >= 2 && HODLR_PROFILE_RANKS % 2 == 0 )) || exit 1
printf '%s\n' "$HODLR_PROFILE_RANKS" > "$out/sparse_ranks.txt"
for grid in ${HODLR_WY_CHECK_GRIDS:-40 100}; do
  timeout --kill-after=15s 360s srun -n "$HODLR_PROFILE_RANKS" -c8 --gpu-bind=none --cpu-bind=cores --wait=0 --kill-on-bad-exit=1 \
    --output="$out/g$grid.rank%t.log" --error="$out/g$grid.rank%t.log" \
    bash test/run_hodlr_stage_rank.sh "$out" "g$grid" 0 "$HODLR_WY_AFTER/panels" "$grid" 3 \
    --sp_compression HODLR --sp_compression_min_sep_size 1000 --sp_compression_min_front_size 1000 \
    --hodlr_leaf_size 128 --hodlr_rel_tol 1e-5 --hodlr_butterfly_levels 0 \
    --sp_Krylov_solver refinement --sp_rel_tol 1e-10 --sp_maxit 30 --hodlr_use_gpu 1 \
    --hodlr_gpu_rectangular 1 --hodlr_gpu_front_max_bytes 268435456 --hodlr_gpu_host_panels 1 \
    --hodlr_gpu_panel_max_bytes 67108864
  python3 - "$out/g$grid.rank0.log" <<'PY'
import math,re,sys
text=open(sys.argv[1]).read()
patterns={'relative_error':r'# relative error = .* = (\S+)',
          'scaled_residual':r'# COMPONENTWISE SCALED RESIDUAL = (\S+)'}
values={}
for key,pattern in patterns.items():
    match=re.findall(pattern,text)
    if len(match)!=1: raise SystemExit('Missing or duplicate '+key)
    values[key]=float(match[0])
if not all(math.isfinite(x) for x in values.values()) or not (
        0<=values['relative_error']<1e-8 and 0<=values['scaled_residual']<1e-6):
    raise SystemExit('Sparse accuracy failed: '+str(values))
print('Sparse accuracy PASS: '+str(values))
PY
done
printf 'PASS\n' > "$out/status.txt"
