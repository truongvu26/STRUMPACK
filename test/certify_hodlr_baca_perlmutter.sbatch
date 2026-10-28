#!/bin/bash
# GPU-only assertions; CPU cases remain a control for the shared BACA weakness.
#SBATCH -A m4872_g
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 00:10:00
#SBATCH -J hodlr_baca_certify
#SBATCH -o profile_results/certify_%j.out
set -euo pipefail
module load PrgEnv-gnu cudatoolkit/13.2 craype-accel-nvidia80
module unload cray-libsci darshan
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BPACK_GPU_HEAP_FRACTION=.85
export MPICH_GPU_SUPPORT_ENABLED=0 BPACK_GPU_AWARE_MPI=0 BPACK_CHECK=''
export BPACK_GPU_BACA_RESIDUAL_GUARD=1
export STRUMPACK_HODLR_REPLAY_REQUIRE_GPU_ACCURACY=1
export STRUMPACK_HODLR_REPLAY_TRIALS=${STRUMPACK_HODLR_REPLAY_TRIALS:-16}
unset STRUMPACK_HODLR_VERIFY_CONSTRUCTION STRUMPACK_HODLR_CHECK_DUMP
cd "$SLURM_SUBMIT_DIR"
out="$PWD/profile_results/$SLURM_JOB_ID"; mkdir -p "$out"
cp "$HODLR_REPLAY_SNAPSHOT/manifest.txt" "$out/manifest.txt"
cp "${BASH_SOURCE[0]}" "$out/job.sbatch"
for case_name in capture localized; do
  input=$HODLR_REPLAY_INPUT
  [[ $case_name != localized ]] || input=--localized-outlier
  timeout --kill-after=30s 240s srun -n2 -c8 --gpu-bind=none --cpu-bind=cores --wait=0 --kill-on-bad-exit=1 \
    --output="$out/$case_name.rank%t.log" --error="$out/$case_name.rank%t.log" bash -c '
      IFS=, read -r -a gpu <<< "${CUDA_VISIBLE_DEVICES:-$SLURM_JOB_GPUS}"
      export CUDA_VISIBLE_DEVICES=${gpu[0]}
      export LD_PRELOAD="$HODLR_REPLAY_SNAPSHOT/lib/libbutterflypack.so.5.0.0${LD_PRELOAD:+:$LD_PRELOAD}"
      exec "$HODLR_REPLAY_SNAPSHOT/test" "$1"
    ' bash "$input"
done
