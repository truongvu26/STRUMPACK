#!/bin/bash
#SBATCH -A m4872_g
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 00:10:00
#SBATCH -J hodlr_recompression
#SBATCH -o profile_results/recompression_%j.out
set -euo pipefail
module load PrgEnv-gnu cudatoolkit/13.2 craype-accel-nvidia80
module unload cray-libsci darshan
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BPACK_GPU_HEAP_FRACTION=.85
export MPICH_GPU_SUPPORT_ENABLED=0 BPACK_GPU_AWARE_MPI=0 BPACK_CHECK=''
cd "$SLURM_SUBMIT_DIR"
out="$PWD/profile_results/$SLURM_JOB_ID"; mkdir -p "$out"
cp "$HODLR_RECOMP_SNAPSHOT/manifest.txt" "$out/manifest.txt"
cp "${BASH_SOURCE[0]}" "$out/job.sbatch"
status=0
timeout --kill-after=30s 480s srun -n2 -c8 --gpu-bind=none --cpu-bind=cores --wait=0 --kill-on-bad-exit=1 \
 --output="$out/test.rank%t.log" --error="$out/test.rank%t.log" bash -c '
  IFS=, read -r -a gpu <<< "${CUDA_VISIBLE_DEVICES:-$SLURM_JOB_GPUS}"
  export CUDA_VISIBLE_DEVICES=${gpu[0]}
  export LD_PRELOAD="$HODLR_RECOMP_SNAPSHOT/lib/libbutterflypack.so.5.0.0${LD_PRELOAD:+:$LD_PRELOAD}"
  exec "$HODLR_RECOMP_SNAPSHOT/test"
 ' || status=$?
printf '%s\n' "$status" > "$out/exit_code"
exit "$status"
