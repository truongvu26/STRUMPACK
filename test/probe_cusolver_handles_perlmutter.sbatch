#!/bin/bash
#SBATCH -A m4872_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH -t 00:05:00
#SBATCH -J cusolver_resources
#SBATCH -o profile_results/handles_%j.out
set -euo pipefail
module load PrgEnv-gnu cudatoolkit/13.2 craype-accel-nvidia80
module unload cray-libsci darshan
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
out="$PWD/profile_results/handles_$SLURM_JOB_ID"
mkdir -p "$out"
for fraction in .85 .50; do
  srun --kill-on-bad-exit=1 -n2 -c8 --gpu-bind=none --cpu-bind=cores \
    --output="$out/fraction$fraction.rank%t.log" \
    python3 test/probe_cusolver_handles.py "$out/$fraction" --fraction "$fraction"
done
