#!/bin/bash
#SBATCH -A m4872_g
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 00:10:00
#SBATCH -J hodlr_capture_accuracy
#SBATCH -o profile_results/accuracy_%j.out
set -euo pipefail
module load PrgEnv-gnu cudatoolkit/13.2 craype-accel-nvidia80
module unload cray-libsci darshan
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 BPACK_GPU_HEAP_FRACTION=.85
export MPICH_GPU_SUPPORT_ENABLED=0 BPACK_GPU_AWARE_MPI=0 BPACK_CHECK=''
export HODLR_PROFILE_RANKS=4 STRUMPACK_HODLR_VERIFY_CONSTRUCTION=1e-3
cd "$SLURM_SUBMIT_DIR"
out="$PWD/profile_results/$SLURM_JOB_ID";mkdir -p "$out"
cp "$HODLR_PROFILE_SNAPSHOT/manifest.txt" "$out/manifest.txt"
cp "${BASH_SOURCE[0]}" "$out/job.sbatch"
read -r -a cases <<< "${HODLR_ACCURACY_CASES:-cpu gpu1 gpu2 gpu3 gpu4 gpu5}"
for variant in "${cases[@]}"; do
 mode=1;cap=268435456;rect=1;panels=${HODLR_ACCURACY_HOST_PANELS:-0}
 [[ $variant != cpu ]] || { mode=0;cap=0;rect=0;panels=0; }
 export STRUMPACK_HODLR_CHECK_DUMP="$out/$variant"
 status=0
 timeout --kill-after=30s "${HODLR_ACCURACY_TIMEOUT:-240}s" srun -n4 -c8 --gpu-bind=none --cpu-bind=cores --wait=0 --kill-on-bad-exit=1 \
  --output="$out/$variant.rank%t.log" --error="$out/$variant.rank%t.log" \
  bash test/run_hodlr_stage_rank.sh "$out" "$variant" 0 "$HODLR_PROFILE_SNAPSHOT/poisson" "${HODLR_ACCURACY_GRID:-40}" 3 \
  --sp_compression HODLR --sp_compression_min_sep_size 1000 --sp_compression_min_front_size 1000 \
  --hodlr_leaf_size 128 --hodlr_rel_tol 1e-5 --hodlr_butterfly_levels 0 \
  --sp_Krylov_solver refinement --sp_rel_tol 1e-10 --sp_maxit 30 --hodlr_use_gpu "$mode" \
  --hodlr_gpu_rectangular "$rect" --hodlr_gpu_front_max_bytes "$cap" --hodlr_gpu_host_panels "$panels" || status=$?
 printf '%s\n' "$status" > "$out/$variant.exit_code"
 # Stop at the first discrepancy. A completed reproducer is not a numerical PASS.
 if ((status));then exit "$status";fi
done
