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
#SBATCH -J hodlr_rect_profile
#SBATCH -o profile_results/rect_profile_%j.out
set -euo pipefail
module load PrgEnv-gnu cudatoolkit/13.2 craype-accel-nvidia80
module unload cray-libsci darshan
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPICH_GPU_SUPPORT_ENABLED=0 BPACK_GPU_AWARE_MPI=0
cd "$SLURM_SUBMIT_DIR"
build="${HODLR_RECTANGULAR_BUILD:-$PWD/build_rectangular_hodlr}"
out="$PWD/profile_results/$SLURM_JOB_ID"; mkdir -p "$out"
cp "${BASH_SOURCE[0]}" "$out/job.sbatch"
git rev-parse HEAD > "$out/source_commit.txt"; git diff HEAD > "$out/source.patch"
cp "$build/CMakeCache.txt" "$build/StrumpackConfig.h" "$out/"
sha256sum "$build/examples/sparse/testPoisson3dMPIDist" > "$out/executables.sha256"
bpack="$PWD/../ButterflyPACK-gpu-rectangular"
git -C "$bpack" rev-parse HEAD > "$out/butterflypack_commit.txt"
git -C "$bpack" diff HEAD > "$out/butterflypack.patch"
sha256sum "$bpack/install_hodlr_gpu/lib64/"lib*butterflypack.so* > "$out/butterflypack.sha256"
srun -n1 -c1 --gpus-per-task=1 --gpu-bind=single:1 nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv > "$out/devices.csv"
launch=(srun -n4 -c8 --cpu-bind=cores --gpu-bind=none --kill-on-bad-exit=1 --wait=30)
for repeat in 1 2 3; do
  variants=(cpu hybrid rectangular constructed)
  (( repeat%2==0 )) && variants=(constructed rectangular hybrid cpu)
  for variant in "${variants[@]}"; do
    mode=1; rectangular=0; cap=0
    [[ "$variant" == cpu ]] && mode=0
    [[ "$variant" == rectangular || "$variant" == constructed ]] && rectangular=1
    [[ "$variant" == constructed ]] && cap=268435456
    label="${variant}_r${repeat}"
    command=("$build/examples/sparse/testPoisson3dMPIDist" 60 3 --sp_compression HODLR
      --sp_compression_min_sep_size 1000 --sp_compression_min_front_size 1000
      --hodlr_leaf_size 128 --hodlr_rel_tol 1e-5 --hodlr_butterfly_levels 0
      --sp_Krylov_solver refinement --sp_rel_tol 1e-10 --hodlr_use_gpu "$mode"
      --hodlr_gpu_rectangular "$rectangular" --hodlr_gpu_front_max_bytes "$cap")
    printf '%q ' "${command[@]}" > "$out/$label.command"
    timeout --kill-after=30s 240s "${launch[@]}" --output="$out/$label.rank%t.log" --error="$out/$label.rank%t.log" \
      bash test/run_hodlr_shared_gpu_rank.sh "${command[@]}"
    printf '0\n' > "$out/$label.exit"
  done
done
python3 test/analyze_hodlr_rectangular_profile.py "$out"
