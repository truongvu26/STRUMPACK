#!/bin/bash
#SBATCH -A m4872_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH -t 00:20:00
#SBATCH -J hodlr_profile
#SBATCH -o profile_results/job_%j.out
set -euo pipefail
module load PrgEnv-gnu cudatoolkit/13.2 craype-accel-nvidia80
module unload cray-libsci darshan
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPICH_GPU_SUPPORT_ENABLED=0 BPACK_GPU_AWARE_MPI=0 BPACK_CHECK=''
cd "$SLURM_SUBMIT_DIR"
out="$PWD/profile_results/$SLURM_JOB_ID"
mkdir -p "$out"
baseline="${HODLR_BASELINE_EXE:-$PWD/../STRUMPACK-gpu-interface/build_gpu_hodlr/examples/sparse/testPoisson3dMPIDist}"
profile="$PWD/build_profile_hodlr/examples/sparse/testPoisson3dMPIDist"
cp "${BASH_SOURCE[0]}" "$out/job.sbatch"
git rev-parse HEAD > "$out/source_commit.txt"
git diff --binary HEAD > "$out/source.patch"
sha256sum "$baseline" "$profile" > "$out/executables.sha256"
{
  date -u
  hostname
  module list 2>&1
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  nsys --version
  ldd "$profile"
} > "$out/environment.txt"
cp build_profile_hodlr/CMakeCache.txt build_profile_hodlr/StrumpackConfig.h "$out/"
run_case() {
  local name=$1 exe=$2 n=$3 mode=$4 traced=$5
  local args=("$exe" "$n" 3 --sp_compression HODLR
    --sp_compression_min_sep_size 1000 --sp_compression_min_front_size 1000
    --hodlr_leaf_size 128 --hodlr_butterfly_levels 0 --hodlr_rel_tol 1e-4
    --sp_rel_tol 1e-10 --sp_Krylov_solver refinement --hodlr_use_gpu "$mode")
  local launcher=()
  if [[ "$traced" == 1 ]]; then
    launcher=(bash "$PWD/test/run_hodlr_nsys_rank.sh" "$out")
  fi
  if [[ "$name" != baseline* ]]; then args+=(--hodlr_verbose); fi
  printf '%q ' timeout 240s srun -n 4 -c 8 --cpu-bind=cores --gpus-per-task=1 \
    --gpu-bind=single:1 "${launcher[@]}" "${args[@]}" > "$out/$name.command"
  printf '\n' >> "$out/$name.command"
  timeout 240s srun -n 4 -c 8 --cpu-bind=cores --gpus-per-task=1 --gpu-bind=single:1 \
    --output="$out/$name.rank%t.log" --error="$out/$name.rank%t.log" \
    "${launcher[@]}" "${args[@]}"
  echo "$name completed"
}
# Alternate CPU/GPU ordering. Every trial starts a fresh solver process.
for n in 40 60; do
  for repeat in 1 2 3; do
    if (( repeat % 2 )); then modes=(0 1); else modes=(1 0); fi
    for mode in "${modes[@]}"; do
      run_case "baseline_n${n}_mode${mode}_r${repeat}" "$baseline" "$n" "$mode" 0
    done
  done
done
run_case detailed_cpu "$profile" 60 0 0
run_case detailed_gpu "$profile" 60 1 0
run_case traced_gpu "$profile" 60 1 1
python3 test/analyze_hodlr_gpu_profile.py "$out"
