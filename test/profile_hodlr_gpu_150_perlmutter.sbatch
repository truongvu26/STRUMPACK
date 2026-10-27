#!/bin/bash
#SBATCH -A m4872_g
#SBATCH -C gpu&a100
#SBATCH -q regular
#SBATCH -N 4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH -t 02:00:00
#SBATCH -J hodlr150_stages
#SBATCH -o profile_results/stages150_%j.out
set -euo pipefail
module load PrgEnv-gnu cudatoolkit/13.2 craype-accel-nvidia80
module unload cray-libsci darshan
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPICH_GPU_SUPPORT_ENABLED=0 BPACK_GPU_AWARE_MPI=0 BPACK_CHECK=''
cd "$SLURM_SUBMIT_DIR"
out="$PWD/profile_results/$SLURM_JOB_ID"; mkdir -p "$out/lib"
base="${HODLR_150_BASE_BUILD:-$PWD/../STRUMPACK-gpu-rectangular/build_rectangular_hodlr}"
diag="${HODLR_150_PROFILE_BUILD:-$PWD/build_stage_profile_hodlr}"
bpack="$PWD/../ButterflyPACK-gpu-rectangular"
grid="${HODLR_150_GRID:-150}"; ranks="$SLURM_NTASKS"
read -r -a variants <<< "${HODLR_150_VARIANTS:-gpu_timing cpu_timing gpu_diagnostic gpu_trace}"
cp "${BASH_SOURCE[0]}" "$out/job.sbatch"
git rev-parse HEAD > "$out/source_commit.txt"; git diff --binary HEAD > "$out/source.patch"
git -C "$bpack" rev-parse HEAD > "$out/butterflypack_commit.txt"
git -C "$bpack" diff --binary HEAD > "$out/butterflypack.patch"
cp "$base/examples/sparse/testPoisson3dMPIDist" "$out/solver_baseline"
cp "$diag/examples/sparse/testPoisson3dMPIDist" "$out/solver_diagnostic"
cp -L "$bpack/install_hodlr_gpu/lib64/libbutterflypack.so.5" "$out/lib/libbutterflypack.so.5"
chmod a-w "$out/solver_baseline" "$out/solver_diagnostic" "$out/lib/libbutterflypack.so.5"
sha256sum "$out/solver_baseline" "$out/solver_diagnostic" "$out/lib/libbutterflypack.so.5" > "$out/frozen.sha256"
cp "$base/StrumpackConfig.h" "$out/baseline_config.h"
cp "$diag/StrumpackConfig.h" "$out/diagnostic_config.h"
cp "$diag/CMakeCache.txt" "$out/"
# DT_RPATH precedes LD_LIBRARY_PATH. Preload the exact copied SONAME so every
# case resolves the immutable dependency rather than a mutable installation.
export LD_PRELOAD="$out/lib/libbutterflypack.so.5${LD_PRELOAD:+:$LD_PRELOAD}"
ldd "$out/solver_baseline" > "$out/libraries_baseline.txt"
ldd "$out/solver_diagnostic" > "$out/libraries_diagnostic.txt"
python3 - "$out" "$grid" "$ranks" "${variants[@]}" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]); expected=str(r/'lib/libbutterflypack.so.5')
for name in ('libraries_baseline.txt','libraries_diagnostic.txt'):
    assert expected in (r/name).read_text(), 'Immutable dependency not resolved'
(r/'plan.json').write_text(json.dumps(dict(grid=int(sys.argv[2]),rhs=3,ranks=int(sys.argv[3]),variants=sys.argv[4:],cap_bytes=268435456),indent=2)+'\n')
PY
scontrol show job "$SLURM_JOB_ID" > "$out/slurm_job.txt"
nsys --version > "$out/nsys_version.txt"
srun -n "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 -c1 --gpu-bind=none \
  bash -c 'hostname; nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv' > "$out/devices.txt"
for label in "${variants[@]}"; do
  exe="$out/solver_baseline"; mode=1; rectangular=1; cap=268435456
  [[ "$label" == cpu_timing ]] && { mode=0; rectangular=0; cap=0; }
  [[ "$label" == gpu_diagnostic || "$label" == gpu_trace ]] && exe="$out/solver_diagnostic"
  args=("$exe" "$grid" 3 --sp_compression HODLR
    --sp_compression_min_sep_size 1000 --sp_compression_min_front_size 1000
    --hodlr_leaf_size 128 --hodlr_rel_tol 1e-5 --hodlr_butterfly_levels 0
    --sp_Krylov_solver refinement --sp_rel_tol 1e-10 --hodlr_use_gpu "$mode"
    --hodlr_gpu_rectangular "$rectangular" --hodlr_gpu_front_max_bytes "$cap")
  printf '%q ' "${args[@]}" > "$out/$label.command"
  status=0
  /usr/bin/time -v -o "$out/$label.time" timeout --kill-after=60s "${HODLR_150_CASE_TIMEOUT:-3000}s" \
    srun -n "$ranks" -c8 --cpu-bind=cores --gpu-bind=none --kill-on-bad-exit=1 --wait=60 \
    --output="$out/$label.rank%t.log" --error="$out/$label.rank%t.log" \
    bash test/run_hodlr_150_profile_rank.sh "$out" "$label" "${args[@]}" || status=$?
  printf '%s\n' "$status" > "$out/$label.exit_code"
  python3 test/analyze_hodlr_gpu_150.py "$out"
  (( status==0 )) || exit "$status"
  echo "$label completed"
done
python3 test/analyze_hodlr_gpu_150.py "$out" --require-complete
sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,MaxRSS,ExitCode -P > "$out/accounting.txt"
