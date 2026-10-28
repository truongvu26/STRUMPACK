#!/bin/bash
#SBATCH -A m4872_g
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH --mem=0
#SBATCH -t 00:25:00
#SBATCH -J hodlr_blas_threads
#SBATCH -o profile_results/blas_threads_%j.out
# Keep solver OpenMP serial; vary only the frozen pthread BLAS worker count.
set -euo pipefail
module load PrgEnv-gnu cudatoolkit/13.2 craype-accel-nvidia80
module unload cray-libsci darshan
cd "$SLURM_SUBMIT_DIR"
export HODLR_PROFILE_SNAPSHOT="${HODLR_BLAS_SNAPSHOT:-$PWD/profile_results/device_wy_snapshot_v1}"
export HODLR_BLAS_LIBRARY="${HODLR_BLAS_LIBRARY:-$PWD/profile_results/openblas_pthreads_snapshot_v1/lib/libopenblas.so.0.3}"
HODLR_PROFILE_SNAPSHOT=$(realpath "$HODLR_PROFILE_SNAPSHOT");HODLR_BLAS_LIBRARY=$(realpath "$HODLR_BLAS_LIBRARY")
export HODLR_PROFILE_RANKS="${HODLR_BLAS_RANKS:-4}" HODLR_RANKS_PER_GPU="${HODLR_RANKS_PER_GPU:-1}"
export HODLR_BLAS_GRID="${HODLR_BLAS_GRID:-100}"
export OMP_NUM_THREADS=1 MPICH_GPU_SUPPORT_ENABLED=0 BPACK_GPU_AWARE_MPI=0 BPACK_CHECK=''
export BPACK_GPU_HEAP_FRACTION=0.85 BPACK_GPU_BACA_RESIDUAL_GUARD=0 BPACK_GPU_HODLR_SVD=qr
unset BPACK_GPU_BACA_RESIDUAL_LOG BPACK_GPU_BACA_RESIDUAL_TRACE STRUMPACK_HODLR_VERIFY_CONSTRUCTION CUDA_LAUNCH_BLOCKING STRUMPACK_HODLR_STAGE_FILE
case "$HODLR_RANKS_PER_GPU" in 1|2) ;; *) exit 1 ;; esac
case "${HODLR_BLAS_INCLUDE_16:-0}" in 0|1) ;; *) echo 'HODLR_BLAS_INCLUDE_16 must be 0 or 1' >&2;exit 1 ;; esac
[[ "$HODLR_PROFILE_RANKS" =~ ^[1-9][0-9]*$ && "$HODLR_BLAS_GRID" =~ ^[1-9][0-9]*$ ]] || exit 1
pernode=$((HODLR_PROFILE_RANKS/SLURM_JOB_NUM_NODES))
(( HODLR_PROFILE_RANKS % SLURM_JOB_NUM_NODES == 0 && pernode > 0 && pernode % HODLR_RANKS_PER_GPU == 0 )) || {
  echo 'MPI ranks must divide evenly over nodes and complete each GPU sharing group' >&2;exit 1;
}
threads=(1 4 8)
[[ ${HODLR_BLAS_INCLUDE_16:-0} == 0 ]] || threads+=(16)
threads+=(1)
max_threads=8;[[ ${HODLR_BLAS_INCLUDE_16:-0} == 0 ]] || max_threads=16
(( ${SLURM_CPUS_PER_TASK:-16} >= max_threads )) || { echo 'Allocate at least as many CPUs per task as BLAS workers' >&2;exit 1; }
started=$(date +%s)
out="$PWD/profile_results/$SLURM_JOB_ID";mkdir -p "$out"
cp "${BASH_SOURCE[0]}" "$out/job.sbatch"
scontrol show job "$SLURM_JOB_ID" > "$out/slurm_job.txt"
for index in "${!threads[@]}";do printf '%02d-t%d %d\n' "$index" "${threads[$index]}" "${threads[$index]}";done > "$out/thread_order.txt"
python3 - "$out/run_manifest.json" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=Path(os.environ['HODLR_PROFILE_SNAPSHOT']);b=Path(os.environ['HODLR_BLAS_LIBRARY'])
d=dict(snapshot=str(p),blas_library=str(b),executable_sha256=sha(p/'panels'),native_library_sha256=sha(p/'lib/libbutterflypack.so.5.0.0'),
    blas_library_sha256=sha(b),snapshot_manifest=(p/'manifest.txt').read_text(),
    grid=int(os.environ['HODLR_BLAS_GRID']),ranks=int(os.environ['HODLR_PROFILE_RANKS']),nodes=int(os.environ['SLURM_JOB_NUM_NODES']),
    ranks_per_gpu=int(os.environ['HODLR_RANKS_PER_GPU']),cpus_per_task=int(os.environ.get('SLURM_CPUS_PER_TASK','16')),
    hint=os.environ.get('SLURM_HINT','nomultithread requested in job script'),require_physical_cores=True,
    rhs=3,omp_threads=1,leaf_size=128,compression_tolerance=1e-5,refinement_tolerance=1e-10,butterfly_levels=0,
    residual_certification=False,svd='qr',gpu_heap_fraction=.85,host_panel_budget_bytes=67108864,front_image_cap_bytes=268435456,
    threading_scope='Only pthread BLAS workers vary. Solver OpenMP remains1; audited STRUMPACK and ButterflyPACK contain no explicit BLAS thread-count overrides.')
Path(sys.argv[1]).write_text(json.dumps(d,indent=2)+'\n')
PY
srun -n "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 -c1 --gpu-bind=none bash -c \
  'hostname;nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv' > "$out/devices.txt"
index=0;first_elapsed=0
while read -r label workers;do
  dir="$out/$label";mkdir -p "$dir"
  remaining=$((${HODLR_BLAS_ALLOCATION_SECONDS:-1500}-$(date +%s)+started));limit=$((remaining-90))
  if (( index == ${#threads[@]}-1 && limit < first_elapsed*6/5+30 ));then
    printf 'Optional final single-thread repeat skipped: insufficient remaining budget\n' > "$dir/skipped_budget.txt"
    break
  fi
  (( limit > 0 )) || { echo 'No time remains for the next primary case';exit 1; }
  export HODLR_OPENBLAS_THREADS="$workers" OPENBLAS_NUM_THREADS="$workers"
  printf '%s\n' "$HODLR_BLAS_GRID" "$HODLR_PROFILE_RANKS" > "$dir/layout.txt"
  printf 'panels\n' > "$dir/cases.txt";printf '0.85\n' > "$dir/heap_fraction.txt"
  printf '%s\n' "$HODLR_RANKS_PER_GPU" > "$dir/rank_per_gpu.txt"
  printf '%s\n' "$workers" > "$dir/blas_threads.txt";cp "$out/devices.txt" "$dir/devices.txt"
  sha256sum "$HODLR_PROFILE_SNAPSHOT/panels" "$HODLR_PROFILE_SNAPSHOT/lib/libbutterflypack.so.5.0.0" "$HODLR_BLAS_LIBRARY" > "$dir/binary_hashes.txt"
  args=("$HODLR_PROFILE_SNAPSHOT/panels" "$HODLR_BLAS_GRID" 3 --sp_compression HODLR
    --sp_compression_min_sep_size 1000 --sp_compression_min_front_size 1000 --hodlr_leaf_size 128
    --hodlr_rel_tol 1e-5 --hodlr_butterfly_levels 0 --sp_Krylov_solver refinement --sp_rel_tol 1e-10 --sp_maxit 30
    --hodlr_use_gpu 1 --hodlr_gpu_rectangular 1 --hodlr_gpu_front_max_bytes 268435456
    --hodlr_gpu_host_panels 1 --hodlr_gpu_panel_max_bytes 67108864)
  printf '%q ' "${args[@]}" > "$dir/panels.command";printf '\n' >> "$dir/panels.command"
  case_started=$(date +%s);status=0
  /usr/bin/time -v -o "$dir/panels.time" timeout --kill-after=15s "${limit}s" \
    srun -n "$HODLR_PROFILE_RANKS" --ntasks-per-node="$pernode" -c "${SLURM_CPUS_PER_TASK:-16}" \
    --gpu-bind=none --cpu-bind=cores --kill-on-bad-exit=1 --wait=0 \
    --output="$dir/panels.rank%t.log" --error="$dir/panels.rank%t.log" \
    bash test/run_hodlr_stage_rank.sh "$dir" panels 0 "${args[@]}" || status=$?
  elapsed=$(($(date +%s)-case_started));printf '%s\n' "$elapsed" > "$dir/elapsed_seconds.txt"
  (( index > 0 )) || first_elapsed=$elapsed
  printf '%s\n' "$status" > "$dir/panels.exit_code"
  check=0
  python3 test/analyze_hodlr_stages.py "$dir" --require-complete || check=$?
  python3 test/analyze_hodlr_blas_threads.py "$out" || check=$?
  (( status == 0 && check == 0 )) || exit 1
  index=$((index+1))
done < "$out/thread_order.txt"
python3 test/analyze_hodlr_blas_threads.py "$out" --require-complete
sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,MaxRSS,ExitCode -P > "$out/accounting.txt"
