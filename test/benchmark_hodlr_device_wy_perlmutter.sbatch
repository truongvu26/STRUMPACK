#!/bin/bash
#SBATCH -A m4872_g
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -N 4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH -t 00:30:00
#SBATCH -J hodlr_device_wy
#SBATCH -o profile_results/device_wy_%j.out
# Paired native-library comparison; both snapshots must contain the same panels executable.
set -euo pipefail
module load PrgEnv-gnu cudatoolkit/13.2 craype-accel-nvidia80
module unload cray-libsci darshan
: "${HODLR_WY_BEFORE:?Set the immutable baseline snapshot directory}"
: "${HODLR_WY_AFTER:?Set the immutable optimized snapshot directory}"
HODLR_WY_BEFORE=$(realpath "$HODLR_WY_BEFORE")
HODLR_WY_AFTER=$(realpath "$HODLR_WY_AFTER")
export HODLR_WY_BEFORE HODLR_WY_AFTER
export HODLR_PROFILE_RANKS="${HODLR_WY_RANKS:-32}" HODLR_RANKS_PER_GPU="${HODLR_RANKS_PER_GPU:-2}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPICH_GPU_SUPPORT_ENABLED=0 BPACK_GPU_AWARE_MPI=0 BPACK_CHECK=''
export BPACK_GPU_HEAP_FRACTION=0.85 BPACK_GPU_BACA_RESIDUAL_GUARD=0 BPACK_GPU_HODLR_SVD=qr
unset BPACK_GPU_BACA_RESIDUAL_LOG BPACK_GPU_BACA_RESIDUAL_TRACE STRUMPACK_HODLR_VERIFY_CONSTRUCTION CUDA_LAUNCH_BLOCKING
export HODLR_WY_GRID="${HODLR_WY_GRID:-150}"
case "$HODLR_RANKS_PER_GPU" in
  1|2) ;;
  *) echo 'HODLR_RANKS_PER_GPU must be 1 or 2' >&2; exit 1 ;;
esac
[[ "$HODLR_PROFILE_RANKS" =~ ^[1-9][0-9]*$ && "$HODLR_WY_GRID" =~ ^[1-9][0-9]*$ ]] || exit 1
pernode=$((HODLR_PROFILE_RANKS/SLURM_JOB_NUM_NODES))
(( HODLR_PROFILE_RANKS % SLURM_JOB_NUM_NODES == 0 && pernode > 0 && pernode % HODLR_RANKS_PER_GPU == 0 )) || {
  echo 'MPI ranks must divide evenly over nodes and complete each GPU sharing group' >&2; exit 1;
}
cd "$SLURM_SUBMIT_DIR"
started=$(date +%s)
out="$PWD/profile_results/$SLURM_JOB_ID"; mkdir -p "$out"
cp "${BASH_SOURCE[0]}" "$out/job.sbatch"
scontrol show job "$SLURM_JOB_ID" > "$out/slurm_job.txt"
printf '00-before before\n01-after after\n02-after after\n03-before before\n' > "$out/case_order.txt"
python3 - "$out/run_manifest.json" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
    return h.hexdigest()
variants={}
for name in ('before','after'):
    p=Path(os.environ['HODLR_WY_'+name.upper()]).resolve()
    variants[name]=dict(snapshot=str(p),executable_sha256=sha(p/'panels'),
        library_sha256=sha(p/'lib/libbutterflypack.so.5.0.0'),
        snapshot_manifest=(p/'manifest.txt').read_text() if (p/'manifest.txt').exists() else '')
assert variants['before']['executable_sha256']==variants['after']['executable_sha256'], 'Executables differ'
assert variants['before']['library_sha256']!=variants['after']['library_sha256'], 'Native libraries must differ'
d=dict(variants=variants,grid=int(os.environ['HODLR_WY_GRID']),ranks=int(os.environ['HODLR_PROFILE_RANKS']),
    nodes=int(os.environ['SLURM_JOB_NUM_NODES']),ranks_per_gpu=int(os.environ['HODLR_RANKS_PER_GPU']),
    rhs=3,omp_threads=1,openblas_threads=1,leaf_size=128,compression_tolerance=1e-5,
    refinement_tolerance=1e-10,butterfly_levels=0,residual_certification=False,
    svd='qr',gpu_heap_fraction=.85,host_panel_budget_bytes=67108864,front_image_cap_bytes=268435456)
Path(sys.argv[1]).write_text(json.dumps(d,indent=2)+'\n')
PY
srun -n "$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 -c1 --gpu-bind=none bash -c \
  'hostname; nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv' > "$out/devices.txt"
while read -r label variant; do
  dir="$out/$label"; mkdir -p "$dir"
  if [[ "$variant" == before ]]; then export HODLR_PROFILE_SNAPSHOT="$HODLR_WY_BEFORE";
  else export HODLR_PROFILE_SNAPSHOT="$HODLR_WY_AFTER"; fi
  printf '%s\n' "$HODLR_WY_GRID" "$HODLR_PROFILE_RANKS" > "$dir/layout.txt"
  printf 'panels\n' > "$dir/cases.txt"; printf '0.85\n' > "$dir/heap_fraction.txt"
  printf '%s\n' "$HODLR_RANKS_PER_GPU" > "$dir/rank_per_gpu.txt"
  cp "$out/devices.txt" "$dir/devices.txt"
  sha256sum "$HODLR_PROFILE_SNAPSHOT/panels" "$HODLR_PROFILE_SNAPSHOT/lib/libbutterflypack.so.5.0.0" > "$dir/binary_hashes.txt"
  args=("$HODLR_PROFILE_SNAPSHOT/panels" "$HODLR_WY_GRID" 3 --sp_compression HODLR
    --sp_compression_min_sep_size 1000 --sp_compression_min_front_size 1000
    --hodlr_leaf_size 128 --hodlr_rel_tol 1e-5 --hodlr_butterfly_levels 0
    --sp_Krylov_solver refinement --sp_rel_tol 1e-10 --sp_maxit 30
    --hodlr_use_gpu 1 --hodlr_gpu_rectangular 1 --hodlr_gpu_front_max_bytes 268435456
    --hodlr_gpu_host_panels 1 --hodlr_gpu_panel_max_bytes 67108864)
  printf '%q ' "${args[@]}" > "$dir/panels.command"; printf '\n' >> "$dir/panels.command"
  remaining=$((${HODLR_WY_ALLOCATION_SECONDS:-1800}-$(date +%s)+started)); limit=$((remaining-90))
  (( limit > 0 )) || { echo 'No time remains for the next case'; exit 1; }
  status=0
  /usr/bin/time -v -o "$dir/panels.time" timeout --kill-after=15s "${limit}s" \
    srun -n "$HODLR_PROFILE_RANKS" --ntasks-per-node="$pernode" -c "${SLURM_CPUS_PER_TASK:-8}" \
    --input=none --gpu-bind=none --cpu-bind=cores --kill-on-bad-exit=1 --wait=0 \
    --output="$dir/panels.rank%t.log" --error="$dir/panels.rank%t.log" \
    bash test/run_hodlr_stage_rank.sh "$dir" panels 0 "${args[@]}" || status=$?
  printf '%s\n' "$status" > "$dir/panels.exit_code"
  check=0
  python3 test/analyze_hodlr_stages.py "$dir" --require-complete || check=$?
  python3 test/analyze_hodlr_device_wy.py "$out" || check=$?
  (( status == 0 && check == 0 )) || exit 1
done < "$out/case_order.txt"
python3 test/analyze_hodlr_device_wy.py "$out" --require-complete
sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,MaxRSS,ExitCode -P > "$out/accounting.txt"
