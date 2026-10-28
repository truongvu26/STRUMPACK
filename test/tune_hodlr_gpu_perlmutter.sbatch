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
#SBATCH -J hodlr_gpu_tune
#SBATCH -o profile_results/gpu_tune_%j.out
# HODLR_TUNE_PLAN is a JSON candidate array; failed candidates do not end the search.
set -euo pipefail
# Execute an immutable job-local copy so edits cannot alter an active Bash reader.
if [[ ${HODLR_TUNE_FROZEN_RUNNER:-0} != 1 ]]; then
  frozen_label=${HODLR_TUNE_RUN_LABEL:-}
  [[ -z "$frozen_label" || "$frozen_label" =~ ^[A-Za-z0-9_-]+$ ]] || exit 1
  frozen_dir="${SLURM_SUBMIT_DIR:?}/profile_results/${SLURM_JOB_ID:?}${frozen_label:+-$frozen_label}"
  mkdir -p "$frozen_dir"
  frozen_runner="$frozen_dir/runner_frozen.sbatch"
  [[ "${BASH_SOURCE[0]}" == "$frozen_runner" ]] || cp "${BASH_SOURCE[0]}" "$frozen_runner"
  export HODLR_TUNE_FROZEN_RUNNER=1
  exec bash "$frozen_runner"
fi
module load PrgEnv-gnu cudatoolkit/13.2 craype-accel-nvidia80
module unload cray-libsci darshan
cd "$SLURM_SUBMIT_DIR"
export HODLR_PROFILE_SNAPSHOT="${HODLR_TUNE_SNAPSHOT:-$PWD/profile_results/guard32_snapshot_v1}"
export HODLR_BLAS_LIBRARY="${HODLR_TUNE_BLAS_LIBRARY:-$PWD/profile_results/openblas_pthreads_snapshot_v1/lib/libopenblas.so.0.3}"
[[ "$HODLR_BLAS_LIBRARY" != none ]] || unset HODLR_BLAS_LIBRARY
export HODLR_PROFILE_RANKS="${HODLR_TUNE_RANKS:-4}" HODLR_RANKS_PER_GPU="${HODLR_RANKS_PER_GPU:-1}"
default_workers=1;[[ -z ${HODLR_BLAS_LIBRARY:-} ]] || default_workers=8
export HODLR_TUNE_GRID="${HODLR_TUNE_GRID:-100}" OPENBLAS_NUM_THREADS="${HODLR_TUNE_BLAS_THREADS:-$default_workers}"
export OMP_NUM_THREADS=1 MPICH_GPU_SUPPORT_ENABLED=0 BPACK_GPU_AWARE_MPI=0 BPACK_CHECK=''
export BPACK_GPU_BACA_RESIDUAL_GUARD=0 BPACK_GPU_HODLR_SVD=qr
unset BPACK_GPU_BACA_RESIDUAL_LOG BPACK_GPU_BACA_RESIDUAL_TRACE STRUMPACK_HODLR_VERIFY_CONSTRUCTION CUDA_LAUNCH_BLOCKING STRUMPACK_HODLR_STAGE_FILE
case "$HODLR_RANKS_PER_GPU" in 1|2) ;; *) exit 1 ;; esac
[[ "$HODLR_PROFILE_RANKS" =~ ^[1-9][0-9]*$ && "$HODLR_TUNE_GRID" =~ ^[1-9][0-9]*$ ]] || exit 1
nodes=${HODLR_TUNE_NODES:-$SLURM_JOB_NUM_NODES}
[[ "$nodes" =~ ^[1-9][0-9]*$ ]] && (( nodes<=SLURM_JOB_NUM_NODES )) || exit 1
export HODLR_TUNE_NODES="$nodes"
mapfile -t allocated_nodes < <(scontrol show hostnames "${SLURM_JOB_NODELIST:?}")
(( ${#allocated_nodes[@]}>=nodes )) || exit 1
node_list=${allocated_nodes[0]}
for ((node_index=1;node_index<nodes;node_index++));do node_list+=",${allocated_nodes[$node_index]}";done
export HODLR_TUNE_NODELIST="$node_list"
pernode=$((HODLR_PROFILE_RANKS/nodes))
(( HODLR_PROFILE_RANKS%nodes==0 && pernode>0 && pernode%HODLR_RANKS_PER_GPU==0 )) || exit 1
cpus=${HODLR_TUNE_CPUS_PER_RANK:-${SLURM_CPUS_PER_TASK:-16}}
[[ "$cpus" =~ ^[1-9][0-9]*$ ]] || exit 1
export HODLR_TUNE_CPUS_PER_RANK="$cpus"
gpus=$((pernode/HODLR_RANKS_PER_GPU))
label=${HODLR_TUNE_RUN_LABEL:-}
[[ -z "$label" || "$label" =~ ^[A-Za-z0-9_-]+$ ]] || exit 1
started=$(date +%s);out="$PWD/profile_results/$SLURM_JOB_ID${label:+-$label}";mkdir -p "$out"
cp "${BASH_SOURCE[0]}" "$out/job.sbatch"
analyzer="$out/analyze_hodlr_gpu_tuning.py"
cp test/analyze_hodlr_gpu_tuning.py "$analyzer"
scontrol show job "$SLURM_JOB_ID" > "$out/slurm_job.txt"
plan=${HODLR_TUNE_PLAN:-$out/default_plan.json}
if [[ -z ${HODLR_TUNE_PLAN:-} ]];then
  printf '[{"label":"baseline"},{"label":"tensor","gpu_mode":2},{"label":"leaf512","leaf":512},{"label":"leaf1024","leaf":1024},{"label":"baseline_repeat"}]\n' > "$plan"
fi
cp "$plan" "$out/input_plan.json"
python3 "$analyzer" "$out" --prepare "$out/input_plan.json"
srun --nodelist="$node_list" -N "$nodes" -n "$nodes" --ntasks-per-node=1 -c1 --gpus-per-node="$gpus" --input=none --gpu-bind=none \
  bash -c 'hostname;nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv' > "$out/devices.txt"
while IFS= read -r label;do
  dir="$out/$label"
  remaining=$((${HODLR_TUNE_ALLOCATION_SECONDS:-1500}-$(date +%s)+started));budget=$((remaining-90))
  if (( budget < ${HODLR_TUNE_MIN_CASE_SECONDS:-60} ));then
    printf 'Insufficient allocation budget: %s seconds remain\n' "$remaining" > "$dir/skipped_budget.txt"
    continue
  fi
  read -r workers fraction case_limit residual_guard < <(python3 - "$dir/config.json" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]));print(c['blas_threads'],c['heap_fraction'],c['timeout_seconds'],c.get('baca_residual_guard',0))
PY
)
  export OPENBLAS_NUM_THREADS="$workers" HODLR_OPENBLAS_THREADS="$workers" BPACK_GPU_HEAP_FRACTION="$fraction"
  export BPACK_GPU_BACA_RESIDUAL_GUARD="$residual_guard"
  limit=$budget;(( case_limit>=limit )) || limit=$case_limit
  python3 - "$dir/argv.json" > "$dir/argv.nul" <<'PY'
import json,sys
sys.stdout.buffer.write(b'\0'.join(s.encode() for s in json.load(open(sys.argv[1])))+b'\0')
PY
  mapfile -d '' -t args < "$dir/argv.nul"
  printf '%s\n' "$HODLR_TUNE_GRID" "$HODLR_PROFILE_RANKS" > "$dir/layout.txt"
  printf 'panels\n' > "$dir/cases.txt";printf '%s\n' "$HODLR_RANKS_PER_GPU" > "$dir/rank_per_gpu.txt"
  printf '%s\n' "$fraction" > "$dir/heap_fraction.txt";cp "$out/devices.txt" "$dir/devices.txt"
  sha256sum "$HODLR_PROFILE_SNAPSHOT/panels" "$HODLR_PROFILE_SNAPSHOT/lib/libbutterflypack.so.5.0.0" > "$dir/binary_hashes.txt"
  [[ -z ${HODLR_BLAS_LIBRARY:-} ]] || sha256sum "$HODLR_BLAS_LIBRARY" >> "$dir/binary_hashes.txt"
  status=0
  /usr/bin/time -v -o "$dir/panels.time" timeout --kill-after=15s "${limit}s" \
    srun --nodelist="$node_list" -N "$nodes" -n "$HODLR_PROFILE_RANKS" --ntasks-per-node="$pernode" -c "$cpus" --gpus-per-node="$gpus" \
    --input=none --gpu-bind=none --cpu-bind=cores --kill-on-bad-exit=1 --wait=0 \
    --output="$dir/panels.rank%t.log" --error="$dir/panels.rank%t.log" \
    bash test/run_hodlr_stage_rank.sh "$dir" panels 0 "${args[@]}" || status=$?
  printf '%s\n' "$status" > "$dir/panels.exit_code"
  python3 "$analyzer" "$out"
done < "$out/case_order.txt"
python3 "$analyzer" "$out" --final
sacct -j "$SLURM_JOB_ID" --format=JobID,State,Elapsed,MaxRSS,ExitCode -P > "$out/accounting.txt"
