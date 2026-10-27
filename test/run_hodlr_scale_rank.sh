#!/bin/bash
# Stable two-rank GPU sharing on every node, including multi-node allocations.
set -euo pipefail
IFS=, read -r -a visible <<< "${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:?}}"
index=$((SLURM_LOCALID / 2))
if (( index >= ${#visible[@]} )); then
  echo 'Two ranks per GPU requires enough visible allocated devices on each node.' >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${visible[$index]}"
uuid=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid --format=csv,noheader)
printf '# scale binding rank=%s local_rank=%s node=%s uuid=%s\n' \
  "$SLURM_PROCID" "$SLURM_LOCALID" "$(hostname)" "$uuid"
# A study owns an immutable copy so unrelated builds cannot change later trials.
snapshot="${SLURM_SUBMIT_DIR}/profile_results/${SLURM_JOB_ID}/solver_snapshot"
if [[ -x "$snapshot" ]]; then
  shift
  exec "$snapshot" "$@"
fi
exec "$@"
