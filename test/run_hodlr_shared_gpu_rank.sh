#!/bin/bash
# Fixed four-rank solver group: ranks 0/1 share the first GPU, 2/3 the second.
set -euo pipefail
IFS=, read -r -a visible <<< "${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:?}}"
index=$((SLURM_LOCALID / 2))
if (( ${#visible[@]} < 2 || index >= ${#visible[@]} )); then
  echo 'Shared-GPU test needs at least two visible allocated GPUs.' >&2
  exit 1
fi
export CUDA_VISIBLE_DEVICES="${visible[$index]}"
uuid=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid --format=csv,noheader)
printf '# shared-GPU binding rank=%s local_rank=%s device=%s uuid=%s\n' \
  "$SLURM_PROCID" "$SLURM_LOCALID" "$CUDA_VISIBLE_DEVICES" "$uuid"
exec "$@"
