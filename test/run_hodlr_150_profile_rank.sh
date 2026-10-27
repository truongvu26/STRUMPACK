#!/bin/bash
set -euo pipefail
out=$1; label=$2; shift 2
IFS=, read -r -a visible <<< "${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:?}}"
index=$((SLURM_LOCALID/2))
(( index < ${#visible[@]} )) || { echo 'Insufficient GPUs for two-rank sharing' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="${visible[$index]}"
uuid=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid --format=csv,noheader)
printf '# profile binding rank=%s local_rank=%s node=%s uuid=%s\n' "$SLURM_PROCID" "$SLURM_LOCALID" "$(hostname)" "$uuid"
if [[ "$label" == gpu_diagnostic || "$label" == gpu_trace ]]; then
  export STRUMPACK_HODLR_STAGE_FILE="$out/$label.stages"
else
  unset STRUMPACK_HODLR_STAGE_FILE
fi
if [[ "$label" != gpu_trace || ( "$SLURM_PROCID" != 0 && "$SLURM_PROCID" != "$((SLURM_NTASKS-1))" ) ]]; then
  exec "$@"
fi
local_dir=$(mktemp -d "${TMPDIR:-/tmp}/hodlr150-${SLURM_JOB_ID}-${SLURM_PROCID}.XXXXXX")
trap 'rm -rf "$local_dir"' EXIT
status=0
nsys profile --trace=cuda,nvtx,cublas --sample=none --cpuctxsw=none --force-overwrite=false \
  --output="$local_dir/rank${SLURM_PROCID}" "$@" || status=$?
shopt -s nullglob
reports=("$local_dir/"*.nsys-rep)
if (( ${#reports[@]} )); then
  cp "${reports[@]}" "$out/"
  if (( status==0 )); then
    nsys export --type=sqlite --output="$local_dir/rank${SLURM_PROCID}.sqlite" "${reports[0]}"
    LD_PRELOAD= python3 test/summarize_hodlr_150_nsys.py \
      "$local_dir/rank${SLURM_PROCID}.sqlite" "$out/cuda.rank${SLURM_PROCID}.json"
  fi
fi
exit "$status"
