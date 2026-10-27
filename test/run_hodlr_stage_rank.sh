#!/bin/bash
# Two MPI ranks per GPU; optionally trace only rank zero and the final rank.
set -euo pipefail
out=$1; variant=$2; trace=$3; shift 3
IFS=, read -r -a visible <<< "${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:?}}"
index=$((SLURM_LOCALID / 2))
(( index < ${#visible[@]} )) || exit 1
export CUDA_VISIBLE_DEVICES="${visible[$index]}"
# The executable has DT_RPATH; preload the immutable SONAME to override it.
export LD_PRELOAD="$HODLR_PROFILE_SNAPSHOT/lib/libbutterflypack.so.5.0.0${LD_PRELOAD:+:$LD_PRELOAD}"
uuid=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid --format=csv,noheader)
printf '# stage binding rank=%s local_rank=%s node=%s uuid=%s\n' "$SLURM_PROCID" "$SLURM_LOCALID" "$(hostname)" "$uuid"
if [[ "$variant" == diagnostic || "$variant" == traced ]]; then
  export STRUMPACK_HODLR_STAGE_FILE="$out/$variant.stages"
fi
if [[ "$trace" != 1 ]] || (( SLURM_PROCID != 0 && SLURM_PROCID != HODLR_PROFILE_RANKS-1 )); then
  exec "$@"
fi
local_dir=$(mktemp -d "${TMPDIR:-/tmp}/hodlr-stage-${SLURM_JOB_ID}-${SLURM_PROCID}.XXXXXX")
trap 'rm -rf "$local_dir"' EXIT
status=0
nsys profile --trace=cuda,nvtx,cublas --sample=none --cpuctxsw=none \
  --force-overwrite=false --output="$local_dir/rank${SLURM_PROCID}" "$@" || status=$?
shopt -s nullglob
reports=("$local_dir"/*.nsys-rep)
if (( ${#reports[@]} )); then
  cp "${reports[@]}" "$out/"
  nsys export --type=sqlite --output="$local_dir/rank${SLURM_PROCID}.sqlite" "${reports[0]}"
  python3 "$SLURM_SUBMIT_DIR/test/summarize_hodlr_stage_cuda.py" "$local_dir/rank${SLURM_PROCID}.sqlite" "$out/rank${SLURM_PROCID}.cuda.json"
fi
exit "$status"
