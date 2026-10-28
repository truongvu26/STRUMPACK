#!/bin/bash
# One or two MPI ranks per GPU; optionally trace selected ranks.
set -euo pipefail
out=$1; variant=$2; trace=$3; shift 3
IFS=, read -r -a visible <<< "${CUDA_VISIBLE_DEVICES:-${SLURM_JOB_GPUS:?}}"
ranks_per_gpu=${HODLR_RANKS_PER_GPU:-2}
case "$ranks_per_gpu" in
  1|2) ;;
  *) echo 'HODLR_RANKS_PER_GPU must be 1 or 2' >&2; exit 1 ;;
esac
index=$((SLURM_LOCALID / ranks_per_gpu))
(( index < ${#visible[@]} )) || { echo 'MPI local rank exceeds the visible GPU layout' >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="${visible[$index]}"
# The executable has DT_RPATH; preload the immutable SONAME to override it.
export LD_PRELOAD="$HODLR_PROFILE_SNAPSHOT/lib/libbutterflypack.so.5.0.0${LD_PRELOAD:+:$LD_PRELOAD}"
uuid=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=uuid --format=csv,noheader)
printf '# stage binding rank=%s local_rank=%s node=%s uuid=%s\n' "$SLURM_PROCID" "$SLURM_LOCALID" "$(hostname)" "$uuid"
if [[ -n "${HODLR_CUSOLVER_AUDIT_LIBRARY:-}" ]]; then
  export HODLR_CUSOLVER_AUDIT="$out/$variant.cusolver"
  export LD_PRELOAD="$LD_PRELOAD:$HODLR_CUSOLVER_AUDIT_LIBRARY"
fi
if [[ "$variant" == diagnostic || "$variant" == traced ]]; then
  export STRUMPACK_HODLR_STAGE_FILE="$out/$variant.stages"
fi
trace_ranks=${HODLR_NSYS_TRACE_RANKS:-0,$((HODLR_PROFILE_RANKS-1))}
if [[ "$trace" != 1 || ",$trace_ranks," != *",$SLURM_PROCID,"* ]]; then
  exec "$@"
fi
local_dir=$(mktemp -d "${TMPDIR:-/tmp}/hodlr-stage-${SLURM_JOB_ID}-${SLURM_PROCID}.XXXXXX")
trap 'rm -rf "$local_dir"' EXIT
status=0
sampling=(--sample=none --cpuctxsw=none)
if [[ ${HODLR_NSYS_CPU_SAMPLING:-0} == 1 ]]; then
  nsys status -e > "$out/rank${SLURM_PROCID}.nsys_environment.txt" 2>&1
  sampling=(--sample=process-tree --backtrace=dwarf --sampling-period=8000000
            --samples-per-backtrace=1 --cpuctxsw=process-tree --resolve-symbols=true)
fi
nsys profile --trace=cuda,nvtx,cublas,osrt --osrt-threshold=100000 "${sampling[@]}" \
  --force-overwrite=false --output="$local_dir/rank${SLURM_PROCID}" "$@" || status=$?
printf '%s\n' "$status" > "$out/rank${SLURM_PROCID}.target_exit_code"
shopt -s nullglob
reports=("$local_dir"/*.nsys-rep)
if (( ${#reports[@]} )); then
  cp "${reports[@]}" "$out/"
fi
exit "$status"
