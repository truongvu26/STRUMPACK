#!/bin/bash
# CPU-only report conversion after all ranks have left the MPI solver step.
# Keeping this outside srun prevents --wait from killing a larger rank report.
set -euo pipefail
out=$1; ranks=$2
local_dir=$(mktemp -d "${TMPDIR:-/tmp}/hodlr-stage-export.XXXXXX")
trap 'rm -rf "$local_dir"' EXIT
trace_ranks="0,$((ranks-1))"
[[ -f "$out/traced_ranks.txt" ]] && trace_ranks=$(cat "$out/traced_ranks.txt")
IFS=, read -r -a selected <<< "$trace_ranks"
for rank in "${selected[@]}"; do
  nsys export --type=sqlite --output="$local_dir/rank$rank.sqlite" \
    "$out/rank$rank.nsys-rep" > "$out/rank$rank.export.log" 2>&1
  python3 "$(dirname "$0")/summarize_hodlr_stage_cuda.py" \
    "$local_dir/rank$rank.sqlite" "$out/rank$rank.cuda.json"
  if [[ ${HODLR_NSYS_CPU_SAMPLING:-0} == 1 ]]; then
    python3 "$(dirname "$0")/summarize_hodlr_stage_cpu.py" \
      "$local_dir/rank$rank.sqlite" "$out/rank$rank.cpu.json"
  fi
  rm "$local_dir/rank$rank.sqlite"
done
