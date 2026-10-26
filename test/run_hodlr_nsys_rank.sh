#!/bin/bash
# Nsight report import needs local storage; avoid MPI interposition on Cray MPICH.
set -euo pipefail
out=$1
shift
local_dir=$(mktemp -d "${TMPDIR:-/tmp}/hodlr-nsys-${SLURM_JOB_ID}-${SLURM_PROCID}.XXXXXX")
trap 'rm -rf "$local_dir"' EXIT
status=0
nsys profile --trace=cuda,nvtx,cublas --sample=none --cpuctxsw=none \
  --force-overwrite=false --output="$local_dir/gpu.rank${SLURM_PROCID}" "$@" || status=$?
shopt -s nullglob
reports=("$local_dir"/gpu.rank*)
if (( ${#reports[@]} )); then cp "${reports[@]}" "$out/"; fi
exit "$status"
