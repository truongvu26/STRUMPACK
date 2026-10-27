#!/bin/bash
#SBATCH -A m4872_g
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 00:10:00
#SBATCH -J hodlr_solver_reuse
#SBATCH -o profile_results/reuse_%j.out
set -euo pipefail
module load PrgEnv-gnu cudatoolkit/13.2 craype-accel-nvidia80
module unload cray-libsci darshan
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPICH_GPU_SUPPORT_ENABLED=0 BPACK_GPU_AWARE_MPI=0 BPACK_CHECK=''
export BPACK_GPU_HEAP_FRACTION=0.85 HODLR_PROFILE_RANKS=4
cd "$SLURM_SUBMIT_DIR"
out="$PWD/profile_results/$SLURM_JOB_ID"; mkdir -p "$out"
cp "$HODLR_PROFILE_SNAPSHOT/manifest.txt" "$out/manifest.txt"
cp "${BASH_SOURCE[0]}" "$out/job.sbatch"
for spec in resource:4 square:4 rectangular:4 rectangular3:3; do
  variant=${spec%:*}; ranks=${spec#*:}
  case "$variant" in
    resource) binary="$HODLR_PROFILE_SNAPSHOT/resource_probe";;
    square) binary="$HODLR_PROFILE_SNAPSHOT/square_test";;
    *) binary="$HODLR_PROFILE_SNAPSHOT/rectangular_test";;
  esac
  timeout --kill-after=30s 240s srun -n "$ranks" -c8 --gpu-bind=none --cpu-bind=cores \
    --kill-on-bad-exit=1 --wait=30 --output="$out/$variant.rank%t.log" --error="$out/$variant.rank%t.log" \
    bash test/run_hodlr_stage_rank.sh "$out" "$variant" 0 "$binary"
  printf '0\n' > "$out/$variant.exit_code"
done
# The identical resource stress must reproduce the unpatched library's failure.
control_status=0
HODLR_PROFILE_SNAPSHOT="$PWD/profile_results/stage_snapshot" \
  timeout --kill-after=30s 120s srun -n2 -c8 --gpu-bind=none --cpu-bind=cores \
  --kill-on-bad-exit=1 --wait=30 --output="$out/control.rank%t.log" --error="$out/control.rank%t.log" \
  bash test/run_hodlr_stage_rank.sh "$out" control 0 "$HODLR_PROFILE_SNAPSHOT/resource_probe" || control_status=$?
printf '%s\n' "$control_status" > "$out/control.exit_code"
(( control_status != 0 )) || { echo 'Unpatched resource control unexpectedly passed'; exit 1; }
rg -q 'cusolverDnCreate failed with status 7' "$out/control.rank0.log" "$out/control.rank1.log"
timeout --kill-after=30s 180s srun -n4 -c8 --gpu-bind=none --cpu-bind=cores \
  --kill-on-bad-exit=1 --wait=30 --output="$out/poisson.rank%t.log" --error="$out/poisson.rank%t.log" \
  bash test/run_hodlr_stage_rank.sh "$out" poisson 0 "$HODLR_PROFILE_SNAPSHOT/baseline" 40 3 \
  --sp_compression HODLR --sp_compression_min_sep_size 1000 --sp_compression_min_front_size 1000 \
  --hodlr_leaf_size 128 --hodlr_rel_tol 1e-5 --hodlr_butterfly_levels 0 \
  --sp_Krylov_solver refinement --sp_rel_tol 1e-10 --hodlr_use_gpu 1 \
  --hodlr_gpu_rectangular 1 --hodlr_gpu_front_max_bytes 268435456
printf '0\n' > "$out/poisson.exit_code"
python3 - "$out" <<'PY'
import json,math,re,sys
from pathlib import Path
r=Path(sys.argv[1]); cases=[]
for variant,n in [('resource',4),('square',4),('rectangular',4),('rectangular3',3),('poisson',4)]:
 for rank in range(n):
  x=json.loads((r/f'{variant}.cusolver.rank{rank}.json').read_text())
  assert x==dict(handles_created=1,params_created=1,creation_failures=0),(variant,rank,x)
 cases.append(dict(case=variant,status='PASS',ranks=n,handles_per_process=1,params_per_process=1))
text=(r/'poisson.rank0.log').read_text()
error=float(re.search(r'# relative error = .* = (\S+)',text)[1])
residual=float(re.search(r'# COMPONENTWISE SCALED RESIDUAL = (\S+)',text)[1])
assert math.isfinite(error) and error<1e-8 and math.isfinite(residual) and residual<1e-6
record=dict(status='PASS',job=r.name,heap_fraction=.85,cases=cases,
            unpatched_control='EXPECTED_CUSOLVER_RESOURCE_FAILURE',relative_error=error,scaled_residual=residual)
(r/'resource_validation.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
PY
