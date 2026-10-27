#!/bin/bash
#SBATCH -A m4872_g
#SBATCH -C gpu&hbm40g
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --gpus-per-node=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH -t 00:15:00
#SBATCH -J hodlr_host_panels
#SBATCH -o profile_results/panels_%j.out
set -euo pipefail
module load PrgEnv-gnu cudatoolkit/13.2 craype-accel-nvidia80
module unload cray-libsci darshan
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPICH_GPU_SUPPORT_ENABLED=0 BPACK_GPU_AWARE_MPI=0 BPACK_CHECK=''
export BPACK_GPU_HEAP_FRACTION=0.85 HODLR_PROFILE_RANKS=4
cd "$SLURM_SUBMIT_DIR"
out="$PWD/profile_results/$SLURM_JOB_ID"; mkdir -p "$out"
cp "$HODLR_PROFILE_SNAPSHOT/manifest.txt" "$out/manifest.txt"
cp "${BASH_SOURCE[0]}" "$out/job.sbatch"
for ranks in 1 3 4; do
  for name in square rectangular; do
    timeout --kill-after=30s 180s srun -n "$ranks" -c8 --gpu-bind=none --cpu-bind=cores \
      --kill-on-bad-exit=1 --wait=30 --output="$out/${name}_${ranks}.rank%t.log" --error="$out/${name}_${ranks}.rank%t.log" \
      bash test/run_hodlr_stage_rank.sh "$out" "${name}_${ranks}" 0 "$HODLR_PROFILE_SNAPSHOT/${name}_test" \
      --hodlr_gpu_host_panels 1 --hodlr_gpu_panel_max_bytes 1024
    printf '0\n' > "$out/${name}_${ranks}.exit_code"
  done
done
for variant in baseline panels panels_baca; do
  mode=1; cap=0; extra=()
  if [[ "$variant" == baseline ]]; then mode=0; cap=268435456; fi
  [[ "$variant" == panels_baca ]] && extra=(--hodlr_lr_leaf 4)
  timeout --kill-after=30s 240s srun -n4 -c8 --gpu-bind=none --cpu-bind=cores \
    --kill-on-bad-exit=1 --wait=30 --output="$out/$variant.rank%t.log" --error="$out/$variant.rank%t.log" \
    bash test/run_hodlr_stage_rank.sh "$out" "$variant" 0 "$HODLR_PROFILE_SNAPSHOT/poisson" 40 3 \
    --sp_compression HODLR --sp_compression_min_sep_size 1000 --sp_compression_min_front_size 1000 \
    --hodlr_leaf_size 128 --hodlr_rel_tol 1e-5 --hodlr_butterfly_levels 0 \
    --sp_Krylov_solver refinement --sp_rel_tol 1e-10 --hodlr_use_gpu 1 --hodlr_gpu_rectangular 1 \
    --hodlr_gpu_front_max_bytes "$cap" --hodlr_gpu_host_panels "$mode" --hodlr_gpu_panel_max_bytes 1048576 "${extra[@]}"
  printf '0\n' > "$out/$variant.exit_code"
done
python3 - "$out" <<'PY'
import json,math,re,sys
from pathlib import Path
r=Path(sys.argv[1]);cases=[]
for name in ['baseline','panels','panels_baca']:
 text=(r/(name+'.rank0.log')).read_text()
 error=float(re.search(r'# relative error = .* = (\S+)',text)[1]);residual=float(re.search(r'# COMPONENTWISE SCALED RESIDUAL = (\S+)',text)[1])
 assert math.isfinite(error) and error<1e-8 and math.isfinite(residual) and residual<1e-6
 if name!='baseline':
  alltext='\n'.join((r/(name+'.rank'+str(i)+'.log')).read_text() for i in range(4))
  peaks=list(map(int,re.findall('staging_peak_bytes_local=(\d+)',alltext)));assert peaks and max(peaks)<=1048576
  assert 'kind=HODLR' in alltext and 'kind=BF' in alltext
 cases.append(dict(case=name,relative_error=error,scaled_residual=residual,status='PASS'))
record=dict(status='PASS',job=r.name,contract_tests='real/complex N/T/C, moves, child-inclusive references, forced multi-window budgets, one/three RHS, 1/3/4 ranks and strict subcommunicators',sparse_cases=cases)
(r/'host_panel_validation.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
PY
