#!/usr/bin/env python3
"""Compare validated sparse trials with fixed binaries and varying pthread BLAS workers."""
import argparse,json,math,re,statistics
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--require-complete',action='store_true')
a=p.parse_args();root=a.directory
manifest=json.loads((root/'run_manifest.json').read_text())
order=[(row.split()[0],int(row.split()[1])) for row in (root/'thread_order.txt').read_text().splitlines() if row.strip()]
result=dict(status='RUNNING',job=root.name,manifest=manifest,cases=[],threads={},skipped_optional_cases=[],notes=[
    'Same quiet executable, native ButterflyPACK library, pthread BLAS library, topology and solver options; only BLAS worker count changes.',
    'OMP remains1. Runtime/affinity reports are probe subprocesses immediately before solver exec, not measurements inside the solver.',
    'One trial for each non-baseline thread count is preliminary; random compression can vary ranks and refinement iterations.',
    'The optional final one-thread repeat may be skipped for budget; no primary case may be skipped.'])
failures=[]
try:
    counts=[n for _,n in order]
    if counts not in ([1,4,8,1],[1,4,8,16,1]):raise ValueError('Expected1/4/8/[16]/1 order')
    if len({label for label,_ in order})!=len(order):raise ValueError('Duplicate case directory')
    ranks=manifest['ranks'];sharing=manifest['ranks_per_gpu']
    if sharing not in (1,2) or ranks<1 or ranks%sharing:raise ValueError('Invalid MPI/GPU layout')
    bindings0=None;affinity0=None;configs=set()
    for index,(label,workers) in enumerate(order):
        directory=root/label
        if (directory/'skipped_budget.txt').exists():
            if index!=len(order)-1 or workers!=1:raise ValueError('A primary case was skipped')
            if (directory/'panels.exit_code').exists() or (directory/'stage_summary.json').exists():
                raise ValueError('Attempted repeat cannot be relabeled as skipped')
            result['skipped_optional_cases'].append(label);continue
        source=directory/'stage_summary.json'
        if not source.exists():continue
        d=json.loads(source.read_text())
        if d['status']!='PASS' or d['grid']!=manifest['grid'] or d['ranks']!=ranks or d['rhs']!=manifest['rhs'] or d.get('ranks_per_gpu',2)!=sharing:
            raise ValueError(label+': stage validation/topology failed')
        hashes=[line.split()[0] for line in (directory/'binary_hashes.txt').read_text().splitlines()]
        if hashes!=[manifest['executable_sha256'],manifest['native_library_sha256'],manifest['blas_library_sha256']]:
            raise ValueError(label+': executable/native/BLAS hashes differ')
        if len(d['cases'])!=1 or d['cases'][0]['variant']!='panels':raise ValueError(label+': expected quiet panels case')
        c=d['cases'][0]
        if c['status']!='PASS':raise ValueError(label+': numerical case failed')
        for key in ('factor_seconds','solve_seconds','factor_plus_solve_seconds','relative_error','scaled_residual'):
            if not math.isfinite(c[key]) or c[key]<0:raise ValueError(label+': invalid '+key)
        if c['factor_seconds']<=0 or c['relative_error']>=1e-8 or c['scaled_residual']>=1e-6:raise ValueError(label+': numerical accuracy or timing failed')
        if abs(c['factor_seconds']+c['solve_seconds']-c['factor_plus_solve_seconds'])>1e-6:raise ValueError(label+': inconsistent total')
        bindings=c['gpu_uuid_rank_counts']
        if sum(bindings.values())!=ranks or len(bindings)!=ranks//sharing or set(bindings.values())!={sharing}:
            raise ValueError(label+': GPU sharing differs')
        if bindings0 is None:bindings0=bindings
        elif bindings!=bindings0:raise ValueError(label+': GPU placement changed')
        probes=[];affinity={};cores_by_node={}
        for rank in range(ranks):
            probe=json.loads((directory/('panels.blas.rank%d.json'%rank)).read_text())
            if (probe['status']!='PASS' or probe['rank']!=rank or probe['library_sha256']!=manifest['blas_library_sha256']
                or Path(probe['library']).resolve()!=Path(manifest['blas_library']).resolve()
                or probe['reported_threads']!=workers or probe['requested_threads']!=workers
                or probe['parallel_kind']!=1 or probe['omp_max_threads']!=1 or len(probe['loader_openblas_dependencies'])!=1):
                raise ValueError(label+': BLAS runtime verification failed on rank'+str(rank))
            if len(probe['affinity_cpus'])<workers or probe['physical_core_count'] is None or probe['physical_core_count']<workers:
                raise ValueError(label+': insufficient verified physical CPU cores on rank'+str(rank))
            cores={tuple(pair) for pair in probe['physical_core_ids']}
            used=cores_by_node.setdefault(probe['node'],set())
            if len(cores)!=probe['physical_core_count'] or used.intersection(cores):
                raise ValueError(label+': rank physical CPU placements overlap or disagree')
            used.update(cores)
            affinity[rank]=(probe['node'],probe['affinity_cpus']);configs.add(probe['config']);probes.append(probe)
        if affinity0 is None:affinity0=affinity
        elif affinity!=affinity0:raise ValueError(label+': rank CPU affinity changed')
        record=dict(c);record.update(label=label,blas_threads=workers,runtime=probes)
        matches=re.findall(r'^#\s+- maximum HODLR rank = (.*)$',(directory/'panels.rank0.log').read_text(),re.M)
        if len(matches)>1 or (matches and int(matches[0])<=0):raise ValueError(label+': maximum HODLR rank is not unique/positive')
        if matches:record['maximum_hodlr_rank']=int(matches[0])
        result['cases'].append(record)
    if len(configs)>1:raise ValueError('BLAS runtime version/configuration changed')
    if len(result['cases'])+len(result['skipped_optional_cases'])==len(order):
        for workers in sorted({c['blas_threads'] for c in result['cases']}):
            cases=[c for c in result['cases'] if c['blas_threads']==workers]
            result['threads'][str(workers)]=dict(trials=len(cases),**{
                'median_'+key:statistics.median(c[key] for c in cases)
                for key in ('factor_seconds','solve_seconds','factor_plus_solve_seconds')})
        baseline=result['threads']['1']
        for rec in result['threads'].values():
            rec['factor_speedup_vs_one_thread']=baseline['median_factor_seconds']/rec['median_factor_seconds']
            rec['total_speedup_vs_one_thread']=baseline['median_factor_plus_solve_seconds']/rec['median_factor_plus_solve_seconds']
        result['status']='PASS'
except (ValueError,KeyError,OSError,TypeError,IndexError) as e:
    failures.append(str(e));result['status']='FAIL';result['threads']={}
result['failures']=failures
(root/'blas_threads_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['status'],[(c['label'],c['factor_seconds'],c['solve_seconds']) for c in result['cases']],result['threads'],flush=True)
if failures or (a.require_complete and result['status']!='PASS'):raise SystemExit(1)
