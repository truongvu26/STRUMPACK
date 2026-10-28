#!/usr/bin/env python3
"""Validate paired native-library trials before comparing median sparse timings."""
import argparse,json,math,re,statistics
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('directory',type=Path)
p.add_argument('--require-complete',action='store_true')
a=p.parse_args(); root=a.directory
manifest=json.loads((root/'run_manifest.json').read_text())
order=[line.split() for line in (root/'case_order.txt').read_text().splitlines() if line.strip()]
result=dict(status='RUNNING',job=root.name,manifest=manifest,cases=[],variants={},notes=[
    'Before/after/after/before quiet sparse runs use the same executable and topology; only the native ButterflyPACK library changes.',
    'Two runs per variant are preliminary timings, not a statistical performance guarantee.',
    'Randomized compression may vary ranks and refinement iterations. Every case must pass before a speed ratio is reported.',
    'Each MPI rank uses one GPU. The active GPU count is ranks divided by ranks_per_gpu, which can differ from the allocation size.'])
failures=[]
try:
    variants=manifest['variants']
    if set(variants)!={'before','after'}: raise ValueError('Expected before and after snapshots')
    if variants['before']['executable_sha256']!=variants['after']['executable_sha256']:
        raise ValueError('Before/after executables differ')
    if variants['before']['library_sha256']==variants['after']['library_sha256']:
        raise ValueError('Before/after libraries are identical')
    if order!=[['00-before','before'],['01-after','after'],['02-after','after'],['03-before','before']]:
        raise ValueError('Expected before/after/after/before order')
    ranks=manifest['ranks']; sharing=manifest['ranks_per_gpu']
    if sharing not in (1,2) or ranks<=0 or ranks%sharing:
        raise ValueError('Invalid rank/GPU topology')
    result['active_gpu_count']=ranks//sharing
    first_bindings=None
    for label,variant in order:
        directory=root/label; source=directory/'stage_summary.json'
        if not source.exists(): continue
        d=json.loads(source.read_text())
        if d['status']!='PASS': raise ValueError(label+': stage validation did not pass')
        if (d['grid']!=manifest['grid'] or d['ranks']!=ranks or d['rhs']!=manifest['rhs']
            or d.get('ranks_per_gpu',2)!=sharing):
            raise ValueError(label+': topology or problem differs')
        rows=(directory/'binary_hashes.txt').read_text().splitlines()
        if len(rows)!=2: raise ValueError(label+': missing binary hashes')
        hashes=[line.split()[0] for line in rows]
        expected=variants[variant]
        if hashes!=[expected['executable_sha256'],expected['library_sha256']]:
            raise ValueError(label+': snapshot binary hashes changed')
        if len(d['cases'])!=1 or d['cases'][0]['variant']!='panels':
            raise ValueError(label+': expected exactly one quiet panels case')
        c=d['cases'][0]
        if c['status']!='PASS': raise ValueError(label+': solver case failed')
        bindings=c['gpu_uuid_rank_counts']
        if (sum(bindings.values())!=ranks or len(bindings)!=ranks//sharing
            or set(bindings.values())!={sharing}):
            raise ValueError(label+': actual GPU binding differs from requested topology')
        if first_bindings is None: first_bindings=bindings
        elif bindings!=first_bindings: raise ValueError(label+': GPU placement changed across paired cases')
        for key in ('factor_seconds','solve_seconds','factor_plus_solve_seconds','relative_error','scaled_residual'):
            if not math.isfinite(c[key]) or c[key]<0: raise ValueError(label+': invalid '+key)
        if c['relative_error']>=1e-8 or c['scaled_residual']>=1e-6:
            raise ValueError(label+': numerical accuracy failed')
        if abs(c['factor_seconds']+c['solve_seconds']-c['factor_plus_solve_seconds'])>1e-6:
            raise ValueError(label+': inconsistent total time')
        record=dict(c); record.update(label=label,variant=variant)
        matches=re.findall(r'^#\s+- maximum HODLR rank = (.*)$',(directory/'panels.rank0.log').read_text(),re.M)
        if len(matches)>1 or (matches and int(matches[0])<=0):
            raise ValueError(label+': maximum HODLR rank must be unique and positive')
        if matches: record['maximum_hodlr_rank']=int(matches[0])
        result['cases'].append(record)
    if len(result['cases'])==len(order):
        for variant in ('before','after'):
            cases=[c for c in result['cases'] if c['variant']==variant]
            result['variants'][variant]=dict(trials=len(cases),**{
                'median_'+key:statistics.median(c[key] for c in cases)
                for key in ('factor_seconds','solve_seconds','factor_plus_solve_seconds')})
        before=result['variants']['before']; after=result['variants']['after']
        if min(after['median_factor_seconds'],after['median_factor_plus_solve_seconds'])<=0:
            raise ValueError('After median timing must be positive')
        result['factor_speedup']=before['median_factor_seconds']/after['median_factor_seconds']
        result['factor_plus_solve_speedup']=before['median_factor_plus_solve_seconds']/after['median_factor_plus_solve_seconds']
        result['status']='PASS'
except (ValueError,KeyError,OSError,TypeError) as e:
    failures.append(str(e));result['status']='FAIL';result['variants']={}
    result.pop('factor_speedup',None);result.pop('factor_plus_solve_speedup',None)
result['failures']=failures
(root/'device_wy_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['status'],[(c['label'],c['factor_seconds'],c['solve_seconds']) for c in result['cases']],
      'factor speedup',result.get('factor_speedup'),'total speedup',result.get('factor_plus_solve_speedup'),flush=True)
if failures or (a.require_complete and result['status']!='PASS'): raise SystemExit(1)
