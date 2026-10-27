#!/usr/bin/env python3
"""Incremental correctness and per-rank phase accounting for the 150-cubed study."""
import argparse
from collections import Counter,defaultdict
import csv
import json
import math
from pathlib import Path
import re
import statistics
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--require-complete',action='store_true');a=p.parse_args();root=a.root
plan=json.loads((root/'plan.json').read_text());cases=[];failures=[]
def number(text,pat,kind=float):
    v=re.findall(pat,text)
    if len(v)!=1: raise ValueError('Expected one '+pat)
    n=kind(v[0].replace(',',''))
    if not math.isfinite(n): raise ValueError('Nonfinite '+pat)
    return n
for label in plan['variants']:
    status=root/(label+'.exit_code')
    if not status.exists(): continue
    c=dict(label=label,status='FAIL',exit_code=int(status.read_text()))
    try:
        if c['exit_code']: raise ValueError('Step exited '+str(c['exit_code']))
        logs=[(root/f'{label}.rank{r}.log').read_text() for r in range(plan['ranks'])];text=logs[0];alltext='\n'.join(logs)
        c['relative_error']=number(text,r'# relative error = .* = (\S+)');c['scaled_residual']=number(text,r'# COMPONENTWISE SCALED RESIDUAL = (\S+)')
        c['hodlr_fronts']=number(text,r'nr of HODLR Frontal matrices = ([\d,]+)',int)
        c['refinement_iterations']=number(text,r'number of Krylov iterations = (\d+)',int)
        for phase in ('factor','solve'): c[phase+'_seconds']=number(text,r'#\s+- '+phase+r' time = (\S+)')
        c['factor_plus_solve_seconds']=c['factor_seconds']+c['solve_seconds']
        if not (0<=c['relative_error']<1e-8 and 0<=c['scaled_residual']<1e-6 and c['hodlr_fronts']>1): raise ValueError('Numerical/multiple-front check failed')
        binding=Counter(re.findall(r'# profile binding .* uuid=(\S+)',alltext));c['gpu_uuid_rank_counts']=dict(binding)
        if len(binding)!=plan['ranks']//2 or set(binding.values())!={2}: raise ValueError('Two-ranks-per-GPU binding failed')
        c['prepared_blocks']={b:len(re.findall(r'GPU front evaluator block='+b+r'.* preparation=',alltext)) for b in ('F11','F12','F21','F22')}
        c['cap_fallback_reports']=alltext.count('reason=prepared_image_exceeds_byte_cap')
        c['device_schur_calls_reported']=sum(map(int,re.findall(r'# GPU Schur .* calls_local=(\d+)',alltext)))
        if label!='cpu_timing' and not (c['prepared_blocks']['F12'] and c['device_schur_calls_reported']): raise ValueError('Current GPU path not exercised')
        if label in ('gpu_diagnostic','gpu_trace'):
            perrank=[];byphase=defaultdict(lambda:defaultdict(list));fronts=[]
            for rank in range(plan['ranks']):
                path=root/f'{label}.stages.rank{rank}.csv'; rows=list(csv.DictReader(path.open())); totals=defaultdict(lambda:[0.,0.,0])
                for row in rows:
                    inc,exc=float(row['inclusive']),float(row['exclusive'])
                    if not (math.isfinite(inc) and 0<=exc<=inc+1e-8): raise ValueError('Invalid stage duration')
                    key=(row['section'],row['phase']);t=totals[key];t[0]+=inc;t[1]+=exc;t[2]+=1
                    if row['phase'].endswith('.construct.host') and int(row['front_rank'])==0:
                        fronts.append(dict(rank=rank,front=int(row['front']),phase=row['phase'],rows=int(row['rows']),cols=int(row['cols']),procs=int(row['procs']),seconds=inc))
                if ('solver.factor','solver.factor') not in totals: raise ValueError('Missing solver factor scope')
                factor=totals[('solver.factor','solver.factor')][0]
                partition=sum(t[1] for (section,phase),t in totals.items() if section=='solver.factor')
                if abs(partition-factor)>max(.001,1e-8*factor): raise ValueError('Exclusive stage partition does not match factor envelope')
                peak=max(int(row['peak_image_bytes']) for row in rows)
                perrank.append(dict(rank=rank,factor_envelope_seconds=factor,exclusive_partition_seconds=partition,peak_resident_image_bytes=peak))
                for key,t in totals.items():
                    byphase[key]['inclusive'].append((rank,t[0]));byphase[key]['exclusive'].append((rank,t[1]));byphase[key]['calls'].append((rank,t[2]))
            c['rank_accounting']=perrank;c['stages']=[]
            for (section,phase),data in byphase.items():
                group=dict(section=section,phase=phase,calls=int(sum(v for _,v in data['calls'])))
                for kind in ('inclusive','exclusive'):
                    values=dict(data[kind]);v=[values.get(r,0.) for r in range(plan['ranks'])]
                    group[kind]=dict(mean_rank_seconds=statistics.mean(v),max_rank_seconds=max(v),sum_rank_seconds=sum(v))
                c['stages'].append(group)
            c['stages'].sort(key=lambda x:x['exclusive']['mean_rank_seconds'],reverse=True)
            c['largest_host_construction_fronts']=sorted(fronts,key=lambda x:x['seconds'],reverse=True)[:20]
        if label=='gpu_trace':
            c['cuda']=[json.loads((root/f'cuda.rank{r}.json').read_text()) for r in sorted({0,plan['ranks']-1})]
            if not all(x['kernel_count']>0 and x['nvtx_range_count']>0 for x in c['cuda']): raise ValueError('Missing CUDA/NVTX evidence')
        c['status']='PASS'
    except (OSError,ValueError,KeyError) as e:
        c['error']=str(e);failures.append(label+': '+str(e))
    cases.append(c)
complete=len(cases)==len(plan['variants']) and not failures
result=dict(status='FAIL' if failures else ('PASS' if complete else 'RUNNING'),job=root.name,plan=plan,cases=cases,failures=failures,
    notes=['Timing samples use frozen uninstrumented executable; phase/trace diagnostics are separate runs.',
           'All stage times are local host wall times; use exclusive times for disjoint accounting, inclusive times overlap.',
           'Do not sum rank maxima to estimate solver critical-path time.',
           'Resident image bytes exclude native factors, workspaces, metadata and MPI arenas.',
           'CUDA traces cover ranks zero and last only, with no MPI interposition; GPU activity is per-process, not device occupancy.'])
(root/'profile150_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['status'],[(c['label'],c['status'],c.get('factor_seconds')) for c in cases])
if failures or (a.require_complete and not complete): raise SystemExit(1)
