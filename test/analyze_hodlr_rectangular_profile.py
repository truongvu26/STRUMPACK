#!/usr/bin/env python3
"""Controlled same-allocation, fresh-process comparison of the new paths."""
from collections import Counter
import json
import math
from pathlib import Path
import re
import statistics
import sys
root=Path(sys.argv[1]); cases=[]; failures=[]
def scalar(text,pattern,kind=float):
    found=re.findall(pattern,text)
    if len(found)!=1: raise ValueError('Expected one '+pattern)
    x=kind(found[0].replace(',',''))
    if not math.isfinite(x): raise ValueError('Nonfinite '+pattern)
    return x
for repeat in (1,2,3):
    for variant in ('cpu','hybrid','rectangular','constructed'):
        label=f'{variant}_r{repeat}'; case=dict(label=label,variant=variant,repeat=repeat,status='FAIL')
        try:
            if int((root/(label+'.exit')).read_text()): raise ValueError('Nonzero solver exit')
            logs=[(root/f'{label}.rank{rank}.log').read_text() for rank in range(4)]
            text=logs[0]; all_text='\n'.join(logs)
            case['relative_error']=scalar(text,r'# relative error = .* = (\S+)')
            case['scaled_residual']=scalar(text,r'# COMPONENTWISE SCALED RESIDUAL = (\S+)')
            case['hodlr_fronts']=scalar(text,r'nr of HODLR Frontal matrices = ([\d,]+)',int)
            case['refinement_iterations']=scalar(text,r'number of Krylov iterations = (\d+)',int)
            for phase in ('factor','solve'): case[phase+'_seconds']=scalar(text,r'#\s+- '+phase+r' time = (\S+)')
            case['total_seconds']=case['factor_seconds']+case['solve_seconds']
            if not (0<=case['relative_error']<1e-8 and 0<=case['scaled_residual']<1e-6 and case['hodlr_fronts']>1):
                raise ValueError('Accuracy or multiple-front requirement failed')
            bindings=Counter(re.findall(r'# shared-GPU binding .* uuid=(\S+)',all_text))
            if len(bindings)!=2 or set(bindings.values())!={2}: raise ValueError('Expected two ranks per each of two GPUs')
            case['gpu_bindings']=dict(bindings)
            case['device_schur_calls_reported']=sum(map(int,re.findall(r'# GPU Schur .* calls_local=(\d+)',all_text)))
            case['rectangular_gpu_construction_reports']=all_text.count('# Rectangular GPU construction')
            case['collective_prepared_images']=len(re.findall(r'preparation=collective_host_image ranks=[234]',all_text))
            if variant in ('rectangular','constructed') and not case['device_schur_calls_reported']:
                raise ValueError('Device Schur pipeline not exercised')
            if variant=='constructed' and not (case['rectangular_gpu_construction_reports'] and case['collective_prepared_images']):
                raise ValueError('Rectangular/distributed GPU construction not exercised')
            case['status']='PASS'
        except (OSError,ValueError) as e:
            case['error']=str(e); failures.append(label+': '+str(e))
        cases.append(case)
summaries=[]
for variant in ('cpu','hybrid','rectangular','constructed'):
    trials=[c for c in cases if c['variant']==variant and c['status']=='PASS']
    if not trials: continue
    group=dict(variant=variant,trials=len(trials),refinement_iterations=[c['refinement_iterations'] for c in trials])
    for phase in ('factor_seconds','solve_seconds','total_seconds'):
        values=[c[phase] for c in trials]
        group[phase]=dict(median=statistics.median(values),minimum=min(values),maximum=max(values))
    summaries.append(group)
result=dict(status='FAIL' if failures else 'PASS',job=root.name,grid=60,rhs=3,ranks=4,gpus=2,ranks_per_gpu=2,
    cap_bytes=268435456,hodlr_relative_tolerance=1e-5,source_commit=(root/'source_commit.txt').read_text().strip(),
    butterflypack_commit=(root/'butterflypack_commit.txt').read_text().strip(),
    cases=cases,summaries=summaries,failures=failures,notes=[
        'Three alternating fresh-process trials per variant, one allocation, identical Release executable.',
        'Task timers/native verbose checks disabled; API counters and preparation messages remain enabled.',
        'Adaptive compression samples and refinement work vary; ranges are observations, not confidence intervals.',
        'Constructed variant uses bounded replicated host entry preparation and resident device images.',
        'Factor/solve exclude initial sparse matrix creation and distribution.'])
(root/'rectangular_profile.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['status'],json.dumps(summaries))
if failures: raise SystemExit(1)
