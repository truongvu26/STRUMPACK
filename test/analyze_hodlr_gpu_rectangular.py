#!/usr/bin/env python3
"""Require numerical, rectangular-device and distributed-construction evidence."""
import json
import math
from pathlib import Path
import re
import sys
root=Path(sys.argv[1]); failures=[]; cases=[]
def number(text,pattern,kind=float):
    found=re.findall(pattern,text)
    if len(found)!=1: raise ValueError('Expected one '+pattern)
    value=kind(found[0].replace(',',''))
    if not math.isfinite(value): raise ValueError('Nonfinite '+pattern)
    return value
for ranks in (1,3,4):
    try:
        for rank in range(ranks):
            if 'Rectangular GPU construction/multiply and device Schur: PASS' in (root/f'direct_{ranks}.rank{rank}.log').read_text(): break
        else: raise ValueError('No direct-test PASS marker')
    except (OSError,ValueError) as error: failures.append(f'direct_{ranks}: {error}')
    if ranks==3: continue
    for variant in ('hybrid','rectangular','constructed'):
        label=f'{variant}_{ranks}'; case=dict(label=label,status='FAIL',ranks=ranks)
        try:
            if int((root/(label+'.exit')).read_text()): raise ValueError('Nonzero solver exit')
            logs=[(root/f'{label}.rank{rank}.log').read_text() for rank in range(ranks)]
            text=logs[0]; all_text='\n'.join(logs)
            case['relative_error']=number(text,r'# relative error = .* = (\S+)')
            case['scaled_residual']=number(text,r'# COMPONENTWISE SCALED RESIDUAL = (\S+)')
            case['hodlr_fronts']=number(text,r'nr of HODLR Frontal matrices = ([\d,]+)',int)
            for phase in ('factor','solve'): case[phase+'_seconds']=number(text,r'#\s+- '+phase+r' time = (\S+)')
            if case['relative_error']>=1e-8 or case['scaled_residual']>=1e-6 or case['hodlr_fronts']<2:
                raise ValueError('Accuracy or multiple-front condition failed')
            case['device_schur_calls']=sum(map(int,re.findall(r'# GPU Schur .* calls_local=(\d+)',all_text)))
            case['device_schur_boundary_bytes']=sum(map(int,re.findall(r'# GPU Schur .* host_boundary_bytes_local=(\d+)',all_text)))
            case['rectangular_gpu_construction_reports']=all_text.count('# Rectangular GPU construction')
            case['collective_prepared_images']=len(re.findall(r'preparation=collective_host_image ranks=[234]',all_text))
            case['resident_entry_bytes_reports']=sum(map(int,re.findall(r'resident_bytes_local=(\d+)',all_text)))
            if variant!='hybrid' and not case['device_schur_calls']: raise ValueError('No device Schur calls')
            if variant=='constructed':
                if not case['rectangular_gpu_construction_reports']: raise ValueError('No rectangular GPU BACA construction')
                if not case['resident_entry_bytes_reports']: raise ValueError('No resident entry images')
                if ranks==4 and not case['collective_prepared_images']: raise ValueError('No distributed construction preparation')
            case['status']='PASS'
        except (OSError,ValueError) as error:
            case['error']=str(error); failures.append(label+': '+str(error))
        cases.append(case)
for name,marker in [('device_contract.log','PASS'),('device_memcheck.log','ERROR SUMMARY: 0 errors'),('square_integration.log','GPU front evaluator HODLR integration: PASS')]:
    try:
        if marker not in (root/name).read_text(): raise ValueError('Missing '+marker)
    except (OSError,ValueError) as error: failures.append(name+': '+str(error))
result=dict(status='FAIL'  if failures else 'PASS',job=root.name,grid=40,rhs=3,
    cases=cases,failures=failures,notes=[
      'One trial per variant/layout; preliminary timings, no speedup claim.',
      'Distributed entry preparation is a bounded replicated host image uploaded once per matrix.',
      'GPU Schur intermediates stay on device; MPI coefficient reductions may use host staging.',
      'Native BF host mirrors remain for existing extraction; nonzero butterfly levels remain unsupported.'])
(root/'rectangular_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(result['status'],json.dumps(failures))
if failures: raise SystemExit(1)
