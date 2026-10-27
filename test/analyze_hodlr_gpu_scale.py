#!/usr/bin/env python3
"""Incremental numerical validation and uninstrumented scaling summaries."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import statistics

parser=argparse.ArgumentParser()
parser.add_argument('directory',type=Path)
parser.add_argument('--ranks',type=int,required=True)
parser.add_argument('--require-complete',action='store_true')
args=parser.parse_args()
root=args.directory

def scalar(text,pattern,kind=float):
    matches=re.findall(pattern,text)
    if len(matches)!=1: raise ValueError('Expected one '+pattern)
    value=kind(matches[0])
    if not math.isfinite(value): raise ValueError('Nonfinite '+pattern)
    return value

plan=json.loads((root/'plan.json').read_text()) if (root/'plan.json').exists() else dict(grids=[100,150,200],repetitions=3)
expected={(n,variant,repeat) for n in plan['grids'] for variant in ['cpu','hybrid','evaluator']
          for repeat in range(1,plan['repetitions']+1)}
cases=[]
failures=[]
for status_path in sorted(root.glob('n*_r*.exit_code')):
    label=status_path.name[:-len('.exit_code')]
    match=re.fullmatch(r'n(\d+)_(cpu|hybrid|evaluator)_r([123])',label)
    if not match: continue
    n,variant,repeat=int(match[1]),match[2],int(match[3])
    case=dict(label=label,grid=n,unknowns=n**3,variant=variant,repeat=repeat,
              exit_code=int(status_path.read_text()),status='FAIL')
    try:
        if case['exit_code']: raise ValueError('Solver step exit '+str(case['exit_code']))
        logs=[root/(label+'.rank%d.log'%rank) for rank in range(args.ranks)]
        if not all(path.exists() for path in logs): raise ValueError('Missing rank logs')
        text=logs[0].read_text()
        case['relative_error']=scalar(text,r'# relative error = .* = (\S+)')
        case['scaled_residual']=scalar(text,r'# COMPONENTWISE SCALED RESIDUAL = (\S+)')
        case['hodlr_fronts']=scalar(text,r'nr of HODLR Frontal matrices = (\d+)',int)
        case['refinement_iterations']=scalar(text,r'number of Krylov iterations = (\d+)',int)
        for phase in ['factor','solve']:
            case[phase+'_seconds']=scalar(text,r'#\s+- '+phase+r' time = (\S+)')
            if case[phase+'_seconds']<=0: raise ValueError('Nonpositive phase time')
        case['factor_plus_solve_seconds']=case['factor_seconds']+case['solve_seconds']
        if not (0<=case['relative_error']<1e-8 and 0<=case['scaled_residual']<1e-6 and case['hodlr_fronts']>1):
            raise ValueError('Accuracy or multi-front check failed')
        bindings=[]
        prepared=Counter()
        fallbacks=Counter()
        for path in logs:
            rank_text=path.read_text()
            bindings+=re.findall(r'# scale binding .* uuid=(\S+)',rank_text)
            for block in ['F11','F22']:
                prepared[block]+=len(re.findall(r'GPU front evaluator block='+block+r'.* preparation=',rank_text))
            for reason in ['prepared_image_exceeds_byte_cap','distributed_front_requires_collective_preparation']:
                fallbacks[reason]+=rank_text.count('reason='+reason)
        if len(bindings)!=args.ranks or len(set(bindings))!=args.ranks//2 or set(Counter(bindings).values())!={2}:
            raise ValueError('Expected exactly two ranks per GPU')
        case['prepared_device_blocks']=dict(prepared)
        case['host_fallback_counts']=dict(fallbacks)
        case['gpu_uuid_rank_counts']=dict(Counter(bindings))
        if variant=='evaluator' and not prepared['F11']: raise ValueError('No GPU front was prepared')
        if variant!='evaluator' and sum(prepared.values()): raise ValueError('Unexpected evaluator activation')
        case['status']='PASS'
    except (ValueError,OSError) as error:
        case['error']=str(error); failures.append(label+': '+str(error))
    cases.append(case)

summaries=[]
for n in plan['grids']:
    for variant in ['cpu','hybrid','evaluator']:
        trials=[case for case in cases if case['grid']==n and case['variant']==variant and case['status']=='PASS']
        if not trials: continue
        group=dict(grid=n,variant=variant,completed_repetitions=len(trials))
        for phase in ['factor_seconds','solve_seconds','factor_plus_solve_seconds']:
            values=[case[phase] for case in trials]
            group[phase]=dict(median=statistics.median(values),minimum=min(values),maximum=max(values))
        summaries.append(group)
complete={(case['grid'],case['variant'],case['repeat']) for case in cases}==expected and not failures
result=dict(status='FAIL' if failures else ('PASS' if complete else 'RUNNING'),job=root.name,
            ranks=args.ranks,nodes=args.ranks//8,gpus=args.ranks//2,ranks_per_gpu=2,
            rhs=3,plan=plan,source_commit=(root/'source_commit.txt').read_text().strip(),
            cases=cases,summaries=summaries,failures=failures,
            notes=['Uninstrumented Release runs; native verbose checks disabled.',
                   'Fresh-process repetitions per the recorded plan, alternating variant order; one repetition gives preliminary timings.',
                   'Device evaluator uses an 8 MiB per-block cap and explicit host fallback.',
                   'Prepared-device counts require successful compression callbacks internally.',
                   'Different compression samples/refinement counts can change solve timings.'])
(root/'scale_summary.json').write_text(json.dumps(result,indent=2)+'\n')
print('{}: {} completed cases, {} failures'.format(result['status'],len(cases),len(failures)))
if failures or (args.require_complete and not complete): raise SystemExit(1)
