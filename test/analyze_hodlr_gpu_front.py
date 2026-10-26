#!/usr/bin/env python3
"""Reject sparse runs without finite accuracy and actual device evaluation."""
import json, math, re, sys
from pathlib import Path
out=Path(sys.argv[1])
def require(condition,message):
    if not condition: raise SystemExit(message)
for file,needle in [('device_contract.log','GPU front evaluator device contract: PASS'),
                    ('integration.log','GPU front evaluator HODLR integration: PASS'),
                    ('device_memcheck.log','ERROR SUMMARY: 0 errors')]:
    require(needle in (out/file).read_text(),f'{file} failed')
records={}
for ranks in [1,4]:
    for variant in ['hybrid','evaluator','capped']:
        name=f'{variant}_{ranks}'
        text='\n'.join((out/f'{name}.rank{rank}.log').read_text() for rank in range(ranks))
        root=(out/f'{name}.rank0.log').read_text()
        # The Poisson driver reports its achieved forward error and residual.
        error=re.search(r'# relative error = .* = (\S+)',root,re.I)
        residual=re.search(r'# COMPONENTWISE SCALED RESIDUAL = (\S+)',root,re.I)
        require(error and residual,f'{name}: missing accuracy')
        values=[float(error[1]),float(residual[1])]
        require(all(math.isfinite(x) and x>=0 for x in values),f'{name}: nonfinite result')
        require(values[0]<1.e-8 and values[1]<1.e-6,f'{name}: inaccurate solve {values}')
        fronts=re.search(r'nr of HODLR Frontal matrices = (\d+)',root)
        require(fronts and int(fronts[1])>1,f'{name}: expected multiple HODLR fronts')
        calls=[int(x) for x in re.findall(r'evaluator_calls=(\d+)',text)]
        require(calls,f'{name}: missing backend reports')
        if variant=='evaluator':
            require(max(calls)>0,f'{name}: no device evaluation')
            require('HODLR GPU construction level' in text,f'{name}: no native GPU BACA')
            for block in ['F11','F22']:
                prepared=re.search(r'GPU front evaluator block='+block+r'.* preparation=host_',text)
                require(prepared,f'{name}: missing prepared device {block}')
            if ranks>1:
                require('distributed_front_requires_collective_preparation' in text,f'{name}: distributed fallback missing')
        else:
            require(max(calls)==0,f'{name}: evaluator unexpectedly enabled')
            if variant=='capped': require('prepared_image_exceeds_byte_cap' in text,f'{name}: cap fallback missing')
        construction=[line for line in text.splitlines() if 'operation=construction' in line]
        def total(field):
            matches=[re.search(field+r'=(\d+)',line) for line in construction]
            return sum(int(match[1]) for match in matches if match)
        factors=re.findall(r'#\s+- factor time = (\S+)',root)
        solves=re.findall(r'#\s+- solve time = (\S+)',root)
        require(len(factors)==1 and len(solves)==1,f'{name}: missing wall times')
        times=[float(factors[0]),float(solves[0])]
        require(all(math.isfinite(x) and x>0 for x in times),f'{name}: invalid timings')
        records[name]={'relative_error':values[0],'scaled_residual':values[1],
                       'hodlr_fronts':int(fronts[1]),'maximum_evaluator_calls':max(calls),
                       'evaluator_calls_sum':total('evaluator_calls'),
                       'evaluator_upload_bytes_sum':total('evaluator_upload_bytes_local'),
                       'factor_seconds_diagnostic':times[0],'solve_seconds_diagnostic':times[1],
                       'native_gpu_baca_level_reports':text.count('HODLR GPU construction level')}

result={'status':'PASS','job':out.name,'cases':records,
        'scope':'Bounded single-rank fronts; distributed host construction; three RHS',
        'timing_scope':'Verbose functional diagnostics; not a speed benchmark'}
(out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
