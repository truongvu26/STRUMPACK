#!/usr/bin/env python3
"""Validate each solver case and aggregate local, nested phase measurements."""
import argparse,csv,json,math,re,statistics,subprocess,sys
from collections import Counter,defaultdict
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('directory',type=Path); p.add_argument('--require-complete',action='store_true'); a=p.parse_args(); root=a.directory
n,ranks=map(int,(root/'layout.txt').read_text().split()); expected=(root/'cases.txt').read_text().split()
result=dict(status='RUNNING',job=root.name,grid=n,unknowns=n**3,ranks=ranks,rhs=3,cases=[],notes=[
 'Quiet GPU/CPU cases supply baseline wall times. Diagnostic and traced cases add overhead.',
 'Phase totals sum local rank work, not global elapsed time. Nested inclusive times overlap.',
 'Exclusive values subtract instrumented child scopes only; uninstrumented work remains in the parent.',
 'Resident-image bytes exclude ButterflyPACK factors, pool reservations and temporary device allocations.',
 'One quiet trial per backend gives a preliminary comparison, not a repeated benchmark.'])
failures=[]
heap_file=root/'heap_fraction.txt'
result['bpack_gpu_heap_fraction']=float(heap_file.read_text()) if heap_file.exists() else .85
inventory=root/'devices.txt'
if inventory.exists():
    devices=[]
    for row in csv.reader(inventory.read_text().splitlines()):
        if len(row)==4 and row[0].strip().startswith('NVIDIA'):
            devices.append(dict(name=row[0].strip(),uuid=row[1].strip(),memory_mib=int(row[2].split()[0]),driver=row[3].strip()))
    result['gpu_devices']=devices
def number(text,pattern):
    matches=re.findall(pattern,text)
    if len(matches)!=1: raise ValueError('Expected one '+pattern)
    v=float(matches[0].replace(',',''))
    if not math.isfinite(v): raise ValueError('Nonfinite '+pattern)
    return v
for variant in expected:
    status=root/(variant+'.exit_code')
    if not status.exists(): continue
    case=dict(variant=variant,status='FAIL',exit_code=int(status.read_text()))
    try:
        if case['exit_code']: raise ValueError('Solver exit '+str(case['exit_code']))
        logs=[root/f'{variant}.rank{r}.log' for r in range(ranks)]
        if not all(x.exists() for x in logs): raise ValueError('Missing rank log')
        text=logs[0].read_text()
        for key,pattern in [('relative_error',r'# relative error = .* = (\S+)'),('scaled_residual',r'# COMPONENTWISE SCALED RESIDUAL = (\S+)'),('factor_seconds',r'#\s+- factor time = (\S+)'),('solve_seconds',r'#\s+- solve time = (\S+)'),('hodlr_fronts',r'nr of HODLR Frontal matrices = ([\d,]+)'),('refinement_iterations',r'number of Krylov iterations = (\d+)')]:
            case[key]=number(text,pattern)
        case['factor_plus_solve_seconds']=case['factor_seconds']+case['solve_seconds']
        if not (0<=case['relative_error']<1e-8 and 0<=case['scaled_residual']<1e-6 and case['hodlr_fronts']>1): raise ValueError('Numerical check failed')
        bindings=Counter(x for log in logs for x in re.findall(r'# stage binding .* uuid=(\S+)',log.read_text()))
        if sum(bindings.values())!=ranks or len(bindings)!=ranks//2 or set(bindings.values())!={2}: raise ValueError('Two ranks per GPU binding failed')
        case['gpu_uuid_rank_counts']=dict(bindings)
        if variant!='cpu' and (root/'cusolver_audit_enabled').exists():
            audit=[json.loads((root/f'{variant}.cusolver.rank{rank}.json').read_text()) for rank in range(ranks)]
            if not all(x['handles_created']==1 and x['params_created']==1 and x['creation_failures']==0 for x in audit):
                raise ValueError('cuSOLVER resources are not bounded to one handle and parameter object per process')
            case['cusolver_resource_audit']=dict(ranks=ranks,handles_per_process=1,params_per_process=1,creation_failures=0)
        if variant in ('diagnostic','traced'):
            by_phase=defaultdict(lambda:defaultdict(list)); fronts=[]; peaks=[]; accounting=[]
            for rank in range(ranks):
                path=root/f'{variant}.stages.rank{rank}.csv'
                with path.open() as f: rows=list(csv.DictReader(f))
                if not rows: raise ValueError('Empty phase CSV')
                peaks.append(max(int(row['peak_image_bytes']) for row in rows))
                labels=Counter(row['phase'] for row in rows)
                if any(labels[k]!=1 for k in ('solver.reorder','solver.factor','solver.solve')): raise ValueError('Missing or duplicate solver scope')
                if not labels['prepare.upload_resident_image']: raise ValueError('Missing CUDA translation-unit measurement')
                factor_rows=[row for row in rows if row['section']=='solver.factor']
                factor_wall=next(float(row['inclusive']) for row in factor_rows if row['phase']=='solver.factor')
                partition=sum(float(row['exclusive']) for row in factor_rows)
                if abs(partition-factor_wall)>max(.001,1e-8*factor_wall): raise ValueError('Exclusive scopes do not partition factor envelope')
                accounting.append(dict(rank=rank,factor_envelope_seconds=factor_wall,exclusive_partition_seconds=partition))
                for row in rows:
                    inc=float(row['inclusive']); exc=float(row['exclusive'])
                    if not (math.isfinite(inc) and 0<=exc<=inc+1e-6): raise ValueError('Invalid phase duration')
                    by_phase[(row['section'],row['phase'])][rank].append(row)
                    if row['section']=='solver.factor' and '.construct.' in row['phase']:
                        fronts.append(dict(rank=rank,front=int(row['front']),rows=int(row['rows']),cols=int(row['cols']),procs=int(row['procs']),phase=row['phase'],seconds=inc))
            phases=[]
            for (section,phase),local in by_phase.items():
                rec=dict(section=section,phase=phase,calls=sum(map(len,local.values())),participating_ranks=len(local))
                for metric in ('inclusive','exclusive'):
                    values=[sum(float(row[metric]) for row in local.get(rank,[])) for rank in range(ranks)]
                    rec[metric+'_rank_seconds']=dict(sum=sum(values),mean=statistics.mean(values),max=max(values),max_rank=values.index(max(values)))
                rec['payload_bytes_sum']=sum(int(row['bytes']) for rows in local.values() for row in rows)
                phases.append(rec)
            case['phases']=sorted(phases,key=lambda r:r['exclusive_rank_seconds']['sum'],reverse=True)
            case['slowest_constructions']=sorted(fronts,key=lambda x:x['seconds'],reverse=True)[:60]
            grouped=defaultdict(list)
            for row in fronts:
                key=tuple(row[k] for k in ('phase','front','rows','cols','procs'))
                grouped[key].append(row)
            unique=[]
            for key,rows in grouped.items():
                rec=dict(zip(('phase','front','rows','cols','procs'),key))
                slowest=max(rows,key=lambda row:row['seconds'])
                rec.update(calls=len(rows),participating_ranks=len({row['rank'] for row in rows}),
                           max_rank=slowest['rank'],max_seconds=slowest['seconds'],
                           mean_seconds=statistics.mean(row['seconds'] for row in rows),
                           rank_seconds=sum(row['seconds'] for row in rows))
                unique.append(rec)
            case['construction_fronts']=sorted(unique,key=lambda row:row['max_seconds'],reverse=True)
            case['peak_resident_image_bytes_by_rank']=peaks
            case['rank_accounting']=accounting
        if variant=='traced':
            report_status=root/'traced.report_exit_code'
            if report_status.exists() and int(report_status.read_text()): raise ValueError('CUDA report conversion failed')
            case['cuda']=[json.loads((root/f'rank{rank}.cuda.json').read_text()) for rank in sorted({0,ranks-1})]
            if not all(x['kernel_count']>0 and x.get('nvtx_events',0)>0 for x in case['cuda']): raise ValueError('Missing CUDA/NVTX evidence')
        case['status']='PASS'
    except (ValueError,OSError,KeyError) as e:
        case['error']=str(e); failures.append(variant+': '+str(e))
    result['cases'].append(case)
complete=len(result['cases'])==len(expected)
result['status']='FAIL' if failures else ('PASS' if complete else 'RUNNING'); result['failures']=failures
(root/'stage_summary.json').write_text(json.dumps(result,indent=2)+'\n')
subprocess.run([sys.executable,str(Path(__file__).with_name('render_hodlr_stage_report.py')),str(root)],check=True)
print(result['status'],[(c['variant'],c['status'],c.get('factor_plus_solve_seconds'),c.get('error')) for c in result['cases']],flush=True)
if failures or (a.require_complete and not complete): raise SystemExit(1)
