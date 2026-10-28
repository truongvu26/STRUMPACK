#!/usr/bin/env python3
"""Validate each solver case and aggregate local, nested phase measurements."""
import argparse,bisect,csv,json,math,re,shlex,statistics,subprocess,sys
from collections import Counter,defaultdict
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('directory',type=Path); p.add_argument('--require-complete',action='store_true'); a=p.parse_args(); root=a.directory
n,ranks=map(int,(root/'layout.txt').read_text().split()); expected=(root/'cases.txt').read_text().split()
sharing_file=root/'rank_per_gpu.txt'
ranks_per_gpu=int(sharing_file.read_text()) if sharing_file.exists() else 2
if ranks_per_gpu not in (1,2) or ranks%ranks_per_gpu:
    raise ValueError('Expected one or two MPI ranks per GPU, with complete sharing groups')
result=dict(status='RUNNING',job=root.name,grid=n,unknowns=n**3,ranks=ranks,ranks_per_gpu=ranks_per_gpu,rhs=3,cases=[],notes=[
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
def panel_statistics(logs,budget):
    """All-front-rank counters, available only when stage profiling is enabled."""
    records=[]
    integer=('rows','cols','procs','front_rank','calls_local','windows_local',
             'upload_bytes_local','staging_peak_bytes_local')
    timers=('host_prepare_seconds_local','metadata_seconds_local','upload_seconds_local')
    for rank,log in enumerate(logs):
        for line in log.read_text().splitlines():
            if not line.startswith('# GPU host panels '): continue
            values=dict(word.split('=',1) for word in line.split()[4:])
            rec=dict(rank=rank,kind=values['kind'])
            rec.update((key,int(values[key])) for key in integer)
            rec.update((key,float(values[key])) for key in timers)
            if (rec['kind'] not in ('HODLR','BF') or any(rec[k]<0 for k in integer)
                or any(not math.isfinite(rec[k]) or rec[k]<0 for k in timers)
                or rec['procs']<1 or rec['front_rank']>=rec['procs']
                or rec['staging_peak_bytes_local']>budget):
                raise ValueError('Invalid host-panel counters or staging budget exceeded')
            records.append(rec)
    if not records or not sum(x['upload_bytes_local'] for x in records):
        raise ValueError('Missing all-rank host-panel upload statistics')
    totals=[]
    for rank in range(len(logs)):
        local=[x for x in records if x['rank']==rank]
        totals.append(dict(rank=rank,constructions=len(local),
            **{k:sum(x[k] for x in local) for k in ('calls_local','windows_local','upload_bytes_local')+timers},
            staging_peak_bytes_local=max((x['staging_peak_bytes_local'] for x in local),default=0)))
    seconds={key[:-6]:dict(
        mean=statistics.mean(x[key] for x in totals),max=max(x[key] for x in totals),
        sum=sum(x[key] for x in totals)) for key in timers}
    return dict(coverage='all front ranks; stage profiling enabled',
        construction_rank_calls=len(records),
        upload_bytes_sum=sum(x['upload_bytes_local'] for x in totals),
        staging_peak_bytes_max=max(x['staging_peak_bytes_local'] for x in totals),
        staging_budget_bytes_per_rank=budget,rank_totals=totals,wall_seconds=seconds,
        records=records)
def trace_statistics():
    report_status=root/'traced.report_exit_code'
    if report_status.exists() and int(report_status.read_text()): raise ValueError('CUDA report conversion failed')
    selected=sorted({int(x) for x in (root/'traced_ranks.txt').read_text().strip().split(',')}) if (root/'traced_ranks.txt').exists() else sorted({0,ranks-1})
    cuda=[json.loads((root/f'rank{rank}.cuda.json').read_text()) for rank in selected]
    if not all(x['kernel_count']>0 and x.get('nvtx_events',0)>0 for x in cuda): raise ValueError('Missing CUDA/NVTX evidence')
    result=dict(cuda=cuda,trace_data_status='COMPLETE')
    if (root/'nsys_cpu_sampling.txt').exists() and (root/'nsys_cpu_sampling.txt').read_text().strip()=='1':
        result['cpu_sampling']=[json.loads((root/f'rank{rank}.cpu.json').read_text()) for rank in selected]
        if not all(any(s['section']=='solver.factor' and s['running_samples']>0 for s in x['sections']) for x in result['cpu_sampling']):
            raise ValueError('Missing factorization CPU call-stack samples')
    return result
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
        if (sum(bindings.values())!=ranks or len(bindings)!=ranks//ranks_per_gpu
            or set(bindings.values())!={ranks_per_gpu}):
            raise ValueError('Expected '+str(ranks_per_gpu)+' ranks per GPU binding failed')
        case['gpu_uuid_rank_counts']=dict(bindings)
        if variant!='cpu' and (root/'cusolver_audit_enabled').exists():
            audit=[json.loads((root/f'{variant}.cusolver.rank{rank}.json').read_text()) for rank in range(ranks)]
            if not all(x['handles_created']==1 and x['params_created']==1 and x['creation_failures']==0 for x in audit):
                raise ValueError('cuSOLVER resources are not bounded to one handle and parameter object per process')
            case['cusolver_resource_audit']=dict(ranks=ranks,handles_per_process=1,params_per_process=1,creation_failures=0)
        if variant in ('diagnostic','traced'):
            by_phase=defaultdict(lambda:defaultdict(list)); fronts=[]; peaks=[]; accounting=[]
            arrival_gaps=defaultdict(list)
            for rank in range(ranks):
                path=root/f'{variant}.stages.rank{rank}.csv'
                with path.open() as f: rows=list(csv.DictReader(f))
                if not rows: raise ValueError('Empty phase CSV')
                peaks.append(max(int(row['peak_image_bytes']) for row in rows))
                labels=Counter(row['phase'] for row in rows)
                if any(labels[k]!=1 for k in ('solver.reorder','solver.factor','solver.solve')): raise ValueError('Missing or duplicate solver scope')
                if not (labels['prepare.upload_resident_image'] or labels['prepare.host_panels']): raise ValueError('Missing GPU image or host-panel preparation measurement')
                factor_rows=[row for row in rows if row['section']=='solver.factor']
                compress=[row for row in factor_rows if row['phase']=='front.compress']
                ends=sorted(float(row['end']) for row in compress)
                for row in compress:
                    start=float(row['start']); before=bisect.bisect_right(ends,start)
                    if int(row['procs'])>1 and before:
                        arrival_gaps[(int(row['front']),int(row['procs']))].append(
                            dict(rank=rank,seconds=start-ends[before-1]))
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
            gaps=[]
            for (front,procs),local in arrival_gaps.items():
                slowest=max(local,key=lambda row:row['seconds'])
                low=min(row['seconds'] for row in local)
                gaps.append(dict(front=front,procs=procs,participating_ranks=len(local),
                    min_seconds=low,max_seconds=slowest['seconds'],
                    spread_seconds=slowest['seconds']-low,max_rank=slowest['rank']))
            case['parent_precompression_gaps']=sorted(gaps,key=lambda row:row['spread_seconds'],reverse=True)[:20]
            command=root/(variant+'.command')
            args=shlex.split(command.read_text()) if command.exists() else []
            if '--hodlr_gpu_host_panels' in args and args[args.index('--hodlr_gpu_host_panels')+1]=='1':
                budget=int(args[args.index('--hodlr_gpu_panel_max_bytes')+1]) if '--hodlr_gpu_panel_max_bytes' in args else 67108864
                case['host_panels']=panel_statistics(logs,budget)
                if case['host_panels']['construction_rank_calls']!=sum(
                    phase['calls'] for phase in phases if '.construct.gpu_panels' in phase['phase']):
                    raise ValueError('Host-panel counters do not cover every construction rank call')
        if variant=='traced':
            case.update(trace_statistics())
        case['status']='PASS'
    except (ValueError,OSError,KeyError) as e:
        case['error']=str(e); failures.append(variant+': '+str(e))
        if variant=='traced':
            try:
                case.update(trace_statistics())
                case['trace_data_status']='COMPLETE_FROM_FAILED_NUMERICAL_CASE'
            except (ValueError,OSError,KeyError) as trace_error:
                case['trace_data_status']='INCOMPLETE'; case['trace_data_error']=str(trace_error)
    result['cases'].append(case)
complete=len(result['cases'])==len(expected)
result['status']='FAIL' if failures else ('PASS' if complete else 'RUNNING'); result['failures']=failures
(root/'stage_summary.json').write_text(json.dumps(result,indent=2)+'\n')
subprocess.run([sys.executable,str(Path(__file__).with_name('render_hodlr_stage_report.py')),str(root)],check=True)
print(result['status'],[(c['variant'],c['status'],c.get('factor_plus_solve_seconds'),c.get('error')) for c in result['cases']],flush=True)
if failures or (a.require_complete and not complete): raise SystemExit(1)
