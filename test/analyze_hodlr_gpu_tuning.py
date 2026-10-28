#!/usr/bin/env python3
"""Prepare controlled GPU HODLR candidates and rank only accurate sparse solves."""
import argparse,hashlib,json,math,os,re,shlex,statistics
from collections import Counter,defaultdict
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('directory',type=Path)
p.add_argument('--prepare',type=Path);p.add_argument('--final',action='store_true')
p.add_argument('--require-complete',action='store_true');a=p.parse_args();root=a.directory
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
if a.prepare:
    snapshot=Path(os.environ['HODLR_PROFILE_SNAPSHOT']).resolve(strict=True)
    library=os.environ.get('HODLR_BLAS_LIBRARY','');blas=Path(library).resolve(strict=True) if library else None
    ranks=int(os.environ['HODLR_PROFILE_RANKS']);sharing=int(os.environ['HODLR_RANKS_PER_GPU'])
    workers=int(os.environ['OPENBLAS_NUM_THREADS']);grid=int(os.environ['HODLR_TUNE_GRID'])
    plan=json.loads(a.prepare.read_text());assert isinstance(plan,list) and plan,'Plan must be a nonempty JSON array'
    defaults=dict(gpu_mode=1,leaf=256,tol=1e-5,min_sep=5000,min_front=100000,lr_leaf=5,
        knn_hodlrbf=64,knn_lrbf=128,baca_batch=16,image_cap=1073741824,panel_cap=67108864,
        blas_threads=workers,heap_fraction=.85,maxit=30,timeout_seconds=300,extra_args=[],
        gpu_rectangular=1,gpu_host_panels=1,solver_tol=1e-10,baca_residual_guard=0)
    flags={'gpu_mode':'hodlr_use_gpu','leaf':'hodlr_leaf_size','tol':'hodlr_rel_tol',
        'min_sep':'sp_compression_min_sep_size','min_front':'sp_compression_min_front_size',
        'gpu_rectangular':'hodlr_gpu_rectangular','gpu_host_panels':'hodlr_gpu_host_panels',
        'lr_leaf':'hodlr_lr_leaf','knn_hodlrbf':'hodlr_knn_hodlrbf','knn_lrbf':'hodlr_knn_lrbf',
        'baca_batch':'hodlr_BACA_block_size','image_cap':'hodlr_gpu_front_max_bytes',
        'panel_cap':'hodlr_gpu_panel_max_bytes','maxit':'sp_maxit','solver_tol':'sp_rel_tol'}
    locked=set(flags.values())|{'sp_compression','hodlr_butterfly_levels','hodlr_gpu_rectangular',
        'hodlr_gpu_host_panels','sp_rel_tol','sp_Krylov_solver'}
    normalized=[];labels=set()
    for index,entry in enumerate(plan):
        assert isinstance(entry,dict),'Each candidate must be an object'
        assert set(entry)<=set(defaults)|{'label','knn'},'Unknown candidate fields: '+str(set(entry)-set(defaults)-{'label','knn'})
        c=dict(defaults);c.update(entry)
        if 'knn' in c:c['knn_hodlrbf']=c['knn_lrbf']=c.pop('knn')
        label=c.pop('label','case-%02d'%index)
        assert re.fullmatch(r'[A-Za-z0-9_-]+',label) and label not in labels,'Invalid or duplicate label'
        labels.add(label)
        for key in ('gpu_mode','leaf','min_sep','min_front','lr_leaf','knn_hodlrbf','knn_lrbf','baca_batch','image_cap','panel_cap','blas_threads','maxit','timeout_seconds','gpu_rectangular','gpu_host_panels','baca_residual_guard'):
            assert isinstance(c[key],int) and not isinstance(c[key],bool) and c[key]>=0,'Invalid '+key
        assert c['gpu_rectangular'] in (0,1) and c['gpu_host_panels'] in (0,1) and c['baca_residual_guard'] in (0,1)
        assert c['gpu_mode'] in (1,2) and c['lr_leaf'] in (4,5),'GPU HODLR requires mode1/2 and lr_leaf4/5'
        assert min(c[k] for k in ('leaf','baca_batch','panel_cap','blas_threads','maxit','timeout_seconds'))>0
        assert c['blas_threads']<=int(os.environ.get('HODLR_TUNE_CPUS_PER_RANK',os.environ.get('SLURM_CPUS_PER_TASK','16')))
        assert blas or c['blas_threads']==1,'Threaded BLAS requires an explicitly selected pthread library'
        assert math.isfinite(c['tol']) and 0<c['tol']<1 and 0<c['heap_fraction']<1
        assert math.isfinite(c['solver_tol']) and 0<c['solver_tol']<=1e-10, 'Final solver tolerance may only be tightened'
        assert isinstance(c['extra_args'],list) and all(isinstance(s,str) for s in c['extra_args'])
        assert not any(s.lstrip('-').split('=')[0] in locked for s in c['extra_args'] if s.startswith('-')),'extra_args cannot override recorded/core solver settings'
        if c['gpu_host_panels']:
            assert not any(s.lstrip('-').split('=')[0]=='hodlr_verbose' for s in c['extra_args']), 'GPU host panels require HODLR diagnostics disabled'
        argv=[str(snapshot/'panels'),str(grid),'3','--sp_compression','HODLR',
            '--sp_Krylov_solver','refinement','--hodlr_butterfly_levels','0',
            '--hodlr_quiet']
        for key,flag in flags.items():argv+=['--'+flag,str(c[key])]
        argv+=c['extra_args'];directory=root/label;directory.mkdir()
        write(directory/'config.json',c);write(directory/'argv.json',argv)
        (directory/'panels.command').write_text(' '.join(shlex.quote(s) for s in argv)+'\n')
        normalized.append(dict(label=label,settings=c,relaxed_compression=c['tol']>1e-5))
    manifest=dict(snapshot=str(snapshot),analyzer_sha256=sha(Path(__file__)),
        build_settings=json.loads((snapshot/'build_settings.json').read_text()) if (snapshot/'build_settings.json').exists() else None,
        executable_sha256=sha(snapshot/'panels'),
        native_library_sha256=sha(snapshot/'lib/libbutterflypack.so.5.0.0'),blas_library=str(blas) if blas else None,
        blas_library_sha256=sha(blas) if blas else None,snapshot_manifest=(snapshot/'manifest.txt').read_text(),
        grid=grid,rhs=3,ranks=ranks,nodes=int(os.environ.get('HODLR_TUNE_NODES',os.environ['SLURM_JOB_NUM_NODES'])),ranks_per_gpu=sharing,
        slurm_job_id=os.environ.get('SLURM_JOB_ID'),allocation_nodes=int(os.environ['SLURM_JOB_NUM_NODES']),
        selected_nodes=os.environ.get('HODLR_TUNE_NODELIST'),
        cpus_per_rank=int(os.environ.get('HODLR_TUNE_CPUS_PER_RANK',os.environ.get('SLURM_CPUS_PER_TASK','16'))),
        omp_threads=1,accuracy=dict(relative_error=1e-8,scaled_residual=1e-6),plan=normalized,
        objective='Validated factor-plus-solve wall time; factor time secondary. Compression tolerance changes are explicit.')
    write(root/'run_manifest.json',manifest);(root/'case_order.txt').write_text('\n'.join(c['label'] for c in normalized)+'\n')
    raise SystemExit(0)

manifest=json.loads((root/'run_manifest.json').read_text());ranks=manifest['ranks'];sharing=manifest['ranks_per_gpu']
result=dict(status='RUNNING',job=root.name,analysis_script_sha256=sha(Path(__file__)),manifest=manifest,cases=[],rankings=[],notes=[
    'Only numerically passing GPU HODLR candidates enter rankings; failed and skipped candidates remain visible.',
    'Changing compression tolerance is reported explicitly; final solution/residual bounds are unchanged.',
    'Best tested settings are not a guarantee of a global optimum. Randomized compression and single trials can vary.'])
groups=defaultdict(list);bindings0=None;affinity0=None
def number(text,pattern):
    values=re.findall(pattern,text,re.M)
    if len(values)!=1:raise ValueError('Missing or duplicate measurement: '+pattern)
    v=float(values[0].replace(',',''))
    if not math.isfinite(v):raise ValueError('Nonfinite measurement')
    return v
def optional_timings(text,time_file):
    """Add available setup/wall timings without turning absent diagnostics into failures."""
    values={}
    for key,label in (('nd_seconds','nd time'),('symbolic_seconds','symb-factor time'),
                      ('sep_reorder_seconds','sep-reorder time')):
        matches=re.findall(r'^#\s+- '+re.escape(label)+r' = (\S+)',text,re.M)
        if len(matches)!=1:continue
        try:v=float(matches[0])
        except ValueError:continue
        if math.isfinite(v) and v>=0:values[key]=v
    for key,field in (('host_panel_prepare_seconds_rank0','host_prepare_seconds_local'),
                      ('host_panel_metadata_seconds_rank0','metadata_seconds_local'),
                      ('host_panel_upload_seconds_rank0','upload_seconds_local'),
                      ('host_panel_upload_bytes_rank0','upload_bytes_local')):
        matches=re.findall(r'^# GPU host panels .*?'+field+r'=(\S+)',text,re.M)
        if matches:
            try:
                numbers=[float(v) for v in matches]
                if all(math.isfinite(v) and v>=0 for v in numbers):values[key]=sum(numbers)
            except ValueError:pass
    matches=re.findall(r'^# GPU front evaluator .*?host_prepare_seconds=(\S+)',text,re.M)
    if matches:
        try:
            numbers=[float(v) for v in matches]
            if all(math.isfinite(v) and v>=0 for v in numbers):values['prepared_image_seconds_rank0']=sum(numbers)
        except ValueError:pass
    try:timing=time_file.read_text()
    except OSError:return values
    matches=re.findall(r'^\s*Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)\s*$',timing,re.M)
    if len(matches)!=1:return values
    parts=matches[0].split(':')
    if len(parts) not in (2,3):return values
    try:
        seconds=float(parts[-1]);minutes=int(parts[-2]);hours=int(parts[0]) if len(parts)==3 else 0
        if not (math.isfinite(seconds) and 0<=seconds<60 and minutes>=0 and hours>=0):return values
        if len(parts)==3 and minutes>=60:return values
        values['end_to_end_seconds']=hours*3600+minutes*60+seconds
    except ValueError:pass
    return values
for planned in manifest['plan']:
    directory=root/planned['label'];c=dict(planned,status='PLANNED');result['cases'].append(c)
    if (directory/'skipped_budget.txt').exists():c.update(status='SKIPPED',reason=(directory/'skipped_budget.txt').read_text().strip());continue
    status=directory/'panels.exit_code'
    if not status.exists():continue
    try:
        c['exit_code']=int(status.read_text())
        if c['exit_code']:raise ValueError('Solver/launcher exit '+str(c['exit_code']))
        logs=[(directory/('panels.rank%d.log'%r)).read_text() for r in range(ranks)];text=logs[0]
        patterns=dict(factor_seconds=r'^#\s+- factor time = (\S+)',solve_seconds=r'^#\s+- solve time = (\S+)',
            relative_error=r'^# relative error = .* = (\S+)',scaled_residual=r'^# COMPONENTWISE SCALED RESIDUAL = (\S+)',
            hodlr_fronts=r'nr of HODLR Frontal matrices = ([\d,]+)',refinement_iterations=r'number of Krylov iterations = (\d+)')
        c.update((key,number(text,pattern)) for key,pattern in patterns.items())
        dense=[]
        for source_rank,log in enumerate(logs):
            for procs,seconds,sep,upd in re.findall(r'^# DenseMPI factorization complete, .*?P=(\d+), T=\d+: (\S+) seconds,.*?ds=(\d+), du=(\d+)',log,re.M):
                if math.isfinite(float(seconds)) and float(seconds)>=0:
                    dense.append(dict(seconds=float(seconds),front_ranks=int(procs),separator_rows=int(sep),update_rows=int(upd),source_rank=source_rank))
        if dense:
            c['slowest_dense_mpi_front']=max(dense,key=lambda v:v['seconds'])
            c['max_cpu_dense_front_seconds']=c['slowest_dense_mpi_front']['seconds']
        c['factor_plus_solve_seconds']=c['factor_seconds']+c['solve_seconds']
        c.update(optional_timings(text,directory/'panels.time'))
        if not (c['factor_seconds']>0 and c['solve_seconds']>=0 and c['hodlr_fronts']>0 and
            0<=c['relative_error']<manifest['accuracy']['relative_error'] and
            0<=c['scaled_residual']<manifest['accuracy']['scaled_residual'] and
            c['refinement_iterations']<=planned['settings']['maxit']):raise ValueError('Accuracy, iteration or HODLR execution check failed')
        if any(re.search(r'On entry to .*parameter.*illegal value|xxmr2d:out of memory', s, re.I) for s in logs):raise ValueError('LAPACK/ScaLAPACK runtime error reported despite completed solve')
        if not any(re.search(r'^# (GPU front evaluator|GPU host panels|Native GPU|HODLR backend=GPU mode=[12].*operation=(factor|solve))',s,re.M) for s in logs):raise ValueError('Missing GPU HODLR execution evidence')
        bindings=dict(Counter(v for s in logs for v in re.findall(r'# stage binding .* uuid=(\S+)',s)))
        if sum(bindings.values())!=ranks or len(bindings)!=ranks//sharing or set(bindings.values())!={sharing}:raise ValueError('GPU placement/sharing mismatch')
        if bindings0 is None:bindings0=bindings
        if bindings!=bindings0:raise ValueError('GPU placement changed')
        expected=[manifest['executable_sha256'],manifest['native_library_sha256']]
        if manifest['blas_library_sha256']:expected.append(manifest['blas_library_sha256'])
        if [s.split()[0] for s in (directory/'binary_hashes.txt').read_text().splitlines()]!=expected:raise ValueError('Binary hashes differ')
        affinity=[];used=defaultdict(set)
        if manifest['blas_library_sha256']:
            for rank in range(ranks):
                probe=json.loads((directory/('panels.blas.rank%d.json'%rank)).read_text());workers=planned['settings']['blas_threads']
                if not (probe['status']=='PASS' and probe['rank']==rank and probe['library_sha256']==manifest['blas_library_sha256'] and
                    probe['reported_threads']==workers and probe['omp_max_threads']==1 and probe['parallel_kind']==1 and
                    len(probe['loader_openblas_dependencies'])==1 and probe['physical_core_count']>=workers):raise ValueError('BLAS runtime/CPU placement verification failed')
                cores={tuple(x) for x in probe['physical_core_ids']}
                if used[probe['node']]&cores:raise ValueError('Physical CPU cores overlap')
                used[probe['node']].update(cores);affinity.append((probe['node'],probe['affinity_cpus']))
            if affinity0 is None:affinity0=affinity
            if affinity!=affinity0:raise ValueError('CPU affinity changed')
        matches=re.findall(r'^#\s+- maximum HODLR rank = (\d+)',text,re.M)
        if len(matches)==1:c['maximum_hodlr_rank']=int(matches[0])
        c.update(status='PASS',gpu_uuid_rank_counts=bindings)
        config={k:v for k,v in planned['settings'].items() if k!='timeout_seconds'}
        groups[json.dumps(config,sort_keys=True)].append(c)
    except (OSError,ValueError,KeyError,TypeError,IndexError) as e:c.update(status='FAIL',reason=str(e))
for config,cases in groups.items():
    result['rankings'].append(dict(settings=json.loads(config),trials=len(cases),labels=[c['label'] for c in cases],
        relaxed_compression=cases[0]['relaxed_compression'],**{'median_'+key:statistics.median(c[key] for c in cases)
        for key in ('factor_seconds','solve_seconds','factor_plus_solve_seconds')}))
    for key in ('nd_seconds','symbolic_seconds','sep_reorder_seconds','end_to_end_seconds',
                'host_panel_prepare_seconds_rank0','host_panel_metadata_seconds_rank0','host_panel_upload_seconds_rank0',
                'host_panel_upload_bytes_rank0','prepared_image_seconds_rank0','max_cpu_dense_front_seconds'):
        if all(key in c for c in cases):result['rankings'][-1]['median_'+key]=statistics.median(c[key] for c in cases)
result['rankings'].sort(key=lambda c:(c['median_factor_plus_solve_seconds'],c['median_factor_seconds']))
result['all_candidates_tested']=all(c['status'] in ('PASS','FAIL') for c in result['cases'])
result['finished_plan']=all(c['status'] in ('PASS','FAIL','SKIPPED') for c in result['cases'])
if result['finished_plan']:
    result['status']=('PASS_WITH_REJECTED_CANDIDATES' if any(c['status']=='FAIL' for c in result['cases']) else 'PASS') if result['all_candidates_tested'] else 'PARTIAL'
    if not result['rankings']:result['status']='FAIL'
write(root/'tuning_summary.json',result)
print(result['status'],[(c['label'],c['status'],c.get('factor_plus_solve_seconds'),c.get('reason')) for c in result['cases']],flush=True)
if a.require_complete and not result['all_candidates_tested']:raise SystemExit(1)
if a.final and not result['rankings']:raise SystemExit(1)
