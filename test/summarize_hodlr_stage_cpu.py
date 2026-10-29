#!/usr/bin/env python3
"""Attribute running-CPU stacks by thread scopes or same-process solver windows."""
import heapq,itertools,json,re,sqlite3,sys
from collections import Counter,defaultdict
from pathlib import Path

path,out=map(Path,sys.argv[1:3])
db=sqlite3.connect(str(path)); db.row_factory=sqlite3.Row
tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
if not {'COMPOSITE_EVENTS','SAMPLING_CALLCHAINS','NVTX_EVENTS'}<=tables:
    raise SystemExit('Missing CPU call stacks or NVTX scopes in Nsight export')
strings=dict(db.execute('SELECT id,value FROM StringIds'))
# Nsight globalTid = globalPid | OS_tid, with the low 24 bits reserved for
# the thread id. Require a matching PROCESSES row instead of treating an
# arbitrary timestamp match (or an unvalidated bit shift) as process identity.
tid_mask=(1<<24)-1
processes={}
if 'PROCESSES' in tables:
    for row in db.execute('SELECT globalPid,pid,name FROM PROCESSES'):
        gpid,pid=row['globalPid'],row['pid']
        if gpid is not None and pid is not None and not gpid&tid_mask and 0<pid<=tid_mask:
            if (gpid>>24)&tid_mask==pid:
                processes[gpid]=dict(pid=pid,name=row['name'],main_tid=gpid|pid)
def process(tid):
    if not isinstance(tid,int): return None
    gpid=tid&~tid_mask
    return gpid if gpid in processes else None
nvtx=defaultdict(list); roots=defaultdict(list)
for row in db.execute('SELECT * FROM NVTX_EVENTS'):
    label=row['text'] or strings.get(row['textId'],'')
    if row['end'] is None or not label: continue
    if not label.startswith(('solver.','front.','F11.','F12.','F21.','F22.','prepare.','Schur.','BF.','HODLR.','solve.')): continue
    interval=(row['start'],row['end'],label)
    nvtx[row['globalTid']].append(interval)
    if label.startswith('solver.'): roots[row['globalTid']].append(interval)
process_roots=defaultdict(list)
for tid,intervals in roots.items():
    gpid=process(tid)
    if gpid is not None and tid==processes[gpid]['main_tid']:
        process_roots[gpid].extend(intervals)
events=defaultdict(list)
for row in db.execute('SELECT * FROM COMPOSITE_EVENTS WHERE threadState=1'):
    events[row['globalTid']].append((row['start'],row['id']))
owners={}
for tid,samples in events.items():
    intervals=sorted(nvtx[tid]); at=0; active=[]
    gpid=process(tid)
    role=('unknown_process' if gpid is None else
          'main_thread' if tid==processes[gpid]['main_tid'] else 'worker_thread')
    for instant,sample in sorted(samples):
        while at<len(intervals) and intervals[at][0]<=instant:
            start,end,label=intervals[at]
            heapq.heappush(active,(-start,at,end,label)); at+=1
        while active and active[0][2]<=instant: heapq.heappop(active)
        own=[r for r in roots[tid] if r[0]<=instant<r[1]]
        section=max(own,key=lambda r:r[0])[2] if own else 'outside'
        method='same_thread_solver_scope' if own else 'outside_solver_windows'
        if not own and gpid is not None:
            windows=[r for r in process_roots.get(gpid,()) if r[0]<=instant<r[1]]
            if len(windows)==1:
                section=windows[0][2]
                method='same_process_solver_window'
            elif len(windows)>1:
                method='ambiguous_process_solver_windows'
        phase=active[0][3] if active else ('process_window_only' if method=='same_process_solver_window' else 'unattributed')
        owners[sample]=(section,phase,method,role,gpid,tid)

def category(frames,entry_seen):
    modules=' '.join(module.lower() for _,module in frames)
    symbols=' '.join(name for name,_ in frames)
    scalapack='libscalapack' in modules
    mpi=any(x in modules for x in ('libmpi','libfabric','libcxi','libxpmem')) or bool(re.search(r'\bP?MPI_',symbols))
    if scalapack and mpi: return 'MPI/transport under ScaLAPACK or BLACS'
    if mpi: return 'MPI/transport outside ScaLAPACK' if entry_seen else 'MPI/transport (caller incomplete)'
    if scalapack: return 'ScaLAPACK/PBLAS/BLACS outside MPI'
    if any(x in modules for x in ('libcuda','libcublas','libcusolver','libnvrtc')): return 'CUDA/library host work'
    if 'libopenblas' in modules or 'liblapack' in modules:
        return 'Host BLAS/LAPACK outside ScaLAPACK' if entry_seen else 'Host BLAS/LAPACK (caller incomplete)'
    if 'libbutterflypack' in modules: return 'ButterflyPACK host work outside the above'
    return 'STRUMPACK/other host work or unresolved'

counts=defaultdict(Counter); leaf=defaultdict(Counter); inclusive=defaultdict(Counter)
unresolved=Counter(); sampled=Counter(); entry_counts=Counter()
methods=defaultdict(Counter); roles=defaultdict(Counter)
caller_coverage=defaultdict(Counter); sampled_threads=Counter()
query='SELECT * FROM SAMPLING_CALLCHAINS ORDER BY id,stackDepth'
for sample,rows in itertools.groupby(db.execute(query),key=lambda r:r['id']):
    if sample not in owners: continue
    rows=list(rows)
    frames=[(strings.get(r['symbol'],'unknown'),strings.get(r['module'],'unknown')) for r in rows]
    entry_seen=any(name=='main' or name.startswith('__libc_start_main') or name=='_start' for name,_ in frames)
    thread_entry_seen=any(name in ('start_thread','__clone','clone','thread_start') or
        name.startswith(('__pthread_start','pthread_start')) for name,_ in frames)
    section,phase,method,role,gpid,tid=owners[sample]; key=(section,phase); kind=category(frames,entry_seen)
    counts[key][kind]+=1; sampled[section]+=1
    methods[section][method]+=1; roles[section][role]+=1; sampled_threads[(gpid,tid,role)]+=1
    caller_coverage[section][role+'_samples']+=1
    if entry_seen: caller_coverage[section][role+'_program_entry_samples']+=1
    if thread_entry_seen: caller_coverage[section][role+'_thread_entry_samples']+=1
    if entry_seen: entry_counts[section]+=1
    if rows[0]['unresolved']: unresolved[section]+=1
    name,module=frames[0]; leaf[section][(name,module)]+=1
    for name,module in set(frames): inclusive[section][(name,module)]+=1
if not sampled['solver.factor']: raise SystemExit('No running-CPU backtraces inside solver.factor')
sections=[]
for section,total in sorted(sampled.items()):
    categories=Counter()
    for (scope,phase),values in counts.items():
        if scope==section: categories.update(values)
    sections.append(dict(section=section,running_samples=total,unresolved_leaf_samples=unresolved[section],
        entry_frame_samples=entry_counts[section],entry_frame_fraction=entry_counts[section]/total,
        attribution_methods=dict(methods[section]),thread_roles=dict(roles[section]),
        caller_coverage=dict(caller_coverage[section]),
        categories=[dict(category=k,samples=v,fraction=v/total) for k,v in categories.most_common()],
        top_leaf_functions=[dict(name=k[0],module=k[1],samples=v,fraction=v/total) for k,v in leaf[section].most_common(30)],
        top_scalapack_functions=[dict(name=k[0],module=k[1],samples=v,fraction=v/total) for k,v in inclusive[section].most_common() if 'libscalapack' in k[1].lower()][:20],
        top_inclusive_functions=[dict(name=k[0],module=k[1],samples=v,fraction=v/total) for k,v in inclusive[section].most_common(40)]))
scheduled=defaultdict(float)
if 'SCHED_EVENTS' in tables:
    running={}
    for row in db.execute('SELECT * FROM SCHED_EVENTS ORDER BY start'):
        tid=row['globalTid']; instant=row['start']
        if tid not in roots: continue
        if row['isSchedIn']: running[tid]=instant
        elif tid in running:
            begin=running.pop(tid)
            for start,end,label in roots[tid]:
                scheduled[label]+=max(0,min(end,instant)-max(start,begin))/1e9
scope_wall=defaultdict(float)
for intervals in roots.values():
    for start,end,label in intervals: scope_wall[label]+=(end-start)/1e9
result=dict(rank=int(path.stem.replace('rank','')),sections=sections,scope_wall_seconds=dict(scope_wall),
    attribution_version=2,
    process_window_attribution_enabled=bool(process_roots),
    running_cpu_events=len(owners),running_cpu_events_with_backtraces=sum(sampled.values()),
    running_cpu_events_without_backtraces=len(owners)-sum(sampled.values()),
    process_window_sources=[dict(globalPid=gpid,**processes[gpid],
        solver_windows=[dict(start_ns=start,end_ns=end,section=label) for start,end,label in intervals])
        for gpid,intervals in sorted(process_roots.items())],
    sampled_threads=[dict(globalPid=gpid,globalTid=tid,role=role,running_samples=n)
        for (gpid,tid,role),n in sorted(sampled_threads.items(),key=lambda v:v[0][1])],
    main_thread_scheduled_cpu_seconds=dict(scheduled),
    diagnostics=[dict(row) for row in db.execute('SELECT severity,text FROM DIAGNOSTIC_EVENT')] if 'DIAGNOSTIC_EVENT' in tables else [],
    phases=[dict(section=k[0],phase=k[1],running_samples=sum(v.values()),categories=dict(v)) for k,v in counts.items()],
    notes=['Running-CPU sample fractions are not fractions of solver wall time.',
      'Same-thread solver scopes and nested phases take precedence. Intervals are half-open [start,end).',
      'Workers without a solver scope inherit only an unambiguous main-thread solver window in the same validated process; other processes and unknown process identities are not assigned by time alone.',
      'Same-process window attribution means the CPU sample occurred during that solver call, not that the solver caused the work. Concurrent background threads in that process may also be counted; attribution_methods and thread_roles expose this temporal coverage.',
      'A worker inherits no main-thread nested phase or missing caller. process_window_only is a temporal solver window, not a causal phase attribution.',
      'entry_frame_fraction measures stacks reaching program entry, so worker stacks normally reduce it. caller_coverage separates main/worker program-entry and thread-entry coverage; reaching pthread start does not recover the submitting BLAS/ScaLAPACK caller.',
      'ScaLAPACK attribution is inclusive of a libscalapack stack frame; MPI/transport below it is separated.',
      'When a stack does not reach a known program entry, an unobserved ScaLAPACK caller is not ruled out.',
      'BLACS and PBLAS share libscalapack and are included in its category.',
      'Sampled call stacks and scheduled-CPU intervals cover selected processes; GPU sharing is recorded in the job layout.',
      'Inclusive function counts overlap. Host work can overlap CUDA activity and must not be added to GPU durations.'])
out.write_text(json.dumps(result,indent=2)+'\n'); db.close()
print('CPU summary',result['rank'],sampled['solver.factor'],'factor backtraces')
