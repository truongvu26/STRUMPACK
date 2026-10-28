#!/usr/bin/env python3
"""Attribute Nsight running-CPU samples; keep MPI within ScaLAPACK separate."""
import heapq,itertools,json,re,sqlite3,sys
from collections import Counter,defaultdict
from pathlib import Path

path,out=map(Path,sys.argv[1:3])
db=sqlite3.connect(str(path)); db.row_factory=sqlite3.Row
tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
if not {'COMPOSITE_EVENTS','SAMPLING_CALLCHAINS','NVTX_EVENTS'}<=tables:
    raise SystemExit('Missing CPU call stacks or NVTX scopes in Nsight export')
strings=dict(db.execute('SELECT id,value FROM StringIds'))
nvtx=defaultdict(list); roots=defaultdict(list)
for row in db.execute('SELECT * FROM NVTX_EVENTS'):
    label=row['text'] or strings.get(row['textId'],'')
    if row['end'] is None or not label: continue
    if not label.startswith(('solver.','front.','F11.','F12.','F21.','F22.','prepare.','Schur.','BF.','HODLR.','solve.')): continue
    interval=(row['start'],row['end'],label)
    nvtx[row['globalTid']].append(interval)
    if label.startswith('solver.'): roots[row['globalTid']].append(interval)
events=defaultdict(list)
for row in db.execute('SELECT * FROM COMPOSITE_EVENTS WHERE threadState=1'):
    events[row['globalTid']].append((row['start'],row['id']))
owners={}
for tid,samples in events.items():
    intervals=sorted(nvtx[tid]); at=0; active=[]
    for instant,sample in sorted(samples):
        while at<len(intervals) and intervals[at][0]<=instant:
            start,end,label=intervals[at]
            heapq.heappush(active,(-start,at,end,label)); at+=1
        while active and active[0][2]<instant: heapq.heappop(active)
        section=next((label for start,end,label in roots[tid] if start<=instant<=end),'outside')
        owners[sample]=(section,active[0][3] if active else 'unattributed')

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
query='SELECT * FROM SAMPLING_CALLCHAINS ORDER BY id,stackDepth'
for sample,rows in itertools.groupby(db.execute(query),key=lambda r:r['id']):
    if sample not in owners: continue
    rows=list(rows)
    frames=[(strings.get(r['symbol'],'unknown'),strings.get(r['module'],'unknown')) for r in rows]
    entry_seen=any(name=='main' or name.startswith('__libc_start_main') or name=='_start' for name,_ in frames)
    section,phase=owners[sample]; key=(section,phase); kind=category(frames,entry_seen)
    counts[key][kind]+=1; sampled[section]+=1
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
        categories=[dict(category=k,samples=v,fraction=v/total) for k,v in categories.most_common()],
        top_leaf_functions=[dict(name=k[0],module=k[1],samples=v,fraction=v/total) for k,v in leaf[section].most_common(30)],
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
result=dict(rank=int(path.stem.replace('rank','')),sections=sections,
    main_thread_scheduled_cpu_seconds=dict(scheduled),
    diagnostics=[dict(row) for row in db.execute('SELECT severity,text FROM DIAGNOSTIC_EVENT')] if 'DIAGNOSTIC_EVENT' in tables else [],
    phases=[dict(section=k[0],phase=k[1],running_samples=sum(v.values()),categories=dict(v)) for k,v in counts.items()],
    notes=['Running-CPU sample fractions are not fractions of solver wall time.',
      'ScaLAPACK attribution is inclusive of a libscalapack stack frame; MPI/transport below it is separated.',
      'When a stack does not reach a known program entry, an unobserved ScaLAPACK caller is not ruled out.',
      'BLACS and PBLAS share libscalapack and are included in its category.',
      'Sampled call stacks and scheduled-CPU intervals cover selected processes; untraced peers share their GPUs.',
      'Inclusive function counts overlap. Host work can overlap CUDA activity and must not be added to GPU durations.'])
out.write_text(json.dumps(result,indent=2)+'\n'); db.close()
print('CPU summary',result['rank'],sampled['solver.factor'],'factor backtraces')
