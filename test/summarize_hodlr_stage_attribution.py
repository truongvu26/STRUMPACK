#!/usr/bin/env python3
"""CUDA-only Nsight summary; attribute GPU activity by its host launch scope."""
from collections import defaultdict
import heapq
import json
from pathlib import Path
import sqlite3
import sys
path,out=map(Path,sys.argv[1:3]);db=sqlite3.connect(str(path));db.row_factory=sqlite3.Row
tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
strings=dict(db.execute('SELECT id,value FROM StringIds'))
def rows(table): return [dict(r) for r in db.execute('SELECT * FROM '+table)] if table in tables else []
def union(intervals):
    total=0;lo=hi=None
    for a,b in sorted(intervals):
        if b<a: raise ValueError('Invalid activity interval')
        if hi is None: lo,hi=a,b
        elif a>hi: total+=hi-lo;lo,hi=a,b
        else: hi=max(hi,b)
    return (total+(hi-lo if hi is not None else 0))/1e9
nvtx=defaultdict(list)
for r in rows('NVTX_EVENTS'):
    label=r.get('text') or strings.get(r.get('textId'),'')
    if not label or r.get('end') is None: continue
    if not label.startswith(('solver.','front.','F11.','F12.','F21.','F22.','prepare.','Schur.','BF.','HODLR.','solve.')): continue
    nvtx[r['globalTid']].append((r['start'],r['end'],label))
runtime=rows('CUPTI_ACTIVITY_KIND_RUNTIME');driver=rows('CUPTI_ACTIVITY_KIND_DRIVER')
threads=defaultdict(list)
for i,r in enumerate(runtime+driver): threads[r['globalTid']].append((r['start'],i,r))
owner={};api=defaultdict(lambda:dict(calls=0,host_seconds=0.));correlation={}
all_api=defaultdict(lambda:dict(calls=0,host_seconds=0.))
for tid,points in threads.items():
    intervals=sorted(nvtx[tid]);active=[];at=0
    roots=[v for v in intervals if v[2].startswith('solver.')]
    for instant,i,r in sorted(points):
        while at<len(intervals) and intervals[at][0]<=instant:
            start,end,name=intervals[at];heapq.heappush(active,(-start,at,end,name));at+=1
        while active and active[0][2]<instant: heapq.heappop(active)
        phase=active[0][3] if active else 'unattributed'
        section=next((name for start,end,name in roots if start<=instant<=end),'outside')
        key=(section,phase);correlation[r['correlationId']]=key
        name=strings.get(r['nameId'],str(r['nameId']));seconds=(r['end']-r['start'])/1e9
        # Runtime API rows are the public host cost; driver rows are used only
        # for launch attribution, avoiding counting both layers as host totals.
        if i<len(runtime):
            x=api[(section,phase,name)];x['calls']+=1;x['host_seconds']+=seconds
            x=all_api[name];x['calls']+=1;x['host_seconds']+=seconds
stats=defaultdict(lambda:dict(kernels=0,kernel_sum_seconds=0.,copy_sum_seconds=0.,copy_bytes=0,htod_bytes=0,dtoh_bytes=0))
kernels=rows('CUPTI_ACTIVITY_KIND_KERNEL');copies=rows('CUPTI_ACTIVITY_KIND_MEMCPY');memsets=rows('CUPTI_ACTIVITY_KIND_MEMSET')
intervals=[];section_kernels=defaultdict(list);section_activity=defaultdict(list)
top=defaultdict(lambda:dict(calls=0,seconds=0.));copy_totals=defaultdict(lambda:dict(calls=0,bytes=0,seconds=0.))
copy_names=dict(db.execute('SELECT id,label FROM ENUM_CUDA_MEMCPY_OPER')) if 'ENUM_CUDA_MEMCPY_OPER' in tables else {}
for r in kernels:
    key=correlation.get(r.get('correlationId'),('outside','unattributed'));x=stats[key];x['kernels']+=1;x['kernel_sum_seconds']+=(r['end']-r['start'])/1e9
    intervals.append((r['start'],r['end']));name=strings.get(r.get('shortName',r.get('demangledName')),'unknown')
    section_kernels[key[0]].append((r['start'],r['end']));section_activity[key[0]].append((r['start'],r['end']))
    top[name]['calls']+=1;top[name]['seconds']+=(r['end']-r['start'])/1e9
for r in copies:
    key=correlation.get(r.get('correlationId'),('outside','unattributed'));x=stats[key];seconds=(r['end']-r['start'])/1e9
    x['copy_sum_seconds']+=seconds;x['copy_bytes']+=r['bytes'];direction=copy_names.get(r['copyKind'],str(r['copyKind']))
    if 'HtoD' in direction or 'Host-to-Device' in direction: x['htod_bytes']+=r['bytes']
    if 'DtoH' in direction or 'Device-to-Host' in direction: x['dtoh_bytes']+=r['bytes']
    y=copy_totals[direction];y['calls']+=1;y['bytes']+=r['bytes'];y['seconds']+=seconds
    intervals.append((r['start'],r['end']))
    section_activity[key[0]].append((r['start'],r['end']))
for r in memsets:
    key=correlation.get(r.get('correlationId'),('outside','unattributed'))
    section_activity[key[0]].append((r['start'],r['end']))
intervals.extend((r['start'],r['end']) for r in memsets)
result=dict(rank=int(path.stem.replace('rank','')),kernel_count=len(kernels),nvtx_range_count=sum(map(len,nvtx.values())),gpu_active_union_seconds=union(intervals),
    kernel_sum_seconds=sum(x['kernel_sum_seconds'] for x in stats.values()),
    section_activity={section:dict(gpu_kernel_union_seconds=union(section_kernels[section]),
        gpu_activity_union_seconds=union(spans)) for section,spans in section_activity.items()},
    phases=[dict(section=k[0],phase=k[1],**v) for k,v in sorted(stats.items(),key=lambda kv:kv[1]['kernel_sum_seconds'],reverse=True)],
    cuda_runtime_api=[dict(name=k,**v) for k,v in sorted(all_api.items(),key=lambda kv:kv[1]['host_seconds'],reverse=True)[:30]],
    runtime_by_phase=[dict(section=k[0],phase=k[1],name=k[2],**v) for k,v in sorted(api.items(),key=lambda kv:kv[1]['host_seconds'],reverse=True)[:100]],
    memory_copies=[dict(direction=k,**v) for k,v in copy_totals.items()],top_kernels=[dict(name=k,**v) for k,v in sorted(top.items(),key=lambda kv:kv[1]['seconds'],reverse=True)[:20]],
    notes=['Per-process trace, not whole-device occupancy; GPU work from other processes is not measured.',
           'GPU activity attributed by correlated host API launch inside the deepest diagnostic NVTX range.',
           'Kernel/copy duration sums may overlap; union is calculated separately.',
           'MPI tracing is disabled; CPU stage scopes include MPI waiting and host staging.'])
out.write_text(json.dumps(result,indent=2)+'\n');db.close()
print('CUDA summary',result['rank'],result['kernel_count'],'kernels',result['nvtx_range_count'],'ranges')
