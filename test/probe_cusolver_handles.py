#!/usr/bin/env python3
"""Measure cuSOLVER handle resource growth independently of solver numerics."""
import argparse,ctypes,json,os,time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('directory',type=Path)
p.add_argument('--fraction',type=float,required=True)
p.add_argument('--handles',type=int,default=400)
a=p.parse_args()
rank=int(os.environ['SLURM_PROCID']); peers=int(os.environ['SLURM_NTASKS'])
a.directory.mkdir(parents=True,exist_ok=True)
cuda=ctypes.CDLL('libcudart.so'); solver=ctypes.CDLL('libcusolver.so')
cuda.cudaFree.argtypes=[ctypes.c_void_p]
cuda.cudaMalloc.argtypes=[ctypes.POINTER(ctypes.c_void_p),ctypes.c_size_t]
cuda.cudaMemGetInfo.argtypes=[ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
solver.cusolverDnCreate.argtypes=[ctypes.POINTER(ctypes.c_void_p)]
solver.cusolverDnDestroy.argtypes=[ctypes.c_void_p]
def checked(status):
    if status: raise RuntimeError('CUDA status '+str(status))
def memory():
    free=ctypes.c_size_t(); total=ctypes.c_size_t()
    checked(cuda.cudaMemGetInfo(ctypes.byref(free),ctypes.byref(total)))
    return dict(free_bytes=free.value,total_bytes=total.value)
def snapshot(count,status=0):
    record=dict(rank=rank,handles=count,status=status,**memory())
    proc=Path('/proc/self/status').read_text().splitlines()
    record['host_status']=[x for x in proc if x.startswith(('VmRSS:','VmSize:','Threads:'))]
    print(json.dumps(record),flush=True)
checked(cuda.cudaFree(None))
ready=a.directory/('ready'+str(rank)+'.json')
ready.write_text(json.dumps(memory()))
deadline=time.monotonic()+60
paths=[a.directory/('ready'+str(r)+'.json') for r in range(peers)]
while not all(x.exists() and x.stat().st_size for x in paths):
    if time.monotonic()>deadline: raise RuntimeError('Peer context startup timeout')
    time.sleep(.05)
initial_free=min(json.loads(x.read_text())['free_bytes'] for x in paths)
pool=ctypes.c_void_p()
checked(cuda.cudaMalloc(ctypes.byref(pool),int(initial_free*a.fraction/peers)))
(a.directory/('pool_ready'+str(rank))).touch()
while not all((a.directory/('pool_ready'+str(r))).exists() for r in range(peers)):
    if time.monotonic()>deadline: raise RuntimeError('Peer pool startup timeout')
    time.sleep(.05)
snapshot(0)
handles=[]
try:
    for i in range(a.handles):
        handle=ctypes.c_void_p()
        status=solver.cusolverDnCreate(ctypes.byref(handle))
        if status:
            snapshot(len(handles),status)
            break
        handles.append(handle)
        if len(handles)%10==0: snapshot(len(handles))
    snapshot(len(handles))
finally:
    for handle in reversed(handles): solver.cusolverDnDestroy(handle)
    checked(cuda.cudaFree(pool))
