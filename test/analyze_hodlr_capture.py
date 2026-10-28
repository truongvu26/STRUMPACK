#!/usr/bin/env python3
"""Compare captured callback, compressed matrix and prepared evaluator image."""
import json,sys
from pathlib import Path
import numpy as np

p=Path(sys.argv[1])
with p.open('rb') as f:
    header=np.fromfile(f,dtype='<u8',count=5)
    if len(header)!=5 or int(header[0])!=0x31504d5544525453:
        raise ValueError('Not a STRUMPACK construction capture')
    m,n,size,flags=map(int,header[1:])
    dtype=np.dtype(('<c' if flags&1 else '<f')+str(size))
    if m<=0 or n<=0 or m*n*size>64*1024**2: raise ValueError('Invalid capture dimensions')
    matrices={}
    for name in ['reference','compressed']+(['prepared'] if flags&2 else []):
        a=np.fromfile(f,dtype=dtype,count=m*n)
        if a.size!=m*n: raise ValueError('Truncated '+name+' image')
        matrices[name]=a.reshape((m,n),order='F')
    if f.read(1): raise ValueError('Unexpected trailing capture data')
def relative(a,b):
    norm=np.linalg.norm(b)
    return float(np.linalg.norm(a-b)/(norm if norm else 1))
def scalar(x):
    return [float(x.real),float(x.imag)] if np.iscomplexobj(x) else float(x)
result=dict(file=str(p),rows=m,cols=n,dtype=str(dtype),comparisons={})
for name,a in matrices.items():
    if m==n: result[name+'_transpose_asymmetry']=relative(a,a.T)
for left,right in [('compressed','reference'),('prepared','reference'),('compressed','prepared')]:
    if left not in matrices or right not in matrices: continue
    a,b=matrices[left],matrices[right];diff=np.abs(a-b)
    order=np.argsort(diff.ravel())[-12:][::-1]
    rec=dict(relative_frobenius_error=relative(a,b),maximum_entry_error=float(diff.max()),
        largest_entries=[dict(row=int(at//n),col=int(at%n),left=scalar(a.flat[at]),right=scalar(b.flat[at])) for at in order],
        largest_row_errors=[int(x) for x in np.argsort(np.sum(diff**2,axis=1))[-12:][::-1]])
    if m==n:rec['relative_diagonal_error']=relative(np.diag(a),np.diag(b))
    result['comparisons'][left+'_vs_'+right]=rec
out=p.with_suffix('.json');out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
