#!/usr/bin/env python3
"""Record a rank's inherited CPU placement and selected pthread BLAS before exec."""
import argparse,ctypes,hashlib,json,os,subprocess
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--library',type=Path,required=True)
p.add_argument('--executable',type=Path,required=True)
p.add_argument('--threads',type=int,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
library=a.library.resolve(strict=True)
if not a.library.is_absolute() or a.threads<1: raise SystemExit('Expected absolute BLAS path and positive threads')
affinity=sorted(os.sched_getaffinity(0))
if len(affinity)<a.threads: raise SystemExit('BLAS workers exceed the rank CPU affinity mask')
cores=set()
try:
    for cpu in affinity:
        base=Path('/sys/devices/system/cpu/cpu%d/topology'%cpu)
        cores.add((int((base/'physical_package_id').read_text()),int((base/'core_id').read_text())))
    physical=len(cores)
except (OSError,ValueError): physical=None
trace=subprocess.check_output(['ldd',str(a.executable.resolve(strict=True))],stderr=subprocess.STDOUT,universal_newlines=True)
lines=[line.strip() for line in trace.splitlines() if 'libopenblas' in line]
if len(lines)!=1: raise SystemExit('Expected exactly one loaded OpenBLAS library: '+repr(lines))
words=lines[0].split();resolved=words[2] if len(words)>2 and words[1]=='=>' else words[0]
if Path(resolved).resolve()!=library: raise SystemExit('Loader resolved a different OpenBLAS library: '+lines[0])
lib=ctypes.CDLL(str(library))
lib.openblas_get_config.restype=ctypes.c_char_p
lib.openblas_get_parallel.restype=ctypes.c_int
lib.openblas_get_num_threads.restype=ctypes.c_int
omp=ctypes.CDLL('libgomp.so.1');omp.omp_get_max_threads.restype=ctypes.c_int
config=lib.openblas_get_config().decode();actual=lib.openblas_get_num_threads()
parallel=lib.openblas_get_parallel();ompmax=omp.omp_get_max_threads()
timeout_env=os.environ.get('OPENBLAS_THREAD_TIMEOUT')
timeout_parsed=None
if hasattr(lib,'openblas_thread_timeout'):
    lib.openblas_thread_timeout.restype=ctypes.c_uint
    timeout_parsed=lib.openblas_thread_timeout()
timeout_exponent=(max(4,min(30,timeout_parsed)) if timeout_parsed else 28) if timeout_parsed is not None else None
if parallel!=1 or actual!=a.threads or ompmax!=1 or 'USE64BITINT' in config:
    raise SystemExit('Expected pthread LP64 BLAS, requested workers and OMP1: '+repr((config,actual,parallel,ompmax)))
d=dict(status='PASS',rank=int(os.environ.get('SLURM_PROCID','0')),node=os.uname().nodename,
    library=str(library),library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
    config=config,parallel_kind=parallel,requested_threads=a.threads,reported_threads=actual,
    omp_max_threads=ompmax,openblas_thread_timeout_env=timeout_env,
    openblas_thread_timeout_parsed=timeout_parsed,openblas_thread_timeout_exponent=timeout_exponent,
    affinity_cpus=affinity,logical_cpu_count=len(affinity),physical_core_count=physical,
    physical_core_ids=sorted(cores) if physical is not None else None,
    workers_exceed_physical_cores=physical is not None and a.threads>physical,
    loader_openblas_dependencies=lines,
    scope='Probe subprocess immediately before solver exec; inherits the rank environment and affinity. Solver-side thread overrides are not measured here.')
a.output.write_text(json.dumps(d,indent=2)+'\n')
print('# BLAS runtime rank=%d threads=%d omp_max=%d affinity_cpus=%d physical_cores=%s config=%s'%
      (d['rank'],actual,ompmax,len(affinity),physical,config),flush=True)
