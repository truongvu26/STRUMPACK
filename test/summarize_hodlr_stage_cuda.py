#!/usr/bin/env python3
"""CUDA activity and copy summaries; no MPI interposition or occupancy claim."""
import json, sqlite3, sys, subprocess
from pathlib import Path
from summarize_hodlr_nsys import summarize
p=Path(sys.argv[1]); result=summarize(p,1.)
result.pop('gpu_active_union_over_solver_wall')
c=sqlite3.connect(p); c.row_factory=sqlite3.Row
tables={r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
if 'NVTX_EVENTS' in tables:
    result['nvtx_events']=c.execute('SELECT COUNT(*) FROM NVTX_EVENTS').fetchone()[0]
result['notes']=['Selected process activity only; another untraced rank shares this GPU.',
 'GPU-active time is an interval union, not SM occupancy.', 'Runtime API host durations overlap GPU activity and must not be added to it.']
# Preserve the established overall summary and add launch-scope attribution.
# The report was exported on local storage; no MPI interposition is involved.
attributed=Path(sys.argv[2]+'.attribution.tmp')
subprocess.check_call([sys.executable,str(Path(__file__).with_name('summarize_hodlr_stage_attribution.py')),str(p),str(attributed)])
result['launch_scope_attribution']=json.loads(attributed.read_text())
attributed.unlink()
Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+'\n')
