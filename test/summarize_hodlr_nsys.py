#!/usr/bin/env python3
"""Summarize SQLite exports from successful, per-rank CUDA Nsight traces."""
import json
from pathlib import Path
import sqlite3
import sys


def union_seconds(intervals):
    total = 0
    lo = hi = None
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError('Invalid GPU activity interval')
        if hi is None:
            lo, hi = start, end
        elif start > hi:
            total += hi - lo
            lo, hi = start, end
        else:
            hi = max(hi, end)
    return (total + (hi - lo if hi is not None else 0)) / 1.e9


def summarize(path, solver_seconds):
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    activities = []
    for table in ('CUPTI_ACTIVITY_KIND_KERNEL', 'CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET'):
        if table in tables:
            activities.extend(tuple(row) for row in connection.execute('SELECT start,end FROM ' + table))
    kernels = connection.execute('SELECT COUNT(*), SUM(end-start)/1.e9 FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchone()
    if not kernels[0]:
        raise ValueError('Trace contains no CUDA kernels: ' + str(path))
    active = union_seconds(activities)
    runtime = connection.execute('''SELECT s.value AS name, COUNT(*) AS calls,
        SUM(r.end-r.start)/1.e9 AS seconds
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id=r.nameId
        GROUP BY r.nameId ORDER BY seconds DESC''').fetchall()
    copies = connection.execute('''SELECT e.label AS direction, COUNT(*) AS calls,
        SUM(m.bytes) AS bytes, SUM(m.end-m.start)/1.e9 AS seconds
        FROM CUPTI_ACTIVITY_KIND_MEMCPY m JOIN ENUM_CUDA_MEMCPY_OPER e ON e.id=m.copyKind
        GROUP BY e.label''').fetchall()
    top_kernels = connection.execute('''SELECT s.value AS name, COUNT(*) AS calls,
        SUM(k.end-k.start)/1.e9 AS seconds
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName
        GROUP BY k.shortName ORDER BY seconds DESC LIMIT 10''').fetchall()
    result = dict(rank=int(path.stem.replace('rank', '')), gpu_active_union_seconds=active,
                  kernel_count=kernels[0], kernel_duration_sum_seconds=kernels[1],
                  gpu_active_union_over_solver_wall=active / solver_seconds,
                  cuda_runtime_api=[dict(row) for row in runtime],
                  memory_copies=[dict(row) for row in copies],
                  top_kernels=[dict(row) for row in top_kernels])
    connection.close()
    return result


if __name__ == '__main__':
    root, sqlite_root = map(Path, sys.argv[1:3])
    profile = json.loads((root / 'summary.json').read_text())
    solver_seconds = profile['detailed']['traced_gpu']['factor_plus_solve_seconds']
    result = dict(status='PASS', solver_factor_plus_solve_seconds=solver_seconds,
                  ranks=[summarize(sqlite_root / ('rank' + str(rank) + '.sqlite'), solver_seconds)
                         for rank in range(4)], notes=[
                      'GPU-active time is the union of kernel/copy/memset intervals, not an SM occupancy counter.',
                      'The denominator is the instrumented rank-zero sparse factor-plus-solve wall time.',
                      'These diagnostics enable hodlr_verbose, which also enables native construction checks; baseline timing is collected separately.',
                      'CUDA API durations measure host time; do not add them to GPU intervals or inclusive STRUMPACK timers.',
                      'MPI tracing is disabled; communication evidence comes from STRUMPACK internal timers.'
                  ])
    (root / 'cuda_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'ranks'}))
    for rank in result['ranks']:
        print(json.dumps(dict(rank=rank['rank'], active_gpu_seconds=rank['gpu_active_union_seconds'],
                              kernel_count=rank['kernel_count'], top_api=rank['cuda_runtime_api'][0])))
