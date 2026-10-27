#!/usr/bin/env python3
"""Refresh the durable scaling evidence while long scheduler jobs run."""
import argparse
import json
from pathlib import Path
import subprocess
import time

parser=argparse.ArgumentParser()
parser.add_argument('directories',type=Path,nargs='+')
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--report',type=Path,required=True)
parser.add_argument('--poll-seconds',type=int,default=30)
args=parser.parse_args()
marker='.. BEGIN GENERATED SCALING RESULTS'
previous=None

def report(result):
    prefix=args.report.read_text().split(marker)[0].rstrip()
    lines=['', '', marker, '', 'Measured results', '-----------------', '',
           'Numerical status: '+result['status']+'. The JSON record contains individual',
           'trial accuracy, device-construction coverage and allocation accounting.', '',
           '.. list-table:: Sparse factor / solve times in seconds',
           '   :header-rows: 1', '',
           '   * - Grid', '     - CPU', '     - Hybrid GPU', '     - Evaluator GPU',
           '     - Samples per variant']
    for n in [100,150,200]:
        groups={group['variant']:group for group in result['groups'] if group['grid']==n}
        lines.append('   * - {}^3'.format(n))
        for variant in ['cpu','hybrid','evaluator']:
            group=groups[variant]
            if group['completed_repetitions']:
                value='{:.3f} / {:.3f}'.format(group['factor_seconds']['median'],group['solve_seconds']['median'])
            else: value={'FAIL':'failed','NOT_RUN':'not run'}.get(group['status'],'pending')
            lines.append('     - '+value)
        counts='/'.join(str(groups[v]['completed_repetitions']) for v in ['cpu','hybrid','evaluator'])
        lines.append('     - '+counts+' (CPU/hybrid/evaluator)')
    lines+=['', '100^3 rows use medians when repetitions have completed; 150^3 and 200^3',
            'remain single-trial observations. Pending entries are not passing tests.', '',
            'Allocation states:']
    for allocation in result['allocations']:
        lines+=['', '* Job {}: scheduler {}, numerical plan {}.'.format(
            allocation['job'],allocation['scheduler_state'],allocation['numerical_status'])]
    if result['failures']:
        lines+=['', 'Failures:']+['', '\n'.join('* '+failure for failure in result['failures'])]
    args.report.write_text(prefix+'\n'.join(lines)+'\n')

while True:
    subprocess.call(['python3','test/collect_hodlr_gpu_scale.py']+
                    [str(root) for root in args.directories]+['--output',str(args.output)],
                    stdout=subprocess.DEVNULL)
    try:
        result=json.loads(args.output.read_text())
    except (ValueError,OSError):
        time.sleep(args.poll_seconds)
        continue
    report(result)
    state=(result['status'],len(result['cases']),tuple(a['scheduler_state'] for a in result['allocations']))
    if state!=previous:
        print('{}: {}/15 cases; {}'.format(state[0],state[1],state[2]),flush=True)
        previous=state
    if result['status']=='FAIL': raise SystemExit(1)
    active=any(a['scheduler_state'].split()[0] in ['PENDING','RUNNING','COMPLETING','CONFIGURING']
               for a in result['allocations'])
    if result['status']=='PASS' and not active: break
    if not active:
        print('Allocations finished before the requested numerical plan completed.',flush=True)
        raise SystemExit(2)
    time.sleep(args.poll_seconds)
