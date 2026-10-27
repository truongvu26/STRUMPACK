#!/usr/bin/env python3
"""Collect cases across allocations, keeping scheduler and numerical status separate."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import subprocess

parser=argparse.ArgumentParser()
parser.add_argument('directories',type=Path,nargs='+')
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--require-complete',action='store_true')
args=parser.parse_args()
expected={100:3,150:1,200:1}
allocations=[]
cases=[]
for root in args.directories:
    summary=json.loads((root/'scale_summary.json').read_text()) if (root/'scale_summary.json').exists() else {}
    cases.extend(dict(case,job=root.name) for case in summary.get('cases',[]))
    accounting=subprocess.check_output(['sacct','-j',root.name,'--format=JobIDRaw,State,Elapsed,MaxRSS,ExitCode','-P'],universal_newlines=True)
    (root/'accounting_latest.txt').write_text(accounting)
    rows=[line.split('|') for line in accounting.splitlines()[1:] if line.strip()]
    main=next((row for row in rows if row[0]==root.name),None)
    executable=(root/'executable.sha256').read_text().split()[0]
    allocation=dict(job=root.name,source_commit=(root/'source_commit.txt').read_text().strip(),
                    executable_sha256=executable,source_patch_bytes=(root/'source.patch').stat().st_size,
                    scheduler_state=main[1] if main else 'UNKNOWN',
                    scheduler_elapsed=main[2] if main else None,
                    scheduler_exit_code=main[4] if main else None,
                    numerical_status=summary.get('status','RUNNING'),artifacts=str(root),
                    excluded_unplanned_cases=summary.get('excluded_unplanned_cases',[]))
    if (root/'solver_snapshot.sha256').exists():
        allocation['frozen_executable_sha256']=(root/'solver_snapshot.sha256').read_text().split()[0]
        if allocation['frozen_executable_sha256']!=executable:
            raise SystemExit('Frozen solver does not match the recorded study executable')
    if (root/'devices.txt').exists():
        inventory=Counter()
        for line in (root/'devices.txt').read_text().splitlines():
            if line.startswith('NVIDIA '):
                columns=[column.strip() for column in line.split(',')]
                if len(columns)>=3: inventory[columns[0]+' / '+columns[2]]+=1
        allocation['gpu_inventory']=dict(inventory)
    if (root/'reused_cases.json').exists():
        allocation['reused_cases']=json.loads((root/'reused_cases.json').read_text())
    if (root/'scope_change.json').exists():
        allocation['scope_change']=json.loads((root/'scope_change.json').read_text())
    allocations.append(allocation)
if len({allocation['executable_sha256'] for allocation in allocations})!=1:
    raise SystemExit('Scaling allocations did not use the same executable')

groups=[]
failures=[]
complete=True
for n,repeats in expected.items():
    for variant in ['cpu','hybrid','evaluator']:
        trials=[case for case in cases if case['grid']==n and case['variant']==variant]
        passing=[case for case in trials if case['status']=='PASS']
        failures.extend(case['label']+': '+case.get('error','FAIL') for case in trials if case['status']!='PASS')
        if len(passing)!=repeats or {case['repeat'] for case in passing}!=set(range(1,repeats+1)):
            complete=False
        group=dict(grid=n,unknowns=n**3,variant=variant,expected_repetitions=repeats,
                   completed_repetitions=len(passing),status=('FAIL' if any(case['status']!='PASS' for case in trials) else
                           ('PASS' if len(passing)==repeats else
                            ('NOT_RUN' if not trials and all(allocation['scheduler_state'].split()[0] not in ['RUNNING','PENDING','COMPLETING','CONFIGURING'] for allocation in allocations) else 'RUNNING'))),
                   timing_evidence='three repetitions' if repeats==3 else 'single-trial preliminary')
        if passing:
            for phase in ['factor_seconds','solve_seconds','factor_plus_solve_seconds']:
                values=[case[phase] for case in passing]
                group[phase]=dict(median=statistics.median(values),minimum=min(values),maximum=max(values))
            group['maximum_relative_error']=max(case['relative_error'] for case in passing)
            group['maximum_scaled_residual']=max(case['scaled_residual'] for case in passing)
            group['hodlr_fronts']=sorted({case['hodlr_fronts'] for case in passing})
            group['prepared_device_blocks']=passing[0]['prepared_device_blocks']
        groups.append(group)
result=dict(status='FAIL' if failures else ('PASS' if complete else 'RUNNING'),
            expected_repetitions={str(n):count for n,count in expected.items()},
            layout=dict(nodes_per_case=4,mpi_ranks=32,gpus=16,ranks_per_gpu=2,omp_threads=1,rhs=3),
            evaluator_cap_bytes=8388608,allocations=allocations,groups=groups,cases=cases,failures=failures,
            notes=['Allocation status is distinct from completed numerical-case status.',
                   '100^3 allocation is intentionally released after its nine cases; larger sizes use a separate longer allocation.',
                   '150^3 and 200^3 are initial single-trial comparisons, not established speed estimates.',
                   'Native ButterflyPACK verbose checks are disabled for these timings.',
                   'All cases use the identical Release executable, host-staged MPI and zero butterfly levels.',
                   'The resumed 150^3 CPU baseline originated on all-40-GB nodes; its GPU cases use a mixed 40/80-GB allocation. Treat their timings as preliminary, not a controlled speed comparison.'])
args.output.write_text(json.dumps(result,indent=2)+'\n')
print('{}: {}/15 numerical cases PASS'.format(result['status'],sum(case['status']=='PASS' for case in cases)))
if failures or (args.require_complete and not complete): raise SystemExit(1)
