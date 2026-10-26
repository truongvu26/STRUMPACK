#!/usr/bin/env python3
"""Validate metadata A/B trials, lifecycle regression and fixed GPU sharing."""
import collections
import json
from pathlib import Path
import re
import statistics
import sys
from analyze_hodlr_gpu_profile import read_case

root = Path(sys.argv[1])
result = dict(status='PASS', job_id=int(root.name),
              source_commit=(root / 'source_commit.txt').read_text().strip(),
              butterflypack_commit=(root / 'bpack_commit.txt').read_text().strip(),
              baseline=[], detailed={}, notes=[
                  'Three alternating original-CPU/original-GPU/optimized-GPU trials share one allocation.',
                  'Baseline timing excludes internal timers, Nsight, native verbosity and CPU/GPU check mode.',
                  'Adaptive compression is time/PID seeded; numerical ranks and refinement iterations vary.',
                  'The traced run enables native verbosity and checks and is not a baseline timing sample.',
                  'Sharing is tested only for a fixed four-rank full-solver group on one node, two ranks per GPU.'
              ])
for variant in ('old_cpu', 'old_gpu', 'new_gpu'):
    trials = [read_case(root, variant + '_r' + str(repeat)) for repeat in (1, 2, 3)]
    if not all(trial['hodlr_fronts'] == 113 for trial in trials):
        raise ValueError('Expected 113 HODLR fronts in every A/B case')
    group = dict(variant=variant, trials=trials)
    for key in ('factor_seconds', 'solve_seconds', 'factor_plus_solve_seconds'):
        values = [trial[key] for trial in trials]
        group[key] = dict(median=statistics.median(values), minimum=min(values), maximum=max(values))
    result['baseline'].append(group)
for label in ('shared_sparse', 'traced_gpu'):
    case = read_case(root, label)
    text = '\n'.join(path.read_text() for path in root.glob(label + '.rank*.log'))
    native = text.count('HODLR GPU factor (')
    if native != case['hodlr_fronts']:
        raise ValueError(label + ': native GPU factor coverage mismatch')
    case['native_gpu_factors'] = native
    if label == 'shared_sparse':
        uuids = re.findall(r'# shared-GPU binding rank=\d+ .*?uuid=(\S+)', text)
        if len(uuids) != 4 or sorted(collections.Counter(uuids).values()) != [2, 2]:
            raise ValueError('Expected two ranks on each of two distinct GPU UUIDs')
        if not re.search(r'up to\s+2 ranks share a GPU', text):
            raise ValueError('Backend did not report two-rank memory sharing')
        case['ranks_per_gpu'] = 2
        case['gpu_uuids'] = sorted(set(uuids))
    result['detailed'][label] = case
if 'Metadata lifetime: PASS' not in (root / 'metadata_new.log').read_text():
    raise ValueError('Fixed metadata lifecycle regression failed')
if 'small image pins more than 1 MiB' not in (root / 'metadata_old.log').read_text():
    raise ValueError('Original metadata regression was not reproduced')
if 'HODLR hybrid interface: PASS' not in (root / 'shared_api.log').read_text():
    raise ValueError('Shared-GPU real/complex RHS regression failed')
if len(list(root.glob('gpu.rank*.nsys-rep'))) != 4:
    raise ValueError('Missing CUDA traces')
result['metadata_regression'] = dict(original='expected failure', fixed='PASS',
                                     fixed_output=(root / 'metadata_new.log').read_text().strip())
result['shared_gpu_api_regression'] = 'PASS'
(root / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
