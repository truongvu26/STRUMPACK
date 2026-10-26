#!/usr/bin/env python3
"""Validate and summarize baseline trials and inclusive internal timer totals."""
import json
import math
from pathlib import Path
import re
import statistics
import sys


def scalar(text, pattern, convert=float):
    found = re.findall(pattern, text)
    if len(found) != 1:
        raise ValueError('Expected one match for ' + pattern)
    value = convert(found[0])
    if not math.isfinite(value):
        raise ValueError('Nonfinite result: ' + pattern)
    return value


def read_case(root, label):
    logs = sorted(root.glob(label + '.rank*.log'))
    if len(logs) != 4:
        raise ValueError(label + ': expected four rank logs')
    text = (root / (label + '.rank0.log')).read_text()
    error = scalar(text, r'# relative error = .* = (\S+)')
    residual = scalar(text, r'# COMPONENTWISE SCALED RESIDUAL = (\S+)')
    fronts = scalar(text, r'nr of HODLR Frontal matrices = (\d+)', int)
    if not 0 <= error < 1.e-8 or not 0 <= residual < 1.e-6 or fronts < 2:
        raise ValueError(label + ': numerical or multiple-front check failed')
    result = dict(case=label, relative_error=error, scaled_residual=residual,
                  hodlr_fronts=fronts,
                  refinement_iterations=scalar(text, r'number of Krylov iterations = (\d+)', int))
    for phase in ('factor', 'solve'):
        result[phase + '_seconds'] = scalar(text, r'#\s+- ' + phase + r' time = (\S+)')
    if result['factor_seconds'] <= 0 or result['solve_seconds'] <= 0:
        raise ValueError(label + ': invalid wall time')
    result['factor_plus_solve_seconds'] = result['factor_seconds'] + result['solve_seconds']
    return result


def analyze(root, diagnostics=None):
    diagnostics = diagnostics or root
    result = dict(status='PASS', baseline_job_id=int(root.name), diagnostic_job_id=int(diagnostics.name),
                  baseline_source_commit=(root / 'source_commit.txt').read_text().strip(),
                  diagnostic_source_commit=(diagnostics / 'source_commit.txt').read_text().strip(),
                  baseline=[], detailed={}, notes=[
                      'Baseline trials use the uninstrumented Release binary and fresh MPI processes.',
                      'Internal timers and Nsight tracing are separate diagnostic runs; their wall times are not baseline timings.',
                      'ButterflyPACK seeds adaptive CPU compression from clock/PID; factors and refinement work vary between trials.',
                      'Internal timer totals are inclusive and overlap; never add rows or interpret rank maxima as critical-path wall time.',
                      'Only 40^3 and 60^3, three RHS, four ranks, one rank per GPU are characterized.'
                  ])
    for n in (40, 60):
        for mode in (0, 1):
            trials = [read_case(root, 'baseline_n{}_mode{}_r{}'.format(n, mode, repeat))
                      for repeat in (1, 2, 3)]
            group = dict(grid=n, mode=mode, trials=trials)
            for phase in ('factor_seconds', 'solve_seconds', 'factor_plus_solve_seconds'):
                values = [trial[phase] for trial in trials]
                group[phase] = dict(median=statistics.median(values), minimum=min(values), maximum=max(values))
            result['baseline'].append(group)
    for mode, label in ((0, 'detailed_cpu'), (1, 'detailed_gpu'), (1, 'traced_gpu')):
        case = read_case(diagnostics, label)
        text = (diagnostics / (label + '.rank0.log')).read_text()
        timers = {}
        for line in text.splitlines():
            if not line.startswith('|'):
                continue
            columns = [column.strip() for column in line.strip('|').split('|')]
            if len(columns) != 5:
                continue
            try:
                values = list(map(float, columns[1:]))
            except ValueError:
                continue
            if not all(math.isfinite(value) and value >= 0 for value in values):
                raise ValueError(label + ': invalid timer row')
            timers[columns[0]] = dict(zip(('sum_rank_seconds', 'min_rank_seconds', 'max_rank_seconds', 'mean_rank_seconds'), values))
        if not all(name in timers for name in ('H_compress', 'LRBF_compress', 'extract_sep_2d', 'get_submatrix_2d')):
            raise ValueError(label + ': missing internal phase timers')
        case['inclusive_internal_timers'] = timers
        if mode:
            native_count = sum(path.read_text().count('HODLR GPU factor (')
                               for path in diagnostics.glob(label + '.rank*.log'))
            if native_count != case['hodlr_fronts']:
                raise ValueError('GPU factor coverage mismatch')
            case['native_gpu_factors'] = native_count
        if label == 'traced_gpu':
            reports = sorted(diagnostics.glob('gpu.rank*.nsys-rep'))
            if len(reports) != 4:
                raise ValueError('Expected four Nsight rank reports')
            case['nsight_reports'] = [path.name for path in reports]
        result['detailed'][label] = case
    (diagnostics / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    print(json.dumps(analyze(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else None), indent=2))
