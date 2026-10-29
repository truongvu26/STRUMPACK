#!/usr/bin/env python3
"""Collect explicit HODLR tuning summaries without reading live logs or snapshots."""
import argparse
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


TARGETS = {100: 14.2, 150: 67.0, 200: 119.0}
LAYOUT = ('grid', 'ranks', 'nodes', 'ranks_per_gpu')
IMPLICIT_DEFAULTS = dict(gpu_rectangular=1, gpu_host_panels=1,
                         solver_tol=1e-10, baca_residual_guard=0, blas_thread_timeout=28)
BUILD_HASHES = ('executable_sha256', 'native_library_sha256', 'blas_library_sha256')
TIMINGS = ('factor_seconds', 'solve_seconds', 'factor_plus_solve_seconds',
           'nd_seconds', 'symbolic_seconds', 'sep_reorder_seconds',
           'end_to_end_seconds','host_panel_prepare_seconds_rank0',
           'host_panel_metadata_seconds_rank0','host_panel_upload_seconds_rank0',
           'host_panel_upload_bytes_rank0','prepared_image_seconds_rank0',
           'max_cpu_dense_front_seconds')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def snapshot_metadata(text):
    """Retain commit/hash provenance, never an embedded source patch."""
    commits, files = [], []
    for line in text.splitlines():
        if line.startswith('diff --git '):
            break
        if re.fullmatch(r'[0-9a-fA-F]{40}', line):
            commits.append(line)
        match = re.match(r'^([0-9a-fA-F]{64})\s+[* ]?(.+)$', line)
        if match:
            files.append(dict(sha256=match.group(1), path=match.group(2)))
    return dict(sha256=digest(text.encode('utf-8')), commits=commits,
                files=files, embedded_text_omitted=True)


def valid_number(value, positive=False):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def passing(record, manifest):
    if record.get('status') != 'PASS':
        return False
    settings = record['settings']
    accuracy = manifest.get('accuracy', {})
    hashes = [manifest.get(key) for key in BUILD_HASHES]
    hashes_valid = all(isinstance(v, str) and re.fullmatch(r'[0-9a-fA-F]{64}', v)
                       for v in hashes[:2])
    hashes_valid = hashes_valid and (hashes[2] is None or
        isinstance(hashes[2], str) and re.fullmatch(r'[0-9a-fA-F]{64}', hashes[2]))
    return (hashes_valid and valid_number(record.get('factor_seconds'), positive=True)
            and valid_number(record.get('solve_seconds'))
            and valid_number(record.get('hodlr_fronts'), positive=True)
            and valid_number(record.get('relative_error'))
            and record['relative_error'] < min(1e-8, accuracy.get('relative_error', 1e-8))
            and valid_number(record.get('scaled_residual'))
            and record['scaled_residual'] < min(1e-6, accuracy.get('scaled_residual', 1e-6))
            and valid_number(record.get('refinement_iterations'))
            and record['refinement_iterations'] <= settings.get('maxit', 30))


def collect(paths):
    sources, records, groups = [], [], defaultdict(list)
    for index, path in enumerate(paths):
        data = path.read_bytes()
        summary = json.loads(data.decode('utf-8'))
        manifest = summary['manifest']
        layout = {key: manifest[key] for key in LAYOUT}
        if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0
               for v in layout.values()):
            raise ValueError('Invalid layout in ' + str(path))
        source_id = 'source-%03d' % (index + 1)
        build_identity = {key: manifest.get(key) for key in BUILD_HASHES}
        copied_manifest = dict(manifest)
        snapshot = copied_manifest.pop('snapshot_manifest', '')
        copied_manifest['snapshot_provenance'] = snapshot_metadata(snapshot)
        sources.append(dict(id=source_id, path=str(path), sha256=digest(data),
                            job=summary.get('job', path.parent.name),
                            status=summary.get('status'), manifest=copied_manifest,
                            finished_plan=summary.get('finished_plan'),
                            all_candidates_tested=summary.get('all_candidates_tested'),
                            notes=summary.get('notes', [])))
        for case in summary['cases']:
            record = dict(case)
            record.update(source_id=source_id, source_job=sources[-1]['job'], **layout)
            record['build_identity'] = build_identity
            record['record_id'] = source_id + '/' + case['label']
            record['eligible_for_ranking'] = passing(record, manifest)
            if case.get('status') == 'PASS' and not record['eligible_for_ranking']:
                record['collector_rejection'] = 'PASS record fails build-hash/measurement/accuracy/iteration checks'
            records.append(record)
            if not record['eligible_for_ranking']:
                continue
            # Derive the objective from its components instead of trusting a rounded sum.
            record['factor_plus_solve_seconds'] = record['factor_seconds'] + record['solve_seconds']
            settings = dict(IMPLICIT_DEFAULTS)
            settings.update({k: v for k, v in case['settings'].items() if k != 'timeout_seconds'})
            record['implicit_defaults_used'] = {k: v for k, v in IMPLICIT_DEFAULTS.items() if k not in case['settings']}
            key = dict(layout, settings=settings, build_identity=build_identity)
            groups[json.dumps(key, sort_keys=True)].append(record)

    by_grid = defaultdict(list)
    for key, trials in groups.items():
        config = json.loads(key)
        config.update(trials=len(trials), record_ids=[c['record_id'] for c in trials],
                      jobs=sorted({c['source_job'] for c in trials}),
                      relaxed_compression=any(c.get('relaxed_compression',
                          c['settings'].get('tol', 1e-5) > 1e-5) for c in trials))
        for timing in TIMINGS:
            if all(valid_number(c.get(timing)) for c in trials):
                config['median_' + timing] = statistics.median(c[timing] for c in trials)
        config['factor_seconds_range'] = [min(c['factor_seconds'] for c in trials),
                                          max(c['factor_seconds'] for c in trials)]
        source_ids = {c['source_id'] for c in trials}
        build_settings = {json.dumps(s['manifest'].get('build_settings', {}), sort_keys=True)
                          for s in sources if s['id'] in source_ids}
        config['build_settings_variants'] = [json.loads(v) for v in sorted(build_settings)]
        target = TARGETS.get(config['grid'])
        if target is not None:
            factor = config['median_factor_seconds']
            config['screenshot_comparison'] = dict(target_gpu_factor_seconds=target,
                factor_difference_seconds=factor-target,
                factor_speedup_vs_screenshot=target/factor,
                beats_screenshot_factor=factor < target)
        by_grid[config['grid']].append(config)

    rankings, best = [], []
    for grid in sorted({c['grid'] for c in records}):
        configs = sorted(by_grid[grid], key=lambda c: (
            c['median_factor_plus_solve_seconds'], c['median_factor_seconds'],
            json.dumps(c['settings'], sort_keys=True)))
        for rank, config in enumerate(configs, 1):
            config['rank'] = rank
        rankings.append(dict(grid=grid, configurations=configs,
                             screenshot_gpu_factor_seconds=TARGETS.get(grid)))
        fastest_factor = min(configs, key=lambda c: (
            c['median_factor_seconds'], c['median_factor_plus_solve_seconds'])) if configs else None
        best.append(dict(grid=grid, best_configuration=configs[0] if configs else None,
                         best_factor_configuration=fastest_factor,
                         recommended_configuration=next((c for c in configs if c['trials']>=3),None)))
    counts = dict(Counter(c.get('status', 'UNKNOWN') for c in records))
    pending = any(c.get('status') not in ('PASS', 'FAIL', 'SKIPPED') for c in records)
    if not groups:
        status = 'NO_PASSING_CANDIDATES'
    elif pending or counts.get('SKIPPED'):
        status = 'PARTIAL'
    elif counts.get('FAIL') or any(c.get('collector_rejection') for c in records):
        status = 'PASS_WITH_REJECTED_CANDIDATES'
    else:
        status = 'PASS'
    return dict(schema_version=1, generated_utc=datetime.utcnow().isoformat()+'Z',
        collector_sha256=digest(Path(__file__).read_bytes()), status=status,
        objective='Rank by median validated factor-plus-solve seconds; factor seconds secondary.',
        grouping_fields=list(LAYOUT)+['effective settings excluding timeout_seconds']+list(BUILD_HASHES),
        screenshot_gpu_factor_seconds={str(k): v for k, v in TARGETS.items()},
        caveats=[
            'Screenshot commands, hardware, GPU counts and factor timing boundaries are unknown; comparisons are indicative, not matched benchmarks.',
            'These are the best tested settings, not a global optimum; one-trial configurations are not repeat-validated.',
            'Only source PASS records that meet final error <1e-8, scaled residual <1e-6, positive GPU HODLR fronts and iteration limits enter rankings.',
            'Failed, skipped and pending candidates remain in cases; collection status does not mean every candidate passed or was tested.',
            'Repeats require identical layout, effective settings excluding timeout, and executable/native/BLAS hashes; different builds never share a repeat group. Historical fixed GPU flags/solver tolerance/guard defaults are normalized explicitly.',
            'Screenshot comparisons use median factor time; primary selection uses median factor-plus-solve time. A separate best-factor configuration exposes solve-time tradeoffs.'],
        sources=sources, case_status_counts=counts, cases=records,
        rankings_by_grid=rankings, best_by_grid=best)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summaries', nargs='+', type=Path,
                        help='Explicit tuning_summary.json inputs; no directory scanning')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        paths = [p.resolve(strict=True) for p in args.summaries]
        if len(paths) != len(set(paths)):
            parser.error('Duplicate summary inputs would count trials twice')
        if args.output.resolve() in paths:
            parser.error('Output must not overwrite an input summary')
        report = collect(paths)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    print('%s: %d cases, %s -> %s' % (
        report['status'], len(report['cases']), report['case_status_counts'], args.output))
    for item in report['best_by_grid']:
        best = item['best_configuration']
        if best is None:
            print('%d^3: no passing configuration' % item['grid'])
        else:
            print('%d^3: %.3fs factor + %.3fs solve, %d trial(s), %d ranks/%d nodes' % (
                item['grid'], best['median_factor_seconds'], best['median_solve_seconds'],
                best['trials'], best['ranks'], best['nodes']))


if __name__ == '__main__':
    main()
