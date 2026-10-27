#!/usr/bin/env python3
"""Human-readable timing and next-target evidence, derived from finished cases."""
import json,sys
from collections import defaultdict
from pathlib import Path
root=Path(sys.argv[1]); data=json.loads((root/'stage_summary.json').read_text())
lines=[f"# {data['grid']}³ GPU HODLR profiling — job {data['job']}", '',f"Status: {data['status']}", '',
       f"ButterflyPACK device-pool fraction: {data.get('bpack_gpu_heap_fraction',.85):.2f}.", '',
       'GPU models: '+(', '.join(sorted({x['name'] for x in data.get('gpu_devices',[])})) or 'not recorded')+'.', '',
       'Quiet cases measure solver performance. Diagnostic cases explain local phase costs; tracing adds overhead.', '',
       '| Case | Factor (s) | Solve (s) | Total (s) | Relative error | Scaled residual |',
       '|---|---:|---:|---:|---:|---:|']
for c in data['cases']:
    if c['status']=='PASS': lines.append(f"| {c['variant']} | {c['factor_seconds']:.3f} | {c['solve_seconds']:.3f} | {c['factor_plus_solve_seconds']:.3f} | {c['relative_error']:.3g} | {c['scaled_residual']:.3g} |")
    else: lines.append(f"| {c['variant']} — FAIL: {c.get('error','')} | | | | | |")
quiet={c['variant']:c for c in data['cases'] if c['status']=='PASS'}
if 'gpu' in quiet and 'cpu' in quiet:
    lines += ['',f"Quiet CPU/GPU factor-plus-solve ratio: {quiet['cpu']['factor_plus_solve_seconds']/quiet['gpu']['factor_plus_solve_seconds']:.3f}. One trial per backend; this is preliminary."]
if 'diagnostic' in quiet:
    c=quiet['diagnostic']; phases=[p for p in c['phases'] if p['section']=='solver.factor']; budget=defaultdict(float)
    def group(phase):
        if phase.endswith('.prepare') or phase.startswith('prepare.'): return 'Entry-image assembly, child extraction, replication and upload'
        if '.construct.host' in phase: return 'Compression using host callbacks (includes native waiting)'
        if '.construct.gpu' in phase: return 'Compression using device evaluator (includes native waiting)'
        if phase=='F11.factor.gpu': return 'Native GPU HODLR factorization (includes native waiting)'
        if phase=='Schur.apply.device': return 'Compound Schur GPU operator and communication'
        if phase=='Schur.compress.randomized': return 'Host randomized Schur compression outside GPU operator'
        if phase in ('HODLR.inverse','BF.multiply'): return 'Other inverse and rectangular multiply calls'
        return 'Traversal, dense fronts, setup and remaining work'
    for p in phases: budget[group(p['phase'])]+=p['exclusive_rank_seconds']['mean']
    total=sum(budget.values())
    lines += ['', '## Factorization work distribution', '',
      'Mean rank wall seconds, using exclusive nested scopes. This partitions local factor envelopes; it is not a global critical path or pure CPU/GPU execution time.', '',
      '| Component | Mean rank seconds | Fraction |', '|---|---:|---:|']
    for k,v in sorted(budget.items(),key=lambda kv:kv[1],reverse=True): lines.append(f'| {k} | {v:.3f} | {100*v/total:.1f}% |')
    lines += ['', '## Construction backend coverage', '',
      '| Block and backend | Unique fronts | Rank calls | Mean rank wall (s) | Factor envelope fraction |',
      '|---|---:|---:|---:|---:|']
    for p in sorted((p for p in phases if '.construct.' in p['phase']),
                    key=lambda p:p['exclusive_rank_seconds']['mean'],reverse=True):
        count=sum(q['phase']==p['phase'] for q in c.get('construction_fronts',[]))
        seconds=p['exclusive_rank_seconds']['mean']
        lines.append(f"| {p['phase']} | {count} | {p['calls']} | {seconds:.3f} | {100*seconds/total:.1f}% |")
    lines += ['', '## Entry-image preparation detail', '',
      'Exclusive wall time separates nested child extraction, packing, replication and upload scopes.', '',
      '| Phase | Calls | Mean rank wall (s) | Maximum rank wall (s) |',
      '|---|---:|---:|---:|']
    for p in sorted((p for p in phases if p['phase'].startswith('prepare.') or p['phase'].endswith('.prepare')),
                    key=lambda p:p['exclusive_rank_seconds']['mean'],reverse=True):
        q=p['exclusive_rank_seconds']
        lines.append(f"| {p['phase']} | {p['calls']} | {q['mean']:.3f} | {q['max']:.3f} |")
    lines += ['', '## Slowest construction fronts', '',
      'Each front appears once per block/backend; maximum and mean are over participating ranks. Different fronts may execute concurrently.', '',
      '| Phase | Front | Shape | Front ranks | Slowest rank | Maximum (s) | Mean (s) |',
      '|---|---:|---|---:|---:|---:|---:|']
    for p in c.get('construction_fronts',[])[:15]:
        lines.append(f"| {p['phase']} | {p['front']} | {p['rows']}×{p['cols']} | {p['procs']} | {p['max_rank']} | {p['max_seconds']:.3f} | {p['mean_seconds']:.3f} |")
    lines += ['',f"Maximum logical resident entry-image payload: {max(c['peak_resident_image_bytes_by_rank'])/2**30:.3f} GiB per rank. This excludes native factors and pool reservations.", '',
      '## Selecting the next change', '',
      'Use the largest measured component together with the slowest front calls and CUDA traces:', '',
      '- Large host-callback construction calls: replace full replicated images with bounded tiled device evaluation, retaining sparse entries and child Schur contributions. Raising the cap alone increases memory pressure.',
      '- Entry-image preparation: reduce host child materialization, replication and repeated packing; define distributed child ownership and stream dependencies before changing evaluation.',
      '- Schur compression outside the GPU operator: target the host randomized-compression control and transfers at its outer boundary. Device Schur intermediate vectors already remain resident.',
      '- Device construction or factorization: inspect kernel sizes, synchronization and allocation costs in the CUDA summaries before changing numerical algorithms.',
      '- Solve overhead: inspect small-RHS transfers, repeated calls during refinement and two-rank GPU contention. Factorization and solve need separate decisions.', '',
      'Rank work can include MPI waits and GPU contention. A large local phase alone does not establish global speedup potential.']
    solve=[p for p in c['phases'] if p['section']=='solver.solve']
    solve_total=sum(p['exclusive_rank_seconds']['mean'] for p in solve)
    lines += ['', '## Solve phase distribution', '',
      'Exclusive rank wall times include all refinement iterations; outer forward/backward scopes retain uninstrumented work and communication.', '',
      '| Phase | Calls | Mean rank wall (s) | Fraction |', '|---|---:|---:|---:|']
    for p in sorted(solve,key=lambda p:p['exclusive_rank_seconds']['mean'],reverse=True):
        seconds=p['exclusive_rank_seconds']['mean']
        lines.append(f"| {p['phase']} | {p['calls']} | {seconds:.3f} | {100*seconds/solve_total:.1f}% |")
for c in data['cases']:
    if 'cuda' not in c: continue
    lines += ['', '## Selected-rank CUDA activity', '', 'Only ranks zero and the last rank are traced; each shares a GPU with an untraced peer.', '',
      '| Rank | Kernel count | GPU active union (s) | Sum kernel duration (s) | NVTX events |', '|---|---:|---:|---:|---:|']
    for x in c['cuda']: lines.append(f"| {x['rank']} | {x['kernel_count']} | {x['gpu_active_union_seconds']:.3f} | {x['kernel_duration_sum_seconds']:.3f} | {x.get('nvtx_events',0)} |")
for c in data['cases']:
    if 'cuda' not in c: continue
    for x in c['cuda']:
        a=x.get('launch_scope_attribution')
        if not a: continue
        phases=[p for p in a['phases'] if p['section']=='solver.factor']
        lines += ['',f"### Rank {x['rank']}: CUDA work by launch phase",'',
          'GPU durations are sums of this process\'s operations; they can overlap and are not host phase times.', '',
          '| Phase | Kernels | Kernel sum (s) | Copy sum (s) | Copy payload (GiB) |',
          '|---|---:|---:|---:|---:|']
        for q in sorted(phases,key=lambda p:p['kernel_sum_seconds']+p['copy_sum_seconds'],reverse=True)[:10]:
            lines.append(f"| {q['phase']} | {q['kernels']} | {q['kernel_sum_seconds']:.3f} | {q['copy_sum_seconds']:.3f} | {q['copy_bytes']/2**30:.3f} |")
        lines += ['',f"### Rank {x['rank']}: highest CUDA host API costs",'',
          '| API | Calls | Host duration sum (s) |', '|---|---:|---:|']
        for q in a['cuda_runtime_api'][:8]:
            lines.append(f"| {q['name']} | {q['calls']} | {q['host_seconds']:.3f} |")
lines += ['', '## Measurement limits', '']+['- '+x for x in data['notes']]
(root/'stage_report.md').write_text('\n'.join(lines)+'\n')
