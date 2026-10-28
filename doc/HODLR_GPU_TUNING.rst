:orphan:

GPU HODLR settings on Perlmutter
================================

Selected configurations: PASS. The recommended groups contain at least
three successful full sparse Poisson solves, each with three RHS. The
parameter search contains 143 cases: 94 passed, 31 were rejected,
and 18 were deliberately stopped. All scheduled jobs are finished.
The complete settings, failures, hashes and individual measurements are
in HODLR_GPU_TUNING.json. The collection's PARTIAL status preserves the
stopped branches; it does not describe the selected validation status.

Best repeat-validated results
----------------------------

The first table minimizes median factor time. The second table minimizes
median factor-plus-solve time. Factor time is the full sparse driver's
reported factor time, including frontal construction/compression.
Component medians are listed separately;
the median paired total need not equal their sum.

.. list-table:: Fastest repeat-validated factor configurations, seconds
   :header-rows: 1

   * - Grid
     - Layout
     - Factor
     - Solve
     - Paired total
     - Trials
     - Screenshot factor
   * - 100 cubed
     - 8 MPI x 16 BLAS; 8 GPUs / 2 nodes
     - 11.118
     - 2.237
     - 13.355
     - 3
     - 14.2
   * - 150 cubed
     - 8 MPI x 16 BLAS; 8 GPUs / 2 nodes
     - 61.770
     - 14.200
     - 75.970
     - 3
     - 67.0
   * - 200 cubed
     - 16 MPI x 16 BLAS; 16 GPUs / 4 nodes
     - 130.104
     - 39.501
     - 168.811
     - 3
     - 119.0

.. list-table:: Fastest repeat-validated factor-plus-solve configurations
   :header-rows: 1

   * - Grid
     - Factor
     - Solve
     - Paired total
     - MPI x BLAS workers; GPUs
   * - 100
     - 12.062
     - 0.502
     - 12.565
     - 8 x 16; 8 GPUs
   * - 150
     - 63.109
     - 10.247
     - 73.188
     - 16 x 8; 8 GPUs
   * - 200
     - 130.104
     - 39.501
     - 168.811
     - 16 x 16; 16 GPUs

The screenshot comparison is indicative. Its commands, GPU count/model,
node placement and factor timing boundaries are unavailable. These runs
use different MPI/thread layouts. The screenshot lists 4 x 16, 4 x 16 and
8 x 16; the layouts above use more MPI ranks and CPU workers. All measured
GPUs are NVIDIA A100-SXM4-40GB. STRUMPACK OpenMP stays at one thread; the
BLAS column denotes configured pthread OpenBLAS workers per MPI rank.
The raw screenshot target remains unmet for 200 cubed.

The 100-cubed strict 1e-5 compression configuration factors in 12.062
seconds and solves in 0.502 seconds. Its paired total is faster than the
11.118-second factor configuration with 2.237 seconds solve. The
150-cubed front-size configuration likewise trades a lower factor time
for a longer solve. It sets minimum separator 20000 and minimum front
26000, which selects seven compressed fronts in this case. Accuracy is
checked on the original sparse system in both cases.

Settings to reproduce
---------------------

.. list-table:: Driver parameters for the fastest factor configurations
   :header-rows: 1

   * - Grid
     - min_sep
     - min_front
     - leaf_size
     - lr_leaf
     - hodlr_rel_tol
     - sp_rel_tol
   * - 100
     - 3500
     - 100000
     - 512
     - 5
     - 0.01
     - 1e-12
   * - 150
     - 20000
     - 26000
     - 1024
     - 4
     - 0.01
     - 1e-12
   * - 200
     - 8000
     - 100000
     - 1024
     - 5
     - 0.01
     - 1e-12

``min_sep`` and ``min_front`` map to
``--sp_compression_min_sep_size`` and ``--sp_compression_min_front_size``.
``leaf_size`` and ``lr_leaf`` map to ``--hodlr_leaf_size`` and
``--hodlr_lr_leaf``.

The common options are::

   --sp_compression HODLR --sp_Krylov_solver refinement --sp_maxit 100
   --hodlr_use_gpu 2 --hodlr_butterfly_levels 0
   --hodlr_gpu_rectangular 1 --hodlr_gpu_host_panels 1
   --hodlr_gpu_front_max_bytes 0 --hodlr_gpu_panel_max_bytes 67108864
   --hodlr_knn_hodlrbf 0 --hodlr_knn_lrbf 0
   --hodlr_BACA_block_size 64 --hodlr_quiet

Use ``OMP_NUM_THREADS=1``, ``BPACK_GPU_HODLR_SVD=qr``,
``BPACK_GPU_HEAP_FRACTION=0.85``, ``BPACK_GPU_BACA_RESIDUAL_GUARD=0``,
``BPACK_GPU_AWARE_MPI=0`` and ``MPICH_GPU_SUPPORT_ENABLED=0``. All
factor-focused
configurations also set ``--sp_abs_tol 1e-12`` and configure sixteen
pthread BLAS workers per rank with ``OPENBLAS_NUM_THREADS=16``. Their coarser
compression uses iterative refinement to obtain the checked final
solution; solve time and iteration counts remain in the record.

The baseline OpenBLAS library is single threaded. Setting its thread
environment variable alone does not enable parallel BLAS. These tests
use a frozen compatible LP64 pthread OpenBLAS 0.3.30 and verify each rank's
library hash, configured workers, one loaded OpenBLAS dependency and
non-overlapping physical CPU affinity. ``T=1`` in STRUMPACK's log describes
its OpenMP setting, independently of the BLAS worker count.

For example, extract the fastest factor settings for 150 cubed. Use
``recommended_configuration`` and its layout to reproduce the fastest
paired total instead::

   python3 - <<'PYCODE'
   import json
   report = json.load(open('doc/HODLR_GPU_TUNING.json'))
   item = next(v for v in report['best_by_grid'] if v['grid'] == 150)
   settings = item['best_factor_configuration']['settings']
   plan = [dict(settings, label='trial-'+str(i), timeout_seconds=400)
           for i in range(3)]
   json.dump(plan, open('/tmp/hodlr150-plan.json', 'w'), indent=2)
   PYCODE
   export HODLR_TUNE_PLAN=/tmp/hodlr150-plan.json
   export HODLR_TUNE_SNAPSHOT="$PWD/profile_results/guard32_snapshot_v1"
   export HODLR_TUNE_BLAS_LIBRARY="$PWD/profile_results/openblas_pthreads_snapshot_v1/lib/libopenblas.so.0.3"
   export HODLR_TUNE_GRID=150 HODLR_TUNE_RANKS=8
   export HODLR_RANKS_PER_GPU=1 HODLR_TUNE_BLAS_THREADS=16
   export HODLR_TUNE_NODES=2 HODLR_TUNE_CPUS_PER_RANK=16
   salloc --immediate=60 -A m4872_g --qos=interactive -C 'gpu&hbm40g' \
     -N 2 --gpus-per-node=4 --ntasks-per-node=4 --cpus-per-task=16 \
     --hint=nomultithread --mem=0 -t 00:25:00 \
     bash test/tune_hodlr_gpu_perlmutter.sbatch

Run from the STRUMPACK worktree. The snapshot directory contains the
built driver as ``panels``, the native library under ``lib/`` and a
``manifest.txt``. Supply a corresponding snapshot when reproducing on
another installation. The harness freezes itself and its analyzer,
pins the selected nodes, binds physical GPU UUIDs, and records commands,
per-rank logs, affinity, hashes and numerical checks under profile_results.

What the measurements establish
-------------------------------

Do not carry the earlier forced 1000/1000 validation thresholds into a
performance comparison without checking their effect. Separator and
front thresholds are OR conditions: increasing only one can leave many
fronts compressed. They determine whether expensive work uses exact CPU
fronts or GPU HODLR, and must be tuned together with MPI/BLAS placement.

On the four-rank 100-cubed control, increasing verified pthread BLAS
workers from one to sixteen changed factor time from 73.776 to 19.163
seconds. Merely allocating CPUs or setting an environment variable for
a single-threaded library does not achieve this change.

With the same sixteen GPUs and 256 physical CPU cores, changing the
200-cubed layout from 32 MPI x 8 BLAS workers to 16 MPI x 16 workers
reduced median factor time from 177.123 to 130.104 seconds, and median
paired time from 201.852 to 168.811 seconds. Fewer MPI ranks keep more
exact dense work in local BLAS instead of distributed ScaLAPACK.

The 32-rank 200-cubed control recorded a single CPU DenseMPI partial
factorization of about 83 seconds. Its rank-zero host-panel preparation
counter was also substantial. The JSON includes these diagnostics for
each passing case. Preparation counters include extraction and MPI waits;
they are local wall counters and must not be added to CPU-front maxima
as a complete CPU/GPU time decomposition. Local exact dense fronts are
not included in the DenseMPI counter.

Lower thresholds sometimes produced LAPACK DLASCL illegal-argument
messages even when final error checks passed. Those cases are rejected.
The optional residual guard increased some construction callback counts
from tens to thousands and hit the 300-second test budget. The data do
not establish the precise cause of those LAPACK diagnostics. These
settings are validated for the tested Poisson driver, rather than a
claim of a universal parameter optimum.

A global PBLAS block-size 128 experiment was removed. Its safe root-only
trial took 22.390 seconds; no speedup was established. ButterflyPACK's
``nbslpk=32`` is part of its distributed extraction layout, so arbitrary
STRUMPACK block-size changes are incompatible with compressed-child
extraction without an explicit redistribution adapter.

Image redistribution guard
--------------------------

The installed LP64 ScaLAPACK redistribution allocator takes a signed
32-bit byte count. A 22500 x 22500 FP64 image requires 4050000000 bytes,
which overflows that interface. The old check bounded elements instead
of bytes and could terminate with ``xxmr2d:out of memory``.

Commit 52a7deed bounds replica bytes before construction and reports
``prepared_image_exceeds_redistribution_byte_limit`` before falling back
to panels. Job 60472972, run label guard150, completed a full 150-cubed
solve with an 8-GiB image budget and three such fallback diagnostics.
Factor/solve times were 110.241/3.805 seconds, relative error 1.58735e-11
and scaled residual 4.66196e-11. This validates the fallback, rather than
making the large budget a performance recommendation.

The earlier communicator/shared-tree fix, upstream multi-RHS correction,
QR SVD and device WY implementation remain intact. Native ButterflyPACK
code was unchanged in this parameter study. STRUMPACK's own CUDA dense
backend remains disabled. Profiling justifies future work on exact dense
fronts and host assembly/extraction with correct child Schur data; merely
enabling CUDA does not select a dense GPU fallback for HODLR in this code.
Raw logs, job folders, snapshots and build directories remain locally
ignored and are not part of the source commit.
