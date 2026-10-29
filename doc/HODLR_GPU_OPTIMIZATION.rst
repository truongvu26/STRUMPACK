Hybrid GPU HODLR: profiling and focused optimization
==================================================

The optimized 200-cubed hybrid path improves from 129.169 to 101.955 seconds
for factorization after moving large exact local Schur updates to GPU.
These are three-trial medians from job 60474983 on the same 16 MPI ranks,
16 A100 GPUs and four nodes, with 16 physical CPU cores per rank,
pthread OpenBLAS 16 workers and OMP_NUM_THREADS=1. All six solves use three
RHS and pass the original sparse-matrix residual and solution-error checks.

The controls keep ButterflyPACK GPU HODLR enabled. They compare the new
local Schur option off versus on; they are not an all-CPU HODLR baseline.
The previous settings study remains unchanged in HODLR_GPU_TUNING.rst.
Machine-readable evidence is in HODLR_GPU_OPTIMIZATION.json.

Measured bottlenecks
--------------------

Job 60474379 measured 129.943 seconds factor plus 37.5552 seconds solve on
the optimized 200-cubed layout. Job 60474786 repeated the same executable
and settings under Nsight Systems: 131.823 plus 36.8424 seconds. Both passed.
The large local dense-front scopes cover separators of at least 1000;
smaller fronts remain in the enclosing traversal counter.

The diagnostic exclusive mean-rank factor wall partitions are:

.. list-table:: Where the hybrid factor time goes
   :header-rows: 1
   :widths: 70 15 15

   * - Stage
     - Seconds
     - Fraction
   * - Host entry callbacks: sparse/child extraction, packing and waits
     - 52.58
     - 40.5%
   * - Exact local dense fronts
     - 34.39
     - 26.5%
   * - Traversal, small fronts and remaining waits
     - 19.03
     - 14.6%
   * - Native GPU compression outside callbacks, including metadata/waits
     - 13.55
     - 10.4%
   * - Host randomized Schur compression
     - 9.86
     - 7.6%
   * - Compound GPU Schur operator
     - 0.36
     - 0.3%
   * - GPU HODLR factorization API
     - 0.19
     - 0.1%

These are concurrent rank-wall partitions, not an additive global critical
path or pure CPU/GPU compute measurements. Local exact Schur GEMM alone
accounts for 23.21 mean-rank seconds, with a maximum of 29.67 seconds.

Nsight ranks 0, 8 and 15 show only 3.22--3.75 seconds of CUDA kernel/copy/
memset activity union per rank, 2.44--2.84% of factor wall. This is not SM
occupancy. CPU running samples mostly reach OpenBLAS, with about 64--65%
at its worker wait loop and 26% at its DGEMM kernel. Worker samples are
assigned only temporally to an unambiguous solver window in the same
validated process; they do not recover the submitting BLAS/ScaLAPACK caller
or inherit a main-thread nested phase. These sample fractions are not wall
fractions and cannot establish that ScaLAPACK is absent.

The first profiling launch failed in Nsight's process probe before solver
startup. Target-only LD_PRELOAD and an optional node-local driver copy
allowed the subsequent smoke test and full profile to complete. This does
not isolate which change resolved the profiler failure.

Focused changes and matched results
-----------------------------------

The proportional sparse extractor now scans each requested sparse column
once and uses a lookup of requested row positions, preserving duplicate
requests and the first stored duplicate value. It also fixes mixed unsorted
row queries that the old early exit could truncate. The MPI fixture checks
an independent dense oracle across real/complex scalar types, index types,
BLACS ownership, empty cases, duplicates and separator masks.

The original lookup microbenchmark improves 3.85 times at MPI1 and 9.76
times at MPI4. The six matched 200-cubed full solves in job 60474786 show
a much smaller median factor change, 133.626 to 128.783 seconds, and paired
factor-plus-solve change, 172.2836 to 167.4395 seconds. Timing ranges overlap
and one adjacent pair regresses. The microbenchmark gain must not be quoted
as a whole-solver speedup.

The separate exact-front GPU change replaces only F22 -= F21 * F12 after
CPU pivoting/LU/TRSM. It preserves existing child F22 through beta=1, streams
bounded panels, and fences host output before parent extraction. It does
not enable STRUMPACK's separate CUDA dense-front backend. See
HODLR_GPU_DENSE_SCHUR.rst for memory, errors and supported configurations.

.. list-table:: Same-binary 200-cubed medians, three trials per option
   :header-rows: 1
   :widths: 50 25 25

   * - Measurement
     - Local Schur on CPU
     - Local Schur on GPU
   * - Factor seconds
     - 129.169
     - 101.955
   * - Solve seconds
     - 38.801
     - 39.1016
   * - Paired factor plus solve seconds
     - 168.0662
     - 141.8683

Factor time falls 21.07%; paired factor plus solve falls 15.59%. Each GPU
trial executes 210 GPU updates across all 16 ranks with no fallback. The
median maximum per-rank sum of helper wall time is 4.82 seconds, of which
0.91 seconds is device DGEMM event time. Each trial transfers 372.18 GB to
device and 277.58 GB back across all ranks. Explicit helper buffers stay
below 2 GiB per process; these counts exclude native pools and CUDA/cuBLAS
workspace. Transfers and setup remain material even after this improvement.

The stopped job 60474930 used an incorrect four-rank sparse timing layout;
those timings are excluded. Its one- and two-process GPU fixtures passed.
Corrected job 60474983 repeated the fixtures and all six full sparse solves.

Scaling and reproducible settings
--------------------------------

Job 60475240 repeats the comparison at 100 and 150 cubed, three PASS trials
per option. It also runs the tracked proportional-extraction fixture on one
and four MPI ranks. Every rank verifies its actual pthread BLAS worker count,
physical core binding, GPU UUID, timeout environment and parsed timeout getter.

.. list-table:: Matched factor medians with the local Schur option off/on
   :header-rows: 1
   :widths: 15 20 20 20 25

   * - Grid
     - MPI / GPUs / nodes
     - Helper off
     - Helper on
     - Paired total off/on
   * - 100 cubed
     - 8 / 8 / 2
     - 10.8801 s
     - 10.2374 s
     - 13.12237 / 12.48197 s
   * - 150 cubed
     - 8 / 8 / 2
     - 60.0402 s
     - 48.1194 s
     - 74.5095 / 62.7443 s
   * - 200 cubed
     - 16 / 16 / 4
     - 129.169 s
     - 101.955 s
     - 168.0662 / 141.8683 s

The 100/150 factor reductions are 5.91% and 19.85%; paired total reductions
are 4.88% and 15.79%. Each row is a separate matched cohort. Paired totals
are medians of each trial's factor-plus-solve, not sums of separate medians.

Use mode 2, butterfly levels 0, GPU rectangular and host-panel paths enabled,
GPU dense Schur enabled, full-image cap 0, host panel cap 64 MiB, BACA batch
64, KNN counts 0/0, QR SVD, heap fraction 0.85, compression tolerance 1e-2,
solver relative/absolute tolerance 1e-12 and maximum 100 refinement iterations.
The per-grid compression settings are:

.. list-table:: Remaining settings
   :header-rows: 1
   :widths: 20 20 20 20 20

   * - Grid
     - Minimum separator
     - Minimum front
     - Leaf size
     - RecLR_leaf
   * - 100 cubed
     - 3500
     - 100000
     - 512
     - 5
   * - 150 cubed
     - 20000
     - 26000
     - 1024
     - 4
   * - 200 cubed
     - 8000
     - 100000
     - 1024
     - 5

These front thresholds use the existing separator-OR-front eligibility rule.
All rows use OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=16 and
OPENBLAS_THREAD_TIMEOUT=28, with 16 physical CPU cores per MPI rank and one
MPI rank per GPU. MPICH_GPU_SUPPORT_ENABLED and BPACK_GPU_AWARE_MPI are 0.
The benchmark uses compatible LP64 pthread OpenBLAS 0.3.30, loaded instead
of the original single-thread library. Setting its environment variables
alone does not turn a single-thread BLAS build into a threaded one. The
committed tuning harness accepts HODLR_TUNE_BLAS_LIBRARY and checks the
loaded library hash, one OpenBLAS dependency, workers and core placement.
STRUMPACK_USE_CUDA remains OFF; the HODLR CUDA/cuBLAS capability checks pass.

The 200-cubed timeout pilots at exponents 28, 16 and 20 respectively produce
factor/solve times 103.671/36.3271, 99.5132/38.09 and 101.992/37.1544 seconds.
All pass numerical checks and confirm the parsed runtime exponent. Each is
one trial; these results do not establish a new recommended timeout. The
verified default 28 remains selected.

After GPU Schur offload, job 60475487 checks whether fewer BLAS workers
help the remaining CPU work. With the same 16 ranks, GPUs, physical-core
allocation and timeout 28, its 4/8/16-worker pilots respectively measure
factor/solve times 120.675/34.9542, 105.401/35.2116 and 102.476/35.4988
seconds. All three pass. Neither alternative meets the predeclared
factor-improvement threshold, so no additional confirmations are run and
the established 16-worker setting is retained. These are single-trial pilots
and are kept separate from the repeated performance cohorts.

The screenshot's factor times are 14.2, 67 and 119 seconds. The new measured
medians are lower, but screenshot commands, GPU model/count and timing
boundaries are unavailable, and its MPI layouts differ. This is a raw-time
comparison, not a matched-resource speedup.

The completed record contains 33 passing full sparse solves, each with three
RHS. The maximum observed relative solution error is 1.98279e-10 and maximum
componentwise scaled residual is 5.30418e-10, below the retained acceptance
limits of 1e-8 and 1e-6. Standalone GPU and proportional-extraction fixtures
are additional checks. Raw logs, Nsight reports, job folders and build trees
are kept under local ignore rules; the commit contains source and reports.

Remaining host construction work
--------------------------------

The remaining callback wall includes original sparse extraction, child
contributions, the current Schur term, MPI and waiting. Source review shows
that GPU-backed child HODLR extraction fetches selected factor rows back to
host, reconstructs entries with CPU GEMM and host MPI, then releases fetched
row caches after each call. F22 construction also extracts the current
rectangular Schur term through a separate host path.

ButterflyPACK has no public arbitrary-entry extraction API producing device
output. A construction evaluator alone cannot replace these operations;
collectives are forbidden inside its rank-local callback.

Job 60475371 adds diagnostic-only scopes and native extraction counters to
the validated numerical implementation. Its final-code three-RHS run passes
at 109.967 seconds factor plus 42.0741 seconds solve under tracing. These
times are diagnostic measurements from a different allocation, and are not
pooled with the quiet performance trials above.

The remaining callback mean-rank wall is 53.87 seconds: 24.85 seconds child
native F22 extraction, 11.30 seconds current-Schur BF extraction, 8.09 seconds
STRUMPACK collectives/waits, 5.07 seconds host packing/unpacking, roughly
3.62 seconds other child/control work, and 0.94 seconds original sparse
extraction plus temporary allocation. Native counters further identify
13.54 seconds of child result redistribution and 5.37 seconds of current
Schur result redistribution. Together with STRUMPACK collectives this is
27.00 seconds, about half the callback wall, before host packing. Native
reconstruction plus V-row exchange accounts for 3.54 seconds in child
extraction and 4.60 seconds in current Schur extraction. GPU row-fetch/
traversal counters total 3.80 seconds across both paths. No counter resets
were observed. The counters include communication and waiting; they are not
pure CPU arithmetic, PCIe, or network-wire timings.

The explicit STRUMPACK collective means are 3.22 seconds metadata allreduce,
3.37 seconds parent exchange and 1.50 seconds child exchange. Their waits can
move when collectives are rearranged, so their sum is not a promised speedup.
Zero-flop records mostly belong to mixed collective groups with active work
on other ranks. Entirely zero-flop child groups account for only about
0.001 mean-rank native seconds. A rank must not skip a collective based on
its local empty or zero-flop work; actual collective emptiness would need
proof, and is not the main measured opportunity.

The next target is the repeated native-result/child-grid/parent-grid/panel
redistribution and packing chain. A device-output API that keeps the same
chain would leave its largest measured cost in place. First establish the
final panel ownership, then combine transfers and avoid intermediate
materializations. A future contract needs collective preparation outside callbacks,
bounded device-accessible child rows, explicit owner-rank translations,
MPI-free evaluation on the supplied stream, and fenced lifetimes. It must
retain original entries, both child contributions and the current Schur term.
An output owner outside a child's communicator cannot be inserted into that
child's process map. Preparation must translate child-local ranks explicitly
and coordinate parent transport on the parent communicator with its required
members. No full child matrix replication is required or proposed.
