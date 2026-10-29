Hybrid GPU HODLR: validated layout comparison with IEEE LAPACK
==============================================================

All 24 full sparse-solver trials pass: four layouts, two compression
tolerances, and three trials per row. The 200-cubed factor medians remain
above the screenshot reference at both tolerances. Conditional controls are reported separately below. The best confirmed
200-cubed fast configuration uses GPU-aware MPI plus tcmalloc and remains
above the 119-second screenshot factor reference.

This record uses ``testPoisson3dMPIDist`` with three right-hand sides on
NVIDIA A100-SXM4-40GB GPUs. Every MPI rank has one GPU and 16 physical CPU
cores, with 16 pthread OpenBLAS workers and ``OMP_NUM_THREADS=1``. Runtime
probes verify the loaded BLAS library, actual worker count, nonoverlapping
physical cores on each node, GPU UUID binding, and thread-timeout settings.
The final baseline contains no numerical or runtime-error cases. Across
its 24 trials, maximum relative solution error is 3.06209e-10 and maximum
componentwise scaled residual is 8.98422e-10.

The companion ``HODLR_GPU_MATCHED_LAYOUTS.json`` retains the baseline,
completed controls, unrounded timings, numerical checks, settings, hashes
and source paths. Its ``baseline`` field comes from the collected baseline
artifact whose SHA-256 is::

   f41c6926cf2ad7ea0e45ce4812532b5e24cdd8be848c39113d30b02f200d02d9

Fixed baseline
--------------

The table reports STRUMPACK's factor and solve timings. Factor includes
frontal construction/compression and numerical factorization. Solve uses
iterative refinement, so its cost also reflects the number of refinement
iterations. Factor-plus-solve is formed within each trial and then
aggregated; it is not the sum of independent medians. The JSON preserves
the literal timing ranges, including variable refinement counts.

.. list-table:: Fixed baseline: seconds, median [minimum, maximum], three trials per row
   :header-rows: 1
   :widths: 12 13 13 20 20 22

   * - Problem
     - MPI / GPU / nodes
     - Compression tolerance
     - Factor
     - Solve
     - Paired factor + solve
   * - 100 cubed
     - 4 / 4 / 1
     - 1e-02
     - 12.171 [11.957, 13.637]
     - 4.319 [4.285, 4.434]
     - 16.605 [16.276, 17.923]
   * - 100 cubed
     - 4 / 4 / 1
     - 1e-05
     - 14.110 [14.098, 14.265]
     - 0.955 [0.939, 0.969]
     - 15.053 [15.049, 15.235]
   * - 150 cubed
     - 4 / 4 / 1
     - 1e-02
     - 65.617 [64.805, 67.428]
     - 26.429 [25.001, 27.500]
     - 92.428 [91.234, 93.117]
   * - 150 cubed
     - 4 / 4 / 1
     - 1e-05
     - 82.524 [82.333, 82.667]
     - 3.593 [3.447, 3.604]
     - 86.115 [85.937, 86.117]
   * - 200 cubed
     - 8 / 8 / 2
     - 1e-02
     - 133.359 [132.017, 134.874]
     - 76.577 [75.391, 78.024]
     - 208.750 [208.594, 212.898]
   * - 200 cubed
     - 8 / 8 / 2
     - 1e-05
     - 212.666 [210.558, 213.326]
     - 11.956 [4.028, 12.146]
     - 224.812 [214.586, 225.282]
   * - 150 cubed
     - 8 / 8 / 2
     - 1e-02
     - 40.490 [39.912, 41.581]
     - 14.138 [13.253, 14.881]
     - 54.833 [54.051, 55.371]
   * - 150 cubed
     - 8 / 8 / 2
     - 1e-05
     - 53.827 [53.827, 54.196]
     - 1.670 [1.633, 1.677]
     - 55.503 [55.497, 55.828]

Each row uses three trials from one allocation/GPU/core-placement cohort.
The execution order is loose, tight, tight, loose, loose, tight, where
loose is 1e-2 and tight is 1e-5 compression tolerance. Cohorts, layouts,
tolerances and executables are never pooled.

.. list-table:: Accepted full cohorts
   :header-rows: 1
   :widths: 20 45 35

   * - Job
     - Problem / MPI / GPUs / nodes
     - Cases
   * - 60478697
     - 100 cubed / 4 / 4 / 1
     - 6 PASS: 3 per tolerance
   * - 60478521
     - 150 cubed / 4 / 4 / 1
     - 6 PASS: 3 per tolerance
   * - 60478404
     - 200 cubed / 8 / 8 / 2
     - 6 PASS: 3 per tolerance
   * - 60478749
     - 150 cubed / 8 / 8 / 2
     - 6 PASS: 3 per tolerance

Count-matched screenshot comparison
-----------------------------------

The comparison below matches problem size and the reference GPU counts
recorded for this study: four GPUs at 100 and 150 cubed, eight at 200 cubed.
The screenshot's GPU hardware, full command, compression tolerance and
factor timing boundary are unavailable. These are comparisons of reported
factor numbers only; they do not establish a controlled speedup. The
150-cubed/eight-GPU layout is excluded because its GPU count differs from
the four-GPU reference.

.. list-table:: Reported factor comparison at the reference counts
   :header-rows: 1
   :widths: 20 15 22 20 23

   * - Problem / GPUs
     - Tolerance
     - Measured factor median, s
     - Reference factor, s
     - Reported difference
   * - 100 cubed / 4
     - 1e-02
     - 12.171
     - 14.2
     - 14.29% lower
   * - 100 cubed / 4
     - 1e-05
     - 14.110
     - 14.2
     - 0.63% lower
   * - 150 cubed / 4
     - 1e-02
     - 65.617
     - 67.0
     - 2.06% lower
   * - 150 cubed / 4
     - 1e-05
     - 82.524
     - 67.0
     - 23.17% higher
   * - 200 cubed / 8
     - 1e-02
     - 133.359
     - 119.0
     - 12.07% higher
   * - 200 cubed / 8
     - 1e-05
     - 212.666
     - 119.0
     - 78.71% higher

The 100-cubed/1e-5 factor range crosses the 14.2-second reference, and the
150-cubed/1e-2 range crosses 67 seconds. Their lower medians do not imply
that every trial is lower. Both 200-cubed rows remain above 119 seconds,
which motivates the conditional controls below. A stricter compression
tolerance can increase factor cost while reducing refinement cost; the
paired totals show that tradeoff and should be retained alongside factor.

Reproducible numerical settings
-------------------------------

All accepted cases enable both optimization flags,
``--hodlr_gpu_dense_schur 1`` and ``--hodlr_compact_extraction 1``. Common
settings are ``--hodlr_use_gpu 2``, butterfly levels 0, rectangular GPU and
host-panel paths enabled, BACA batch 64, KNN counts 0/0, full-front image
cap 0, host-panel cap 67108864 bytes, QR SVD, GPU heap fraction 0.85 and
BACA residual guard 0. Both compression tolerances retain solver relative
and absolute tolerances 1e-12 and a maximum of 100 refinement iterations.
``OPENBLAS_THREAD_TIMEOUT=28`` is fixed. The baseline has STRUMPACK's own
CUDA dense-front backend disabled; ButterflyPACK owns GPU HODLR work and
the separate exact local GPU Schur helper remains enabled.

.. list-table:: Per-grid numerical settings, shared by both tolerances and all layouts
   :header-rows: 1
   :widths: 12 25 25 20 18

   * - Grid
     - Minimum separator
     - Minimum front
     - HODLR leaf size
     - RecLR leaf
   * - 100
     - 3500
     - 100000
     - 512
     - 5
   * - 150
     - 20000
     - 26000
     - 1024
     - 4
   * - 200
     - 8000
     - 100000
     - 1024
     - 5

The 150-cubed/four-GPU and eight-GPU layouts use the same numerical
settings. The sparse matrix and every child Schur contribution remain
part of frontal entry evaluation.

Confirmed LAPACK diagnosis and recovered fallback
-------------------------------------------------

The earlier ``DLASCL`` argument-4 messages came from the local CPU LAPACK
``DGESDD`` divide-and-conquer SVD used during sequential-front randomized
Schur compression. The trace reaches ``DLASD3`` normalization of its
internal vector: ``RHO=DNRM2(K,Z,1)``, followed by
``DLASCL(CFROM=RHO,CTO=1)``. The observed invalid ``CFROM`` values are NaNs,
not zero input matrix norms.

Job 60478322 captures eight finite, nonzero input matrices across four
ranks. Each ``DGESDD`` call returns ``INFO=1``; the existing ButterflyPACK
``gesvd_robust`` fallback calls ``DGESVD`` on the preserved original input
and returns ``INFO=0`` with finite singular values and vectors in all eight
cases. The diagnostic sparse solve exits successfully and converges.
Thus the runtime diagnostic is distinct from an unrecovered solver NaN.
Those diagnostic/old-library runs remain excluded from timing baselines.

The frozen pthread OpenBLAS 0.3.30 library had been compiled with
``-Ofast``. A same-source, same LP64/ZEN/pthread configuration built with
``-O3 -fno-fast-math`` resolves the exact captured-matrix failure. Fresh
processes replay two matrices, 22350-by-129 and 22350-by-257, at one and
16 BLAS workers for each library. The original library fails all four
``DGESDD`` replays; the IEEE build succeeds in all four. ``DGESVD``
succeeds for both libraries. Maximum full reconstruction relative
residual across successful SVDs is 4.827e-15; U and VT orthogonality
Frobenius errors are below 4.94e-14. Independent strict long-double
sample checks agree. Fresh-process MXCSR is 8064 for both builds, with
FTZ and DAZ disabled. Separate GEMM/LU checks pass at one and 16 workers.

This isolates the BLAS build difference, rather than a particular compiler
transformation. It does not identify a GPU input-value defect and is
separate from the historical upstream multi-RHS correction. Diagnostic
traces, exact input hashes and CPU replay evidence are summarized in
``profile_results/dlascl_trace_v2/study_report_v1.json``. The full baseline
above reruns the sparse path with the IEEE library and passes all 24 cases.

Provenance and excluded attempts
--------------------------------

The final harness records source revision
``51ed30e6f47e411cfb8cd1760a9d0a2b07ad3488`` and immutable executable
snapshot ``profile_results/compact_extract_snapshot_v1``. Frozen creation
metadata and recorded source-file hashes are preserved. Runtime manifests
and each rank's loaded-library probe identify the actual baseline
artifacts. SHA-256 values are::

   STRUMPACK executable:
   ec93fbe4da78b7ffa6da1b7dd16c000cc3f8d8f03cc9ded9de984b6d74c9be81
   ButterflyPACK native library:
   66ebd85ba461d64bc840711661ad71ae400ec93fcea2677dcaa360b6be569bae
   IEEE pthread OpenBLAS library:
   0df6859d4f84c4e2fd7a3a768d6ffe03a96d76159ca0f662487b087353a729ee
   Original Ofast OpenBLAS, excluded from this baseline:
   25181e6b2e566ee1e0ea4e0d0b6981b606e9527cc225289bf4e737c9cf25c8e3

The IEEE library is explicitly preloaded from
``profile_results/openblas_ieee_snapshot_v1/lib/libopenblas.so.0.3``.
The early frozen parent provenance copies for jobs 60478396 and 60478404
retained the original BLAS hash in the generic field. Their case binary
hashes, runtime manifests and loaded-library probes record the actual IEEE
hash above; the metadata erratum preserves that distinction without
rewriting historical artifacts.

Job 60478396's first tight 150-cubed/four-GPU attempt failed during
``MPI_Init_thread`` with a PMI mmap/bootstrap-barrier failure and launcher
exit 15, before solver factorization. It is an excluded infrastructure
attempt, not a numerical failure or timing sample. The accepted complete
150-cubed/four-GPU cohort is job 60478521; its trials are not pooled with
the partial earlier allocation. Original-library jobs 60477881 and
60478174, canceled/unfinished attempts, and instrumented diagnostic jobs
are also excluded from the fixed table. Their evidence remains retained
separately.

Conditional controls
--------------------

Every completed control uses three RHS and both optimization flags set to
one. Control observations use their own allocations and executables;
they are separate from the 24 baseline trials. All reported controls
passed the full sparse solve and numerical bounds. Single pilots remain
single observations.

GPU-aware MPI and tcmalloc
~~~~~~~~~~~~~~~~~~~~~~~~~~

The combined setting retains the baseline executable, native library,
IEEE pthread BLAS, and numerical options. It enables
``MPICH_GPU_SUPPORT_ENABLED=1`` and ``BPACK_GPU_AWARE_MPI=1`` and preloads
``libtcmalloc_minimal.so.4.6.5``. Actual target-process entry and exit audits
verify the MPI flags and the origin of ``malloc``. Before each study, all
eight ranks pass real CUDA-buffer broadcast, allreduce and inter-node
point-to-point tests at lengths 1, 257 and 65539 with independent numerical
oracles. This establishes correct device-buffer MPI operation.

.. list-table:: 200 cubed, 8 MPI / 8 GPUs / 2 nodes; seconds
   :header-rows: 1

   * - Study / tolerance
     - Trials
     - Factor
     - Solve
     - Paired factor + solve
   * - Initial pilot / 1e-02
     - 1
     - 125.327
     - 75.975
     - 201.302
   * - Initial pilot / 1e-05
     - 1
     - 199.992
     - 24.264
     - 224.256
   * - Confirmation / 1e-02
     - 3
     - 128.553 [126.792, 128.857]
     - 74.836 [74.111, 75.635]
     - 202.664 [201.628, 204.492]

The three confirmation trials in job 60479285 reduce median factor time
by 3.60% and paired factor-plus-solve by 2.92% against the corresponding
three-trial baseline. All three use 38 refinement iterations. The initial
pilot job 60478857 is retained separately and is not pooled into that
median. The combined setting does not isolate allocator effects from
GPU-aware MPI effects, or establish zero-copy network transport.

The conservative pilot takes 12 iterations, compared with 2, 6 and 6 in
the conservative baseline. Its 24.264-second solve is 2.022 seconds per
iteration, within the baseline's approximately 1.993-2.024 seconds per
iteration. Its larger raw solve time therefore does not demonstrate a
transport regression. The confirmed fast factor median remains 8.03%
above the count-matched 119-second screenshot value.

The tcmalloc SHA-256 is::

   ef95b6a20cebd14fba276a440167a443840f82e3a2fe40427a2842df471f3eda

STRUMPACK CUDA-ON build
~~~~~~~~~~~~~~~~~~~~~~~

The CUDA-ON control is compiled from archived commit ``51ed30e6`` with
CUDA 13.2, architecture 80, and MAGMA/SLATE/KBLAS disabled. Its generated
configuration differs from the baseline by ``STRUMPACK_USE_CUDA``; all
required ButterflyPACK GPU capability checks pass. Runtime still uses
native library ``66ebd85b...`` and IEEE BLAS ``0df6859d...``.

.. list-table:: CUDA-ON pilots; seconds, one trial per row
   :header-rows: 1

   * - Problem / MPI / GPUs / nodes
     - Tolerance
     - Factor
     - Solve
     - Paired factor + solve
   * - 150 cubed / 4 / 4 / 1
     - 1e-02
     - 64.965
     - 25.666
     - 90.632
   * - 150 cubed / 4 / 4 / 1
     - 1e-05
     - 80.963
     - 3.517
     - 84.481
   * - 200 cubed / 8 / 8 / 2
     - 1e-02
     - 129.709
     - 72.863
     - 202.572
   * - 200 cubed / 8 / 8 / 2
     - 1e-05
     - 209.705
     - 4.060
     - 213.765

Jobs 60479114 and 60479090 pass at both tolerances. These are build and
initialization controls. ``FrontFactory.hpp`` still selects CPU exact
fronts for HODLR compression, together with the existing optional GPU
Schur helper. Enabling this build does not move exact-front LU/TRSM to
the separate GPU dense-front implementation. The conservative 200-cubed
pilot uses two refinement iterations, so its raw solve difference also
reflects iteration count. The single pilots do not establish a
three-trial performance improvement.

The CUDA-ON executable SHA-256 is::

   5c18310179044b79df057629dfad8ddaad522c55d2f219e81f846541b5ccf319

CPU factor OpenMP and BLAS configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first private control retained pthread BLAS and serialized native GPU
transactions. Its one-thread control passes. The two-thread CPU factor
case reaches an actual team of two, but produces hundreds of thousands
of OpenBLAS warnings per rank and no completed factor timing. Job
60478954 was canceled before repeating that incompatible path at four
threads. The interrupted case has no qualifying performance result.

A separate aligned control uses IEEE OpenBLAS built with
``USE_OPENMP=1`` and ``NUM_PARALLEL=16``. Its CPU fixtures validate 16
concurrent matrix multiplications against a long-double oracle, effective
BLAS queue one, OpenMP runtime mode two, and phase/thread restoration.
The full 150-cubed trial verifies actual CPU factor team 16, BLAS one,
serial native GPU entries, and serial reorder and entire solve with
``params::num_threads=1`` and task cutoff zero. Environment OMP remains
one; maximum active OpenMP levels is one. Each rank has one visible GPU
and a 16-physical-core affinity mask. ``MPI_THREAD_MULTIPLE`` is required.

.. list-table:: Private 150 cubed / 4 MPI / 4 GPUs / 1 node / 1e-05; seconds
   :header-rows: 1

   * - CPU factor / BLAS configuration
     - Trials
     - Factor
     - Solve
     - Paired factor + solve
   * - One factor thread / pthread BLAS 16
     - 1
     - 81.851
     - 3.516
     - 85.367
   * - 16 factor threads / OpenMP BLAS 1
     - 1
     - 96.147
     - 4.874
     - 101.020

Job 60479491 passes the full three-RHS numerical check without the earlier
warning storm, but is slower than the one-thread/pthread control. This
changes CPU scheduling and BLAS configuration, including native host work
and solve BLAS. It does not isolate factor-team width or establish general
``OMP_NUM_THREADS>1`` hybrid-backend support. Production source and its
existing thread guards remain unchanged; the private source patch,
fixtures and executable are retained as ignored experimental artifacts.

The aligned executable and OpenMP BLAS SHA-256 values are::

   d21553f8ab11e243639f636f65e7585da6275a8bf48d639f9bf4f9c571dae0fc
   782c56c0afd66431b0e4652d8e3282fd266a9de57da505414d9253855f0e4f12

Artifact handling
-----------------

Only this report and its companion JSON are intended for commit. The raw
``*.log`` files, job directories, ``build_*`` directories, libraries,
binaries, source archives and experimental harnesses remain in locally
ignored locations. Numerical implementation files are unchanged by this
benchmark study.
