CUDA exact subtrees inside a HODLR elimination tree
===================================================

The opt-in ``--hodlr_gpu_dense_fronts 1`` selects STRUMPACK's existing
``FrontGPU`` backend for local exact fronts in a pure HODLR sparse solve.
The default is zero. Compressed fronts continue to use ButterflyPACK.
Exact fronts on more than one MPI rank retain ``FrontDenseMPI`` and
ScaLAPACK. This option accelerates exact-subtree assembly and factorization;
its solve factors and completed root Schur contribution remain on the host.

Why this changes performance
----------------------------

The old factory allowed STRUMPACK's dense CUDA backend only for uncompressed
or BLR solves. A HODLR tree consequently used CPU LU/TRSM for its exact
fronts, even in a CUDA-enabled build. The earlier bounded GPU Schur helper
replaced only the final GEMM. Matched 150-cubed and 200-cubed profiles found
large local CPU TRSM and assembly costs, and no completed distributed exact
fronts. Enabling SLATE would therefore not address those measured fronts.

The new route batches homogeneous local exact subtrees with the established
CUDA assembly, cuSOLVER LU/solve and cuBLAS Schur update. Thresholds can then
move additional single-rank HODLR heads into these exact subtrees. These are
two distinct changes: the controlled backend comparison retains the old
thresholds; the best tested configuration changes the thresholds as well.

Child contributions and ownership
---------------------------------

``FrontGPU::extract_CB_sub_matrix`` previously had an empty implementation.
It now adds entries from the completed root Schur contribution, using the
same update-index mapping as ``FrontDense``. This is a contribution only:
the parent separately assembles original sparse matrix entries. The bridge
handles partial intersections, duplicate and unsorted queries, pre-existing
output values and padded leading dimensions. Existing batched and owned
extraction wrappers use this implementation.

The GPU subtree already copies its root Schur block into owned
``host_Schur_`` storage at completion. Callback extraction adds no device
copy or collective. The parent retains that storage until its construction
finishes, then releases it. Host LU factors, pivots and coupling blocks
survive this release for the existing solve path. Thus parent fronts include
child Schur complements, rather than only original matrix entries.

A GPU node above a non-GPU or compressed child uses the existing split
traversal and the child's virtual factorization/extend-add methods. It does
not treat a HODLR child as a dense GPU front.

Build and supported selection
-----------------------------

Build against the same GPU-enabled ButterflyPACK package as the hybrid
interface, with the existing communicator and multi-RHS corrections. Enable
STRUMPACK CUDA and disable its MAGMA and SLATE alternatives. Representative
additional CMake settings are::

  -DSTRUMPACK_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DTPL_ENABLE_MAGMA=OFF -DTPL_ENABLE_SLATE=OFF \
  -DSTRUMPACK_USE_BPACK=ON -DSTRUMPACK_REQUIRE_BPACK_GPU=ON \
  -DSTRUMPACK_USE_HODLR_GPU_EVALUATOR=ON \
  -DSTRUMPACK_USE_HODLR_GPU_RECTANGULAR=ON \
  -DSTRUMPACK_USE_HODLR_GPU_HOST_PANELS=ON

The first validation uses CUDA 13.2, A100 40 GB devices, PBLAS block size 32,
one MPI rank per GPU, ``OMP_NUM_THREADS=1`` and 16 pthread OpenBLAS workers
per rank on disjoint physical CPU cores. An IEEE-compatible OpenBLAS build
is used, as in the preceding matched-layout report. ButterflyPACK's own
MAGMA dependency may remain linked; STRUMPACK's ``FrontMAGMA`` is disabled.

Selection requires pure HODLR element extraction, ``--hodlr_use_gpu 1|2``,
STRUMPACK GPU mode enabled, butterfly levels zero and a positive GPU stream
count. Unsupported build/configuration combinations throw an explanatory
error. Full-solver ButterflyPACK GPU sharing initialization remains before
numerical traversal. The new route has not been validated for multiple
ranks sharing a GPU, additional OpenMP workers or other GPU backends.

Memory pressure and fallback
----------------------------

Whole-subtree admission uses current free device memory, with a reserve of
``max(256 MiB, free/8)``. If the batch does not fit, traversal splits into
smaller subtrees. A split node estimates its own matrices, pivots and solver
workspace; it allocates all explicit device buffers before modifying host
factors or launching factorization. A node that cannot fit uses CPU LU and
the ``FrontGPU`` solve convention: transformed F12 and original F21. Large
eligible FP64 CPU fallbacks may reuse the earlier bounded GPU Schur helper
when ``--hodlr_gpu_dense_schur 1`` is set.

Allocation failure permits retry only before numerical work starts. CUDA
execution errors propagate; retrying CPU LU on partially modified factors
would be unsafe. The focused test forces pre-admission CPU fallback with
controlled device ballast. It does not inject a failed CUDA allocation or
kernel failure.
Library handle/stream creation and workspace-query failures retain the
existing backend error handling; pinned-host allocation failures propagate.
The fallback is not a recovery mechanism for every CUDA/library allocation.

The measured configurations use ``BPACK_GPU_HEAP_FRACTION=0.35``. This is a
tested setting, not a prerequisite proved by the measurements. The main
ButterflyPACK pool is allocated lazily on its first GPU matrix allocation;
local exact subtrees can complete before that allocation. The pool remains
reserved afterward and can compete with later/interleaved subtrees or a
subsequent factorization. GPU-aware MPI also reserves a separate exchange
arena. Admission must therefore use actual free memory rather than infer
availability from the configured fraction.

Reproduce the selected Poisson configuration
--------------------------------------------

Use three RHS, one stream, GPU mode 2, bounded host panels, rectangular GPU
operations, compact extraction and the dense Schur helper::

  testPoisson3dMPIDist N 3 \
    --sp_compression HODLR --sp_Krylov_solver refinement \
    --hodlr_use_gpu 2 --hodlr_butterfly_levels 0 --hodlr_quiet \
    --hodlr_gpu_dense_fronts 1 --sp_gpu_streams 1 \
    --hodlr_gpu_dense_schur 1 --hodlr_compact_extraction 1 \
    --hodlr_gpu_rectangular 1 --hodlr_gpu_host_panels 1 \
    --hodlr_rel_tol TOL --hodlr_knn_hodlrbf 0 --hodlr_knn_lrbf 0 \
    --hodlr_BACA_block_size 64 --hodlr_gpu_front_max_bytes 0 \
    --hodlr_gpu_panel_max_bytes 67108864 \
    --sp_rel_tol 1e-12 --sp_abs_tol 1e-12 --sp_maxit 100 \
    --sp_compression_min_sep_size MINSEP \
    --sp_compression_min_front_size MINFRONT \
    --hodlr_leaf_size LEAF --hodlr_lr_leaf LRLEAF

The layout-specific values are:

.. list-table:: Best tested threshold candidates
   :header-rows: 1

   * - Grid
     - MPI ranks / GPUs / nodes
     - MINSEP
     - MINFRONT
     - LEAF
     - LRLEAF
   * - 100 cubed
     - 4 / 4 / 1
     - 3500
     - 100000
     - 512
     - 5
   * - 150 cubed
     - 4 / 4 / 1
     - 8000
     - 100000
     - 1024
     - 4
   * - 200 cubed
     - 8 / 8 / 2
     - 15000
     - 45000
     - 1024
     - 5

Use ``TOL=1e-2`` or ``1e-5``. Rank wrappers must bind each process to a
distinct GPU and 16 physical CPU cores. The measured environment additionally
sets ``MPICH_GPU_SUPPORT_ENABLED=1``, ``BPACK_GPU_AWARE_MPI=1``, pthread BLAS
workers to 16 and preloads tcmalloc. Frozen commands, library identities,
rank affinities and runtime checks are recorded in the accompanying JSON.
These thresholds are Poisson/layout-specific; raising MINSEP without checking
front ownership can create distributed CPU exact fronts and lose the gain.
For example, the selected 200-cubed layout makes local fronts of about
10000 separator plus 30000 update rows exact, while a two-rank front of
about 10000 plus 40000 rows remains compressed through MINFRONT=45000.
The selected 150-cubed local roots have about 5625 plus 22500 rows.

Measurements
------------

The accompanying ``HODLR_GPU_EXACT_FRONTS.json`` records individual trials,
separate factor/solve times and the median of each trial's factor-plus-solve
sum. Factor time includes construction/compression. Solve time includes
iterative refinement for all three RHS. Ordering and unrelated startup are
outside those two reported phases. Instrumented stage profiles and
single-trial tuning screens are kept separate from three-trial confirmations.

.. list-table:: Independent confirmation, three trials per row (seconds)
   :header-rows: 1

   * - Grid / GPUs
     - Compression tolerance
     - Median factor
     - Median solve
     - Median paired factor + solve
     - Factor range
   * - 100 cubed / 4
     - 1e-2
     - 6.574
     - 4.037
     - 10.611
     - 6.528--7.449
   * - 100 cubed / 4
     - 1e-5
     - 8.909
     - 0.826
     - 9.740
     - 8.888--8.973
   * - 150 cubed / 4
     - 1e-2
     - 24.678
     - 21.395
     - 45.823
     - 24.429--27.268
   * - 150 cubed / 4
     - 1e-5
     - 29.861
     - 3.320
     - 33.181
     - 29.630--30.327
   * - 200 cubed / 8
     - 1e-2
     - 70.912
     - 55.116
     - 126.272
     - 69.159--71.156
   * - 200 cubed / 8
     - 1e-5
     - 111.944
     - 4.279
     - 116.301
     - 111.566--112.071

Jobs 60481820, 60481667 and 60481669 supply the 100-, 150- and 200-cubed
confirmation cohorts respectively. All 18 trials pass the same final
accuracy limits (relative error below 1e-8 and scaled residual below 1e-6).
The driver's reported solution error is a rank-zero-local Frobenius ratio;
the componentwise scaled residual is a global maximum. Refinement uses a
global Frobenius residual over all three RHS.
No trials are discarded. Each column has its own median: the paired-total
column takes the sum within each trial before taking its median, so it need
not equal the sum of the two phase medians.

For these tested configurations, 1e-2 minimizes factor time, while 1e-5
minimizes factor plus solve. The tighter setting takes two refinement
iterations at every size; the looser setting takes 10, 13 and 25--26 at 100,
150 and 200 cubed respectively. Treating compression tolerance as an
end-to-end solver setting matters more than factor time alone.

Every confirmation uses whole GPU batches, with no splits, CPU fallbacks
or allocation fallbacks. Maximum explicit exact-subtree arenas are about
2.76, 14.20 and 22.89 GiB for the three sizes. These exclude native pools,
CUDA context and library-internal allocations.

The controlled single-trial screens below change only exact-front backend
selection within each pair. ButterflyPACK remains on GPU in every row; the
CPU label refers to the exact fronts, not the whole sparse solver.

.. list-table:: Same executable and original thresholds, seconds (one trial)
   :header-rows: 1

   * - Grid / GPUs
     - Exact backend
     - Factor
     - Solve
     - Factor + solve
   * - 150 cubed / 4
     - CPU
     - 61.988
     - 27.113
     - 89.101
   * - 150 cubed / 4
     - CUDA
     - 40.959
     - 26.416
     - 67.375
   * - 200 cubed / 8
     - CPU
     - 125.103
     - 75.200
     - 200.303
   * - 200 cubed / 8
     - CUDA
     - 103.492
     - 70.130
     - 173.622

Jobs 60481073 and 60481075 then screen the larger exact subtrees and heap
fractions 0.35/0.5. Jobs 60481337 and 60481349 screen one/four streams and
both compression tolerances. Four streams provide no material improvement;
one stream and fraction 0.35 are selected for independent confirmation.
Neither the screen nor the finite threshold search establishes a global
optimum for arbitrary sparse matrices.

The screenshot gives factor times only (14.2, 67 and 119 seconds). Comparisons
use the corresponding 4/4/8 GPU counts; its GPU model, command and timing
boundaries were unavailable. A smaller factor number is a count-matched
comparison with the reported number, not a hardware-controlled reproduction.
The controlled exact-backend comparisons use the same executable and controls.

The subsequent three-trial rerun with ``--hodlr_rel_tol 1e-3`` and
``--sp_rel_tol 1e-6`` is recorded separately in
``HODLR_GPU_EXACT_RELAXED_SOLVE.rst`` and its JSON, including the achieved
accuracy and the requested solver stopping criterion.

Execution counters and tests
----------------------------

Each outer GPU subtree emits ``# HODLR exact dense subtree`` with actual
batch/split/CPU-fallback counts, explicit upload/download bytes and explicit
device peak. The local API wall times include preparation and library calls;
they are not kernel timings. Recursive splits accumulate into one summary.
The optional bounded-Schur fallback subcounters are already included in the
total bytes/peak and CPU-fallback wall time. Do not add them again. Device peak
excludes CUDA context, library-internal workspace and ButterflyPACK pools.
Separate ``backend_statistics()`` records still describe ButterflyPACK work.

Job 60480975 passes the focused fixture with 2 and 4 MPI ranks, unique GPUs,
both legacy and compact MPI extraction, an independent long-double Schur
oracle, forced pivoting, mixed child types, additive/padded/duplicate queries,
and independent three-RHS solves after parent release of child Schur storage.
Controlled low-memory admission reaches CPU fallback with zero device copies
or factor kernels. Full sparse Poisson trials exercise factory selection and
multiple fronts; the focused fixture alone does not establish sparse timings.

The test executable is ``test_HODLR_gpu_exact_front``; the optional
``--check_cpu_fallback`` ballast case requires exclusive use of a GPU per
process. This manual GPU test requires GPU-enabled ButterflyPACK, STRUMPACK
MAGMA OFF, rectangular GPU and host-panel support enabled, MPI thread level
``MPI_THREAD_MULTIPLE`` and exactly one visible CUDA GPU per process (not
only a selected current device). Run with two or four MPI ranks for the MPI
routing checks; one rank skips them. It is not registered as an automatic
CTest because it needs an allocation and site-specific rank/GPU binding.
A CUDA-disabled control build checks the default HODLR CPU sparse
path and rejection of the unsupported opt-in. Runtime validation is FP64;
the template build alone is not a complex-arithmetic runtime claim.

The next measured opportunity is the remaining host exact-front solve and
host-factor traffic. Persisting factors on the device needs a separate
ownership and lifetime design across ButterflyPACK work. This implementation
does not enable a general distributed GPU frontal evaluator or device-resident
exact solves.
