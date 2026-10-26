Hybrid HODLR GPU performance investigation
=========================================

Scope and provenance
---------------------

These measurements precede additional GPU feature implementation. The tested
hybrid path uses host extraction/compression and CPU rectangular Butterfly
operations, with ButterflyPACK GPU HODLR factorization and inverse applications.
STRUMPACK's CUDA dense-front backend is disabled.

The initial baseline uses the final hybrid Release executable from 872543e1
(SHA256 fa05f78a43f057c3da829a70cea2f4baafdf842aff7282498e6eefbd50ef77ad).
It has no internal task timers, Nsight, native HODLR verbosity, or BPACK_CHECK.
There are three fresh-process trials of each backend at each grid size,
alternating order, four MPI ranks, three RHS and one OpenMP thread. One rank
uses each of four A100-SXM4-40GB GPUs on a single node. Both backends use
HODLR tolerance 1e-4 and refinement tolerance 1e-10 with the same separator
thresholds and leaf size. The 40^3 and 60^3 problems have 24 and 113 HODLR fronts.

ButterflyPACK seeds adaptive CPU compression from clock/PID. Compressed ranks
and refinement work therefore vary between trials. These are measurements of
the current lifecycle, including first-use initialization, not comparisons of
identical stored factors. Ranges are observed min/max, not confidence intervals.
Neither larger problems, multi-node scaling, GPU mode 2 nor an asymptotic GPU
crossover are characterized by this experiment.

Initial baseline
-----------------

.. list-table:: Uninstrumented factor plus solve, seconds
   :header-rows: 1

   * - Grid
     - CPU median [range]
     - GPU mode 1 median [range]
     - GPU / CPU median
   * - 40^3
     - 4.689 [4.650, 4.737]
     - 6.527 [6.178, 7.001]
     - 1.392
   * - 60^3
     - 38.515 [38.117, 38.762]
     - 41.288 [40.537, 41.320]
     - 1.072

All twelve trials pass finite relative error <1e-8 and componentwise scaled
residual <1e-6. Report solve iteration counts alongside timings; different
refinement work prevents treating solve time alone as a kernel speed ratio.

Baseline capture job 60337119 completed all twelve trials and its CPU diagnostic,
then FAILED in its separately launched MPI-interposed Nsight GPU diagnostic.
The failed trace's timings and partial reports are excluded. It exhibited
negative native timings and a shared-node layout error under MPI interposition,
plus report-import failure on the home filesystem. Do not call this Slurm job
an overall PASS. The independent completed baseline trials remain valid.

Diagnostic retry job 60337584 PASSED (exit 0:0, 2m46s): CPU internal timers,
GPU internal timers without a tracer, and GPU CUDA/NVTX/cuBLAS tracing without
MPI interposition. Reports were imported on node-local storage and copied back.
All GPU diagnostics cover 113 native GPU factorizations. Communication evidence
comes from STRUMPACK's timers, not an MPI-interposed trace.

Measured phases
----------------

The profiler build enables STRUMPACK_TASK_TIMERS. The Poisson driver finalizes
them before MPI_Finalize. Serial Schur construction now uses the same timer
as MPI fronts, and serial F22 compression has its own category. These changes
compile out when task timers are disabled.

.. list-table:: Inclusive mean rank seconds, untraced diagnostics at 60^3
   :header-rows: 1

   * - Phase
     - CPU
     - GPU mode 1
   * - F11 HODLR construction
     - 1.696
     - 3.239
   * - F22 HODLR construction
     - 16.69
     - 17.59
   * - Rectangular F12/F21 construction
     - 16.82
     - 16.75
   * - Current-front Schur construction
     - 2.210
     - 1.974
   * - HODLR factor calls
     - 0.1465
     - 1.086
   * - Original sparse-entry extraction
     - 0.2203
     - 0.2190
   * - Child submatrix extraction, distributed envelope
     - 4.880
     - 4.900
   * - Schur-entry extraction
     - 5.262
     - 5.272

Rows are inclusive and overlap. In particular, extraction belongs inside
construction. Do not add them or label their rank maxima as critical-path
wall time. Diagnostics enable hodlr_verbose, which also enables native
construction checks, so their wall times are not baseline speed samples.

F22 construction and CPU rectangular compression dominate the measured HODLR
work. Evaluating only original sparse entries on the device targets a small
part of that work; F22 explicitly skips the original A(u,u) entries. A useful
device construction path needs child contributions and the current-front
Schur term, as specified in HODLR_GPU_DESIGN.rst. F12/F21 remain a separate
CPU bottleneck because the current GPU registration is for Bmatrix HODLR
objects, not rectangular BF handles.

CUDA evidence and the first optimization
----------------------------------------

The successful CUDA diagnostic records 8,069--9,122 kernels per rank. The
union of kernel, copy and memset intervals is only 0.109--0.124 seconds per
rank during 44.976 seconds of factor-plus-solve wall time. This is about
0.24--0.27 percent of that diagnostic wall time, NOT an SM occupancy metric
or a measurement of the quiet baseline's utilization.

Host cudaMallocHost calls take 2.99--3.44 seconds per rank, with 85--97 calls.
In contrast, per-rank copy durations total about 0.019--0.020 seconds HtoD and
0.0029--0.0032 seconds DtoH (roughly 404--427 MB and 40--44 MB respectively,
decimal units). Transfer volume alone does not explain the slow path.

Source inspection found that MetaBuilder reserves at least 64 MiB per image,
including builders owned by individual HODLR matrices, and lacked a destructor
for its raw pinned buffer and upload event. ButterflyPACK branch
perf/hodlr-metadata-pinning, commit cf51315, makes the initial granularity
64 KiB while retaining geometric growth, and fences the last upload before
releasing the buffer/event. It does not replace per-matrix state with shared
mutable metadata.

The metadata_lifetime.cpp CUDA regression uses GNU linker wrappers to count
allocations and events. It checks small-image allocation size, growth while
an upload may be in flight, data after metadata destruction and zero live
pinned buffers/events after each scope. Its original-header executable must
fail, and the fixed-header executable must pass. The production change also
affects H2's use of the same metadata helper; a full H2 performance study is
outside these HODLR measurements.

Same-allocation metadata A/B and sharing validation
--------------------------------------------------

Job 60338444 PASSED (exit 0:0, 7m47s). It ran three alternating trials of
original CPU, original GPU and fixed GPU on one allocation, verified that
the two executables resolve different ButterflyPACK libraries, then ran the
metadata regression, fixed GPU-sharing tests and a fixed-backend CUDA trace.
Both ordinary timing executables disable task timers and native verbosity.

.. list-table:: 60^3, three RHS, four ranks; median [observed range], seconds
   :header-rows: 1

   * - Backend
     - Factor
     - Solve
     - Factor plus solve
   * - Original CPU
     - 37.752 [37.750, 38.406]
     - 0.529 [0.430, 0.579]
     - 38.329 [38.182, 38.935]
   * - Original GPU
     - 41.286 [40.455, 41.795]
     - 0.627 [0.556, 0.638]
     - 41.842 [41.093, 42.422]
   * - Fixed GPU
     - 38.981 [38.478, 39.480]
     - 0.673 [0.569, 0.707]
     - 39.654 [39.185, 40.049]

The fixed GPU median factor-plus-solve time is 5.2 percent below the original
GPU median. It is still 3.5 percent above CPU. This is a measured improvement
in the tested medium regime, not a general CPU/GPU crossover or solve-speed
claim; refinement counts vary (CPU 6--8, original GPU 7--8, fixed GPU 7--9).

In the fixed-backend CUDA diagnostic, cudaMallocHost time is 0.148--0.159
seconds per rank, compared with 2.99--3.44 seconds in the original diagnostic.
Allocation counts match per rank (97/94/94/85), and the fixed trace records
92/89/89/80 cudaFreeHost calls. Those traces were separate diagnostic jobs;
the end-to-end A/B table above is the comparison controlled to one allocation.
The focused lifetime regression passes with 393,216 peak pinned bytes and
zero live pinned bytes or upload events after every tested metadata scope.
Its original-header counterpart reproduces the excessive initial allocation.

Two MPI ranks on each of two distinct GPU UUIDs also pass the 40^3, three-RHS
sparse solve (24 native GPU factorizations, relative error 2.27e-11 and scaled
residual 2.45e-9), plus the real/complex one-/three-RHS direct-interface tests.
This validates a fixed four-rank solver group on one node with host-staged
MPI and OMP_NUM_THREADS=1. It does not validate expanding a solver group,
changing GPU bindings after initialization or GPU-aware MPI.

What the evidence prioritizes
------------------------------

The first performance change is the scoped metadata allocation/lifetime fix.
A larger GPU entry-evaluation project should next address F22 construction,
child Schur data and rectangular compression, not merely implement a sparse
stencil on the device. HODLR_GPU_DESIGN.rst specifies the missing collective
preparation contract and ownership/stream rules.

The fixed-group sharing configuration now has numerical evidence. The other
limits remain explicit: concurrent OpenMP GPU calls need backend execution
contexts or an ordered scheduler; matvec construction needs a host-to-GPU
forward-matrix handoff; nonzero butterfly levels need an actual GPU HODBF
representation and kernels. Removing their validators is not an implementation.

Reproduction
-------------

Build a baseline with HODLR_TASK_TIMERS=OFF and the original ButterflyPACK
installation. Build the diagnostic driver with HODLR_TASK_TIMERS=ON and a
separate HODLR_STRUMPACK_BUILD=build_profile_hodlr. Both use
config_perlmutter_gpu_hodlr.sh with HODLR_SKIP_BPACK=1 and the desired
HODLR_BPACK_INSTALL. Create profile_results before submitting::

  mkdir -p profile_results
  HODLR_BASELINE_EXE=/absolute/path/to/baseline/testPoisson3dMPIDist \
    sbatch test/profile_hodlr_gpu_perlmutter.sbatch

For the metadata A/B run, build the fixed dependency separately using
HODLR_BPACK_SOURCE with build_butterflypack_hodlr_gpu.sh, and build its
STRUMPACK consumer in build_optimized_hodlr with task timers OFF::

  HODLR_BASELINE_EXE=/absolute/path/to/original/testPoisson3dMPIDist \
  HODLR_OPTIMIZED_BPACK_SOURCE=/absolute/path/to/fixed/ButterflyPACK \
    sbatch test/profile_hodlr_metadata_perlmutter.sbatch

Build metadata_lifetime.cpp twice with the original/fixed GPU_BACKEND include
path respectively, defining H2_HAVE_GPU and providing CUDA/MAGMA/MPI headers.
Link cudart and GNU --wrap for cudaMallocHost, cudaFreeHost,
cudaEventCreateWithFlags and cudaEventDestroy. Place the binaries at
build_gpu_regressions/metadata_lifetime_old and metadata_lifetime_new in the
fixed checkout, as used by the A/B script.

Export each successful gpu.rankN.nsys-rep to SQLite on local storage::

  nsys export --type=sqlite --output=/tmp/hodlr-nsys/rank0.sqlite \
    profile_results/JOB/gpu.rank0.nsys-rep

Repeat for ranks 1--3, then run test/summarize_hodlr_nsys.py with the result
folder and the local SQLite folder. CUDA API host durations, GPU interval
unions, whole-solver wall times and inclusive phase sums are different
measurements and must remain separate.
