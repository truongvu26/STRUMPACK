GPU HODLR efficiency follow-up
==============================

The subsequent parameter and layout search is recorded in
:doc:`HODLR_GPU_TUNING`.

Status: PASS. All completed experiments below passed sparse numerical
validation and exited 0:0. Earlier launcher/driver failures and partial
results remain explicitly recorded in HODLR_GPU_DEVICE_WY.json and are
excluded from the successful timing medians.

The changes retain the communicator/shared-tree fix, upstream multi-RHS
correction, QR-based large-block SVD, host extraction callbacks and child
Schur contributions. Compression tolerance is 1e-5, leaf size 128,
butterfly levels zero and application RHS count three. Solver OpenMP is
one, STRUMPACK dense CUDA is disabled, and optional residual certification
is disabled throughout these performance tests.

Device WY change
----------------

ButterflyPACK commit ab40497 replaces the zero-Householder-coefficient
host fallback with a device compact-WY application. For Q = I - Y T Y^H,
define D = diag(tau) and G = strict_upper(Y^H Y). The identity
(I + D G) T = D gives a unit-upper triangular solve, including zero tau.
One setup kernel builds the matrix and scales its RHS; existing MAGMA
TRSM completes the operation on the same stream. Existing workspace and
completion fences are reused. This removes the cubic host recurrence,
its round trips and an unconditional coefficient download. Public
C/Fortran interfaces and construction tolerances are unchanged.

Job 60465148 passed 168 independent real/complex WY checks, 288
recompression comparisons, 12 direct SVD checks and 24 captured-factor
replays. Large isolated fallback benchmarks also passed; their component
speedups exclude common setup and do not predict whole-solver speedup.
Corrected sparse job 60466018 passed 40 and 100 cubed with four ranks/two
GPUs. Its 100-cubed factor/solve times were 223.866/5.01999 s, relative
error 1.21013e-10 and scaled residual 2.60245e-8. This was an accuracy
check with a different layout, not a before/after speed comparison.

Matched 150-cubed result
-----------------------

Job 60467720 uses 32 MPI ranks, 16 A100-SXM4-40GB GPUs on four nodes,
sequential OpenBLAS and identical quiet Release STRUMPACK executables.
Only the native library differs. Before/after/after/before order supplies
two trials per variant within one allocation. Binary hashes, placement,
maximum HODLR ranks and numerical results are in the JSON/manifests.

.. list-table:: Median wall seconds
   :header-rows: 1

   * - Native library
     - Factor
     - Solve
     - Factor plus solve
   * - Previous QR-SVD baseline
     - 380.0015
     - 4.254525
     - 384.256025
   * - Device WY
     - 347.048
     - 3.990845
     - 351.038845

Factor speedup is 1.09495x; total speedup is 1.09463x. This is about
33 seconds saved, or 8.6 percent less elapsed time. The two new factor
times were 346.975 and 347.121 s. Compression remains randomized:
maximum ranks were 986/984 before and 987/983 after, with differing
refinement counts. This is a small controlled sample, not a statistical
performance guarantee.

Rank/GPU layout
---------------

Job 60464839 tested 16 ranks/16 GPUs, one rank per GPU, using the same
physical GPU set as the paired experiment. It passed with factor
440.910 s, solve 2.56395 s, relative error 1.27961e-9 and scaled residual
5.01743e-8. Factorization was 27.0 percent longer than the new 32-rank
median. Reducing ranks did not improve this workload. MPI rank count
also changes work distribution and ordering, so this does not isolate
context-sharing overhead alone. Four ranks can drive at most four GPUs
with the current one-device-per-rank backend.

CPU BLAS threading
------------------

The existing sequential OpenBLAS cannot gain workers by changing
OPENBLAS_NUM_THREADS. An isolated matching 0.3.30 pthread build retains
LP64, the libopenblas.so.0 SONAME, GNU Fortran runtime 5 and the ZEN target.
It passed 54 independent CPU GEMM/SVD checks. No shared installation or
WY comparison dependency was replaced.

Job 60467724 uses 100 cubed, four ranks/four GPUs on one node, 16 physical
cores available to each rank and solver OpenMP fixed at one. All rank
probes verified actual pthread counts, a single BLAS library, sufficient
nonoverlapping physical CPU affinity and stable GPU placement.

.. list-table:: BLAS workers; median wall seconds
   :header-rows: 1

   * - Workers per rank
     - Trials
     - Factor
     - Factor plus solve
   * - 1
     - 2
     - 182.878
     - 184.838865
   * - 4
     - 1
     - 180.177
     - 182.598960
   * - 8
     - 1
     - 175.913
     - 178.370860
   * - 16
     - 1
     - 176.040
     - 178.543370

Eight workers gave 1.03626x total speedup; 16 gave 1.03526x. These modest
changes do not explain the screenshot gap. Nonbaseline counts have only
one trial and sampled maximum ranks range from 602 to 612. Pthread BLAS
workers do not enable parallel frontal traversal or lift the GPU OpenMP
guard. The optional test wrapper probes library/worker/affinity settings
only when a BLAS override is selected; the default path is preserved.

Prepared-image budget
---------------------

Job 60468854 brackets one 1-GiB-cap trial with two 256-MiB-cap trials on
the same allocation. It uses 100 cubed, four ranks/four GPUs, eight pthread
BLAS workers and the same native WY library/executable. All solves pass.

.. list-table:: Prepared-image cap; wall seconds
   :header-rows: 1

   * - Cap per eligible image
     - Factor
     - Factor plus solve
   * - 256 MiB, median of two controls
     - 177.941
     - 180.22450
   * - 1 GiB, one trial
     - 166.431
     - 168.87693

The larger cap gives 1.06916x factor and 1.06719x total speedup in this
sample. Logged host-panel fallback events across the four rank logs fall
from 27 in each control to zero at 1 GiB. Images are still assembled on
the host from original entries and child Schur contributions. A larger
cap increases retained host/device inputs; this test does not establish
a safe universal default. The product default remains 256 MiB. The
existing selection is --hodlr_gpu_front_max_bytes 1073741824.

The profiling script accepts HODLR_PANEL_IMAGE_CAP and optional safe
HODLR_PANEL_RUN_LABEL values for separate cases on one allocation. It
records the actual cap and explicitly requests CPU/GPU resources for
each step. File-driven benchmark loops use --input=none to keep srun
from consuming their case-order input. Successful comparisons restarted
on fresh allocations after that driver fix.

Comparison with the supplied screenshot
--------------------------------------

.. list-table:: Available GPU factor timings, seconds
   :header-rows: 1

   * - Problem
     - Screenshot
     - Best measured here
     - Time ratio
   * - 100 cubed
     - 14.2
     - 166.431
     - 11.72x longer
   * - 150 cubed
     - 67
     - 347.048
     - 5.18x longer
   * - 200 cubed
     - 119
     - No current rerun
     - Not compared

Our 100-cubed case uses four ranks/four GPUs, eight pthread BLAS workers,
OpenMP one and a 1-GiB image cap. The 150-cubed case uses 32 ranks/16 GPUs,
sequential BLAS, OpenMP one and a 256-MiB cap. All timings include frontal
assembly/compression in sparse-solver factorization.

The screenshot labels 100/150 as four ranks times 16 threads and 200 as
eight ranks times 16 threads. The user confirmed that commands and the
missing hardware/timing details are unavailable. GPU count/model,
compression settings, code revision and factor-timing scope for its
``Ours`` rows remain unknown. The numerical time gaps are real comparisons
of the supplied values; they do not establish an identical-configuration
regression or identify a unique cause. PR158's 60-cubed first row is
independently documented as whole sparse factorization on four A100
40-GB GPUs: https://github.com/pghysels/STRUMPACK/pull/158.

Next substantial target
-----------------------

The results support continuing with frontal-entry assembly and child-data
extraction. BLAS worker count alone has little effect, and fewer MPI ranks
on the same GPUs was slower. The earlier profile identifies substantial
host-panel preparation, extraction and coordination; exact CPU caller
percentages remain limited by incomplete backtraces.

Construction input images also outlive construction in both STRUMPACK
and native GpuState. A future construction-finish API must detach the
native callback and free native image buffers before STRUMPACK drops its
owner; simply resetting the application pointer would be unsafe. F12/F21
preparation could be interleaved with compression/release to reduce
simultaneous inputs. These lifetime changes were audited, not implemented.

A device frontal evaluator remains the larger extension: it must assemble
original sparse entries plus every child Schur contribution with correct
front-index mappings, distributed ownership, stream ordering and buffer
lifetimes. The current change preserves the validated hybrid path.
