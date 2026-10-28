Device WY efficiency follow-up
==============================

This follow-up targets the host WY fallback identified in
HODLR_GPU_QR_SVD_PROFILE.rst. It retains the communicator/shared-tree fix,
upstream multi-RHS correction, QR-based large-block SVD, host extraction
callbacks and child Schur contributions. Construction tolerances and
the public ButterflyPACK C/Fortran interfaces are unchanged.

Component validation passed in job 60465148: 168 independent real/complex
WY checks, 288 recompression comparisons, 12 direct SVD checks and 24
captured-factor replay comparisons. The same job's sparse stage failed
before solver startup because of the shared-allocation launcher. Corrected
CPU/GPU requests and physical-ID-to-UUID binding are in the test wrapper.
Sparse retry 60466018 completed successfully: 40-cubed factorization
8.12944 s, solve 0.0897196 s, relative error 6.73939e-12; 100-cubed
factorization 223.866 s, solve 5.01999 s, relative error 1.21013e-10 and
scaled residual 2.60245e-8. Both use four ranks, two GPUs and three RHS.
These are functional checks, not a matched performance comparison.

ButterflyPACK commit ab40497 applies compact WY on the device for all
Householder coefficients, including zeros. For ``Q = I - Y T Y^H``, define
``D = diag(tau)`` and ``G = strict_upper(Y^H Y)``. The identity
``(I + D G) T = D`` gives a unit-upper triangular solve, avoiding the
previous host recurrence, downloads and uploads. One setup kernel builds
the matrix and scales the right-hand sides, followed by the existing
MAGMA TRSM on the same stream. Existing workspace and completion fences
are reused.

Validation and timing protocol
------------------------------

``HODLR_GPU_DEVICE_WY.json`` records current job status and binary hashes.
Validation compares against an independent host recurrence, checks native
recompression and captured frontal factors, and runs full 40- and
100-cubed sparse solves. A two-GPU shared allocation is used for these
checks; it is separate from the 150-cubed performance allocation.

The paired 150-cubed test uses the identical quiet Release STRUMPACK
executable, 32 ranks, 16 A100 40-GB GPUs on four nodes, three RHS, one
OpenMP/OpenBLAS thread, leaf size 128 and compression tolerance 1e-5.
Only the native library differs. Residual certification is disabled.
Before/after/after/before order gives two trials per variant on one
allocation. Numerical checks and placement/binary hashes must pass for
all four runs before median timing ratios are emitted. This remains a
small performance sample, with variable randomized compression work.

The isolated WY benchmark includes required device/host transfers for
the legacy host method and setup/TRSM/completion for the device method.
It excludes common input creation, metadata upload and reset copies.
Its speedup must not be substituted for sparse-solver speedup.

A separate 150-cubed case uses 16 ranks with one rank per GPU on the
same number and type of devices. This tests MPI distribution separately
from the source change. Four ranks can drive at most four GPUs with the
current one-device-per-rank backend; that layout is a different resource
comparison from a 16-GPU run.

Comparison with the supplied screenshot
--------------------------------------

The screenshot reports 45.2/14.2 s CPU/GPU factorization for 100 cubed,
366/67 s for 150 cubed and 1205/119 s for 200 cubed. Its layouts are
four ranks times 16 threads for 100/150 and eight ranks times 16 threads
for 200. GPU type/count, timing boundaries, numerical settings and code
revisions were not supplied. These are externally supplied timings,
not independently reproduced results.

The new four-rank/one-thread/two-GPU 100-cubed functional check is
15.77x slower than the screenshot's 14.2-second GPU number. The pre-WY
32-rank/one-thread/16-GPU 150-cubed result is 5.67x slower than its
67-second number; the 838.071-second CPU baseline is 2.29x slower than
366 seconds. The report's 2.19x internal GPU speedup does not establish
performance parity with the screenshot. A matched post-WY 150-cubed
result is pending.

The current linked OpenBLAS is sequential, so increasing its thread
environment variable alone has no effect. An isolated matching-version
0.3.30 pthread build passed 54 independent CPU GEMM/SVD checks at one,
four and eight BLAS threads with OpenMP fixed at one. It retains the
same LP64 ABI, SONAME and Fortran runtime. This prepares a separate
threading experiment; it is not substituted into the WY library pair.

Next measured target
--------------------

Host-panel preparation remains the larger architectural target in the
baseline profile. ButterflyPACK's child-factor row extraction rebuilds
requested-row caches on each call. A subsequent bounded cache would need
factor-generation invalidation, explicit host/device ownership and a
memory cap, while keeping every rank in the required collective order.
Metadata packing is a smaller candidate: the baseline records 89.05
aggregate rank-seconds in metadata work versus 4662.96 in host preparation.
The current change keeps the validated one-thread configuration; enabling
concurrent fronts requires separate Context/DeviceHeap and MPI-thread
validation.
