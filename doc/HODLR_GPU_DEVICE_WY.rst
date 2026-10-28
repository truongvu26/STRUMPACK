Device WY efficiency follow-up
==============================

This follow-up targets the host WY fallback identified in
HODLR_GPU_QR_SVD_PROFILE.rst. It retains the communicator/shared-tree fix,
upstream multi-RHS correction, QR-based large-block SVD, host extraction
callbacks and child Schur contributions. Construction tolerances and
the public ButterflyPACK C/Fortran interfaces are unchanged.

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
