Opt-in GPU update for exact local Schur complements
=================================================

Status
------

``--hodlr_gpu_dense_schur 0|1`` defaults to ``0``. Build, source review,
GPU fixtures and six matched 200-cubed sparse-solver trials passed. The
three-trial median factor time fell from 129.169 to 101.955 seconds on the
same 16 MPI / 16 A100 / four-node allocation. Factor plus solve fell from
168.0662 to 141.8683 seconds. See HODLR_GPU_OPTIMIZATION.rst for settings,
profiling, source hashes and the limits of those comparisons.

The current 200-cubed diagnostic run, job 60474379, measured 129.943 seconds
for factorization. Rank zero spent 18.176 local wall seconds in instrumented
exact dense Schur GEMMs, 4.262 seconds in exact dense TRSM, and 50.958 seconds
in host-panel preparation. These local counters motivate the narrow GPU
update; they are not an additive decomposition of global elapsed time.

Selection and build
-------------------

Enable the update with ``--hodlr_gpu_dense_schur 1`` in an existing HODLR
GPU sparse-solver command. It requires HODLR-family compression,
``--hodlr_use_gpu 1`` or ``2``, and ``OMP_NUM_THREADS=1``. The full solver
communicator initializes the existing ButterflyPACK GPU sharing path before
frontal traversal. The helper performs no MPI operations, device selection,
or STRUMPACK dense GPU backend initialization.

``STRUMPACK_USE_CUDA`` remains ``OFF``. A host-only C++ helper links CUDA
runtime and cuBLAS directly when the ButterflyPACK GPU interface is present.
Configuration compiles and links a CUDA-memory-query/cuBLAS-DGEMM probe;
``STRUMPACK_HAVE_HODLR_GPU_DENSE_SCHUR`` records the result. Installed static
targets carry the CUDA link requirements, and the existing package config
finds CUDAToolkit for ButterflyPACK GPU builds even with dense CUDA disabled.
CPU configurations omit the helper and reject attempts to enable its option.

Only real double precision local dense updates are eligible. The separator
must have at least 1000 entries, and ``2 * dim_upd^2 * dim_sep`` must be at
least ``1e11``. Smaller fronts, other scalar types, and dense MPI fronts
retain the existing CPU update. Complex-double HODLR GPU construction still
works through its existing path while these exact local updates remain on
CPU. Existing whole-backend restrictions, including unsupported FP32 HODLR
GPU mode, are unchanged.

Ownership and computation
-------------------------

Local dense assembly, child extend-add, pivoting, LU and both TRSMs run on
CPU. The helper replaces only ``F22 -= F21 * F12`` after those operations.
The full host ``F22`` remains its owner's child-contribution storage and
already contains the accumulated child Schur data.

The helper uploads ``F21`` once, then streams column panels of ``F12`` and
the corresponding current ``F22`` through packed device buffers. Each
cuBLAS DGEMM uses FP64, alpha ``-1`` and beta ``1``. Thus every output panel
includes its existing child contributions. Matrix leading dimensions may
be padded; only logical rows are copied. As with the original GEMM, output
storage must not alias either input.

Each panel is copied back to host and its nonblocking CUDA stream is fenced.
All output is ready before the helper returns, before work storage is
released, and before a parent extracts this child's contribution. No device
pointer is retained by the front or exposed to parent extraction.

Admission and errors
--------------------

One allocation holds packed ``F21`` and one panel each of ``F12`` and
``F22``. Its explicit footprint is bounded by 2 GiB and by currently free
device memory after reserving ``max(256 MiB, free_memory / 8)``. The panel
width is the largest admitted width, bounded by the number of output columns.
cuBLAS/context workspace and ButterflyPACK pools are outside this explicit
buffer count; the safety reserve leaves room for other device work.

Insufficient configured budget, insufficient free memory, or recoverable
``cudaMalloc`` out-of-memory returns a CPU fallback before any host matrix
is changed. The original complete CPU GEMM then runs once. Invalid shapes,
leading dimensions, null pointers for nonempty matrices, and impossible
strided host spans are rejected before CUDA access. A zero dimension is a
successful no-op and permits null pointers.

Other CUDA/cuBLAS resource or execution errors throw. Earlier output panels
may already have been committed, so the caller must not rerun the complete
CPU GEMM after such an error. The solver must propagate the failure. Resource
destruction fences pending operations and releases only helper-owned state.

Measurements
------------

Each attempted update reports use/fallback status, panel count and width,
host wall time, device GEMM event time, upload/download bytes and explicit
device-buffer peak. The profiling scopes are ``front.dense.GEMM`` and its
nested ``front.dense.GEMM.gpu_attempt`` scope.

Host wall time includes admission, allocation, handle/stream setup, transfers,
DGEMM, synchronization and cleanup. Device GEMM event time excludes panel
copies and host setup; GPU scheduling or other processes can affect elapsed
event time. Transfer counters count logical payload, not host padding.
Neither local event sums nor local rank timings are whole-device occupancy
or a globally additive CPU/GPU decomposition.

Validation scope
----------------

``test_HODLR_gpu_dense_schur`` compares against an independent long-double
``C0 - A*B`` reference with nonzero ``C0``. It exercises padded leading
dimensions, preserved inputs and output guards, multiple panels with an odd
tail, a default-budget shape, pre-mutation budget fallback, zero-dimensional
no-ops, invalid arguments and impossible host spans. The fixture passed with
one process and two independent processes sharing one GPU in job 60474930,
and again in the corrected 16-rank sparse comparison job 60474983.
The former job's sparse timing launch used an incorrect four-rank layout
and was stopped; those timings are excluded. Its standalone fixture results
remain valid. Use the immutable manifest hashes when attaching results.
