Bounded cuSOLVER resources for GPU HODLR
======================================

ButterflyPACK branch ``fix/gpu-cusolver-resource-reuse`` fixes the resource
growth identified by the 150-cubed profile. Implementation commit a22b149
is based on ae6e81a, retaining the rectangular GPU/Schur path and the
communicator-safe integration. STRUMPACK's numerical executable is unchanged,
including its upstream multi-RHS correction.

Previously each retained construction, SVD and distributed-QR object owned
private cuSOLVER handles and parameter objects. With two ranks per 40 GB
GPU, the default 85% device-pool reservation left insufficient external
device memory as front objects accumulated. The original implementation
failed with ``cusolverDnCreate failed with status 7``.

``CuSolverLease`` now shares one handle and one default parameter object
per process on the existing backend device/stream, across real and complex
operations and across matrices. A mutex protects the complete workspace
query/execution sequence. All device work remains ordered on the same
stream; no additional device synchronization or MPI collective is added.
The resource has the same process lifetime as the MAGMA queue. Matrix
destructors release their own data without invalidating other matrices'
cuSOLVER resource. This does not enable concurrent OpenMP construction.

Validation
----------

Focused job 60403880 passed on 40 GB A100 GPUs with two ranks per GPU and
the default ``BPACK_GPU_HEAP_FRACTION=0.85``:

* 130 simultaneously live real/complex construction states per rank, with
  no measurable additional external device-memory growth after warm-up.
* Exactly one cuSOLVER handle and one parameter object created per process,
  verified by an LD_PRELOAD audit of the actual native creation functions.
* Real/complex N/T/C rectangular multiplication and Schur products, padded
  one/three RHS, moves, three/four ranks and strict subcommunicators.
* Child-inclusive HODLR evaluation and a full 40-cubed three-RHS sparse solve
  (relative error 8.93456e-12, scaled residual 3.17358e-10).

The same many-live-states stress against the unpatched library reproduced
``cusolverDnCreate`` status 7. The expected failing control is preserved
separately from the successful fixed-library tests.

The full-size workload uses 150 cubed, 3,375,000 unknowns, three RHS, 1,980
HODLR fronts, four nodes, 32 ranks and 16 explicitly selected 40 GB A100s.
The pool fraction is 0.85; the entry-image cap remains 256 MiB. Existing
STRUMPACK baseline and instrumented executables are loaded with a frozen
copy of the new ButterflyPACK shared library, SHA-256
``5bf529871090ef85494eec39861bdcfe7d856f3b83b2f1f34b4e9db10db6d0a6``.

Diagnostic job 60403959 completed the solver successfully: factorization
717.034 seconds, solve 7.93349 seconds, relative error 3.72815e-10 and scaled
residual 2.673e-8. All 32 ranks passed numerical phase accounting and
created exactly one handle and one parameter object. Its batch status is
FAILED because the report parser did not strip the inventory's ``MiB``
suffix. The parser was corrected and the saved data reanalyzed successfully;
the original batch status and error log remain preserved.

Quiet job 60405138 completed successfully on 40 GB GPUs with the default
85% pool: factorization 700.746 seconds, solve 4.84832 seconds and total
705.59432 seconds. It used nine refinement iterations; relative error was
7.40232e-10 and scaled residual 4.50852e-8. The audit confirmed exactly one
handle and parameter object, with no creation failures, on all 32 ranks.
Thus the full sparse solve no longer requires the 50% pool workaround.
The earlier unpatched/workaround run measured 718.45295 seconds total;
these are separate single trials, not evidence of a statistically
established performance improvement from resource reuse.

An early quiet allocation on 80 GB GPUs (60403957) was canceled when the
hardware mismatch was identified. It is excluded from numerical and timing
claims. Future profiling submissions explicitly request ``gpu&hbm40g`` and
include the actual GPU models in their reports.

Performance direction
---------------------

The resource fix removes the need for the 50% pool workaround; it does not
change the dominant construction algorithm. The new diagnostic profile
still spends 58.1% of exclusive mean rank factor time in host-callback
compression: 34.2% in F22, 12.7% in F12 and 9.8% in F21. Device-evaluator
construction is 7.1%, entry preparation 3.9%, and the setup/traversal/wait
remainder 28.5%. These are rank wall-time budgets including waits, not pure
CPU/GPU execution times or a global critical path.

The next extension is bounded distributed evaluation of large F22 pieces,
then the large rectangular pieces. The design in
``HODLR_GPU_TILED_EVALUATION.rst`` specifies collective preparation windows
before local callbacks, owner-local device leases, child/current-Schur
contributions, front-index mappings, stream fences, release ordering and
acceptance tests. It is a design, not an enabled implementation. The resource
fix does not lift the image cap or claim distributed tiled evaluation.

Machine-readable provenance and results are in ``HODLR_GPU_CUSOLVER_REUSE.json``.
Raw logs, job folders, shared-library snapshots and builds remain locally
ignored and are excluded from commits.
