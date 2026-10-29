Interpreting the supplied screenshot commands
=============================================

The supplied 150-cubed and 200-cubed commands establish a single-RHS,
preconditioned-GMRES reference. The previous tolerance rerun used three RHS
and iterative refinement. These solve timings describe different workloads
and stopping metrics. Factor time also depends on the changed compression
thresholds and GPU construction options.

Command and default differences
-------------------------------

.. list-table:: Supplied arguments versus the previous optimized cohort
   :header-rows: 1

   * - Setting
     - Supplied commands on this checkout
     - Previous optimized cohort
   * - RHS
     - 1
     - 3
   * - Krylov solver
     - AUTO selects preconditioned GMRES
     - Explicit iterative refinement
   * - Solver relative / absolute tolerance
     - 1e-6 / 1e-10
     - 1e-6 / 1e-12
   * - Maximum iterations
     - 1000
     - 100
   * - HODLR relative tolerance
     - 1e-3
     - 1e-3
   * - Minimum separator / minimum front
     - 2000 / 1000000000
     - 8000 / 100000 at 150 cubed; 15000 / 45000 at 200 cubed
   * - HODLR leaf / LR leaf / BACA block
     - 256 / 5 / 16
     - 1024 / 4 / 64 at 150 cubed; 1024 / 5 / 64 at 200 cubed
   * - Rectangular neighbor count
     - 16
     - 0
   * - GPU mode
     - 1
     - 2
   * - GPU streams
     - Default 4
     - 1
   * - Exact-front CUDA, dense Schur, compact extraction, rectangular GPU, host panels
     - Default off
     - Enabled

The example initializes ``nrhs=1`` and treats its second argument as a
possible RHS count. Here that argument is ``--sp_compression``; conversion
to an integer yields zero and is clamped to one. Compressed AUTO solves
with one RHS select preconditioned GMRES. The supplied 100-cubed command was
not included, so its unspecified defaults are not inferred from this pair.

The two solvers' relative residuals have different meanings. The current
GMRES implementation applies the preconditioner on the left and reports an
Arnoldi/Givens estimate of ``||M(b-Ax)||_2 / ||Mb||_2``. Iterative refinement
reports the unpreconditioned global Frobenius residual over all three RHS.
Reported solution error and global componentwise scaled residual remain
separate accuracy measurements.

Compatibility and execution environment
---------------------------------------

``--sp_print_compressed_front_stats`` is supported; its work contributes to
outer factor time. ``--sp_print_root_front_stats`` is absent from this
checkout and is silently ignored by its parser. It is retained in the
literal command record and is not replaced with ``--sp_write_root_front``,
which writes matrix data.

The screenshot's thread environment, executable revision and GPU model were
not supplied. A label of 16 threads per rank and ``-c32`` alone do not
identify OpenMP versus BLAS workers. This checkout's HODLR GPU mode requires
``OMP_NUM_THREADS=1``. The controlled pilot therefore uses one OpenMP thread
and 16 pthread BLAS workers in both CPU and GPU cases, with the known IEEE
BLAS/native library identities. GPU-aware MPI and tcmalloc are common to all
controlled cases and are explicitly recorded.

The pilot uses literal ``-c32 --cpu-bind=cores`` with four ranks per node,
four GPUs per node, and a fresh allocation without the previous
``nomultithread`` hint. Actual physical-core and GPU bindings are probed.
This is the supplied command-line setup on the recorded current-code
environment; it does not establish the missing original environment.

Controlled 150-cubed pilot
--------------------------

.. list-table:: Current-code single-trial pilot, 150 cubed / 4 ranks / 4 GPUs
   :header-rows: 1

   * - Configuration
     - Factor (s)
     - Solve (s)
     - Factor + solve (s)
     - GMRES iterations
   * - Supplied CPU arguments
     - 288.371
     - 4.498
     - 292.869
     - 4
   * - Supplied GPU arguments
     - 284.906
     - 4.327
     - 289.233
     - 4
   * - Optimized GPU, one RHS
     - 28.395
     - 4.261
     - 32.656
     - 3

All three pilots pass the requested preconditioned-GMRES stopping criterion.
Job 60495194 supplies the CPU row; job 60495202 supplies both GPU rows. Both
allocations complete successfully and are released. No 200-cubed pilot was
run in this cohort. These are single-trial observations, not medians.

The binding probes confirm 32 logical SMT CPUs and 16 physical cores per
rank, with four disjoint physical-core masks. Thus the supplied binding and
our earlier ``-c16 --hint=nomultithread`` reach the same physical-core count
on these nodes. Each process has one visible GPU, OpenMP maximum one and 16 BLAS
workers. All executable/native/BLAS/allocator identities and runtime checks
pass; the numerical source and frozen build are unchanged.

The legacy GPU row does not reproduce the reported 67-second factor time.
It does execute GPU operations: the native API audit records 36 completed
FP64 GPU factor calls and 604 solve calls in mode 1, with positive returned
factor-device memory. The CPU row records zero such calls. The optimized
row records 8 factor calls and 48 solves in mode 2, plus 350390 fronts in
whole CUDA exact-subtree batches, with no CPU or allocation fallbacks.
Solve calls can occur during Schur construction as well as outer GMRES;
these counters are neither iteration counts nor kernel timings.

The reference thresholds produce 31 HODLR fronts; the optimized thresholds
produce 3. The legacy path retains host construction and CPU rectangular
operations. Its near-equal CPU/GPU factor times are consistent with those
shared costs. The optimized bundle changes those routes and front
classification together, so its gain is not attributed to one setting.

.. list-table:: Accuracy of each pilot
   :header-rows: 1

   * - Configuration
     - Relative preconditioned GMRES residual
     - Rank-zero-local relative solution error
     - Global componentwise scaled residual
   * - Supplied CPU arguments
     - 3.54697e-7
     - 1.67945e-7
     - 7.37561e-6
   * - Supplied GPU arguments
     - 1.46167e-7
     - 1.06259e-7
     - 2.25183e-5
   * - Optimized GPU, one RHS
     - 2.68514e-8
     - 2.14707e-8
     - 4.7335e-8

The optimized factor time of 28.3949 seconds is below the supplied
67-second reference at the same four-GPU count and one-RHS/GMRES workload.
The original source/build/thread environment remains unknown, so this is
a comparison with the reported number, not an exact hardware/environment
reproduction. The companion ``HODLR_GPU_SCREENSHOT_COMMANDS.json`` retains
commands, measurements, convergence traces, binding/runtime evidence and
native-GPU call counters. Logs, job folders and builds remain locally
ignored and are excluded from the commit.

The CPU and GPU reference rows use the supplied arguments, including all
GPU-extension defaults off. The optimized row keeps one RHS, AUTO/GMRES,
solver tolerances, maximum iterations, statistics output and CPU binding,
while applying the previously selected GPU configuration: exact-front CUDA,
host panels, rectangular GPU, compact extraction and dense Schur enabled;
GPU mode 2, one stream, thresholds 8000/100000, leaf 1024, LR leaf 4, BACA
block 64 and rectangular neighbor count zero. This is a configuration-bundle
comparison. Native heap fraction is its default 0.85 for the reference rows
and the tested 0.35 for the optimized row.

The reported screenshot reference at 150 cubed is CPU factor/solve
366/5.11 seconds (four GMRES iterations), and GPU factor/solve 67/5.74
seconds (five iterations). The supplied 200-cubed factor references are
1205/119 seconds; its solve row was truncated. The latter solve values are
left unspecified.
