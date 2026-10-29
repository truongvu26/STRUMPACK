GPU exact-front Poisson rerun with the requested tolerances
===========================================================

This rerun sets ``--hodlr_rel_tol 1e-3`` and ``--sp_rel_tol 1e-6`` on the
validated CUDA exact-front implementation at commit ``9ac9e512``. It uses
three independent trials for each previous layout: 100 cubed and 150 cubed
on four MPI ranks / four GPUs / one node, and 200 cubed on eight ranks /
eight GPUs / two nodes. Each solve has three RHS.

Only the two requested tolerances change. In particular,
``--sp_abs_tol 1e-12`` and ``--sp_maxit 100`` remain fixed. There is no
numerical source change or rebuild; the source hashes and executable/library
identities are checked against the previous validation snapshot.

Configuration
-------------

All rows use ``--hodlr_gpu_dense_fronts 1``, ``--hodlr_gpu_dense_schur 1``,
``--hodlr_compact_extraction 1``, ``--hodlr_gpu_rectangular 1``,
``--hodlr_gpu_host_panels 1``, ``--hodlr_use_gpu 2``, butterfly levels zero,
one GPU stream, heap fraction 0.35, one OpenMP thread and 16 pthread BLAS
workers per rank on disjoint physical CPU cores. GPU-aware MPI and tcmalloc
remain enabled. GPU binding assigns exactly one GPU to each rank.

The unchanged layout-specific compression thresholds are:

.. list-table:: Thresholds and leaf settings
   :header-rows: 1

   * - Grid
     - Ranks / GPUs / nodes
     - Minimum separator
     - Minimum front
     - Leaf
     - LR leaf
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

Timings
-------

.. list-table:: Three-trial medians, seconds
   :header-rows: 1

   * - Grid / GPUs
     - Factor
     - Solve
     - Paired factor + solve
     - Paired range
     - Iterations
   * - 100 cubed / 4
     - 7.118
     - 0.825
     - 7.946
     - 7.916--8.771
     - 2, 2, 2
   * - 150 cubed / 4
     - 28.064
     - 3.292
     - 31.330
     - 30.988--31.400
     - 2, 2, 2
   * - 200 cubed / 8
     - 77.385
     - 4.265
     - 81.774
     - 81.650--83.642
     - 3, 2, 2

Jobs 60493522, 60493384 and 60493336 supply the 100-, 150- and 200-cubed
cohorts respectively. All nine requested-tolerance trials pass, with actual
GPU exact-front work and no splits, CPU fallbacks or allocation fallbacks.
Every rank has a unique GPU and disjoint 16-core CPU binding. Native, BLAS,
allocator, MPI settings and source/binary hashes pass the runtime checks.

Factor time includes construction and compression. Solve time includes
iterative refinement for all three RHS. Each phase column is its own median;
the paired total is the median of factor-plus-solve within each trial, so it
need not equal the sum of the two phase medians. All trials are retained.
Ordering and unrelated startup are outside these two phases.

Convergence and accuracy
------------------------

The old benchmark gate required reported solution error below 1e-8 and
componentwise scaled residual below 1e-6 while solving at relative tolerance
1e-12. Those criteria are recorded separately here. They do not replace the
newly requested solver stopping criterion.

Before any trial, the rerun harness is configured to require a finite final
``REFINEMENT`` record and the actual solver stopping condition: absolute
residual at most 1e-12, or relative residual at most 1e-6, or componentwise
backward error at most 1e-12. It also checks printed solver settings, matching
iteration counts within the maximum, finite output metrics and normal
completion. A relative allowance of 1e-5 on these thresholds accounts for
the six-significant-digit log formatting. Launcher exit zero alone is
insufficient: the example driver does not check ``factor()``/``solve()``
return values.

The iterative-refinement absolute and relative residuals use a global
Frobenius norm over the three RHS. The componentwise scaled residual is a
global maximum. The example driver's reported solution error is a
rank-zero-local Frobenius ratio; it is not an MPI-global solution-error norm.
The companion JSON retains these metric scopes and every trial's values.

.. list-table:: Worst observed accuracy metrics across each three-trial cohort
   :header-rows: 1

   * - Grid
     - Global relative Frobenius residual
     - Rank-zero-local relative solution error
     - Global componentwise scaled residual
   * - 100 cubed
     - 1.14446e-08
     - 1.77933e-06
     - 1.22241e-06
   * - 150 cubed
     - 1.05257e-08
     - 2.87651e-06
     - 9.83839e-07
   * - 200 cubed
     - 7.95216e-07
     - 3.12115e-05
     - 0.000529009

All nine trials meet the requested relative-residual tolerance. None meets
the former combined strict error/residual gate. In particular, the 200-cubed
rank-zero solution error reaches 3.12115e-5 and the global componentwise
scaled residual reaches 5.29009e-4. The solver relative tolerance controls
the global relative residual; the forward-error and componentwise metrics
remain separately reported.

Comparison with the earlier settings
------------------------------------

.. list-table:: Median paired factor plus solve, seconds
   :header-rows: 1

   * - Grid
     - Prior HODLR 1e-2 / solver 1e-12
     - Prior HODLR 1e-5 / solver 1e-12
     - Requested HODLR 1e-3 / solver 1e-6
     - Reduction from prior best total
   * - 100 cubed
     - 10.611
     - 9.740
     - 7.946
     - 18.4%
   * - 150 cubed
     - 45.823
     - 33.181
     - 31.330
     - 5.6%
   * - 200 cubed
     - 126.272
     - 116.301
     - 81.774
     - 29.7%

Both compression and solver tolerance change in this comparison. Timing
differences therefore reflect the requested accuracy/performance tradeoff
and cannot be attributed to compression tolerance alone. The prior evidence
is retained in ``HODLR_GPU_EXACT_FRONTS.rst`` and its JSON.

The accompanying ``HODLR_GPU_EXACT_RELAXED_SOLVE.json`` records commands,
configuration, hashes, job IDs, trial timings, convergence traces, memory
and transfer counters, actual GPU routes, and binding/runtime audits. Raw
logs, job folders and build directories remain outside the commit through
the local ignore list.
