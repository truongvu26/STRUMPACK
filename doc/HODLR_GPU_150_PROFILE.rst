150-cubed GPU HODLR profile
==========================

The current rectangular-GPU/Schur/device-evaluator implementation completed
the full sparse Poisson solve with 3,375,000 unknowns, three right hand sides,
1,980 HODLR fronts and 348,413 dense fronts. The quiet GPU run measured
713.617 seconds for factorization and 4.83595 seconds for solve, including
nine refinement iterations. Relative error was 3.91154e-10 and componentwise
scaled residual was 2.10489e-8. Job 60399062 contains this run and a successful
CPU control in the same allocation.

.. list-table:: Quiet solver timing, seconds
   :header-rows: 1
   :widths: 25 25 25 25

   * - Backend
     - Factorization
     - Solve
     - Factor plus solve
   * - CPU
     - 853.000
     - 3.37882
     - 856.37882
   * - GPU
     - 713.617
     - 4.83595
     - 718.45295

The GPU run reduces factor-plus-solve time by 16.1% (1.192x CPU/GPU ratio).
Both quiet cases used nine refinement iterations. CPU relative error was
5.40764e-10 and scaled residual was 4.04420e-8. This is one trial per backend,
not a repeated benchmark. Stage profiling and the recovered CUDA traces
are complete.

Configuration and provenance
----------------------------

Four Perlmutter nodes, 32 MPI ranks, 16 A100 GPUs with 40 GB each, two ranks
per GPU, eight allocated CPUs per rank, and one OpenMP/OpenBLAS thread per
rank were used. GPU-aware MPI was disabled. The compression tolerance was
1e-5, solver tolerance 1e-10, leaf size 128, minimum separator/front size
1,000 and butterfly level zero. GPU mode and rectangular GPU operations
were enabled, with a 268,435,456-byte entry-image cap per block.

STRUMPACK's numerical source is 24c5b4ae; ButterflyPACK is ae6e81a
(numerical source 8285e65). The quiet executable and shared library were
frozen before submission. The diagnostic executable adds scopes and task
timers without changing the numerical algorithm. Exact hashes and job
records are in ``HODLR_GPU_150_PROFILE.json``. Raw evidence is retained in
locally ignored ``profile_results/<job>/`` folders.

Required memory workaround
--------------------------

The default ``BPACK_GPU_HEAP_FRACTION=0.85`` failed reproducibly in jobs
60397796 and 60397798 with ``cusolverDnCreate failed with status 7`` during
per-front GPU construction warm-up. Neither run reached a complete
factorization or final numerical check. Trace job 60397800 was canceled
before allocation after its diagnostic prerequisite failed.

Probe job 60398989 isolated the same error using two processes sharing one
GPU. With an 85% pool reservation, handle creation failed after 283 and 206
handles respectively, with only 2,359,296 GPU bytes free. With a 50% pool,
both processes created 400 handles successfully; the minimum sampled free
memory was 11,209,539,584 bytes. Host thread counts stayed at four. This
supports device-memory exhaustion outside the reserved pool as the cause,
rather than a fixed handle-count limit or host-thread exhaustion.

The successful solver runs use ``BPACK_GPU_HEAP_FRACTION=0.50`` with the
original frozen binaries and library. This is a runtime workaround, not a
fix to the unbounded accumulation of per-matrix cuSOLVER handles.

Measured factorization stages
-----------------------------

Stage job 60399068 passed with factorization 715.432 seconds, solve 3.96929
seconds, relative error 9.64393e-11 and scaled residual 1.37690e-8. All 32
ranks passed the exclusive-scope accounting check. The following table
partitions the mean local factor envelope. These are rank wall seconds,
including native waits, not isolated CPU or GPU execution times or a
global critical path.

.. list-table:: Exclusive factorization work
   :header-rows: 1
   :widths: 64 18 18

   * - Component
     - Mean rank seconds
     - Fraction
   * - Host-callback compression
     - 413.834
     - 57.8%
   * - Hierarchy/setup, traversal, dense fronts and remaining work
     - 204.689
     - 28.6%
   * - Device-evaluator compression
     - 51.252
     - 7.2%
   * - Entry preparation, child extraction, replication and upload
     - 28.054
     - 3.9%
   * - Host randomized Schur compression outside the device operator
     - 15.597
     - 2.2%
   * - Compound device Schur operator and communication
     - 1.668
     - 0.23%
   * - Native GPU F11 factorization
     - 0.350
     - 0.049%

The host fallback is concentrated in a few large blocks. Of 1,979 F22
blocks, 94 use host compression and consume 244.521 mean rank seconds,
34.2% of the factor envelope. F12 and F21 each have only 22 host-compressed
fronts, but these consume 91.518 and 68.386 seconds respectively, another
22.4% combined. F11 has only three host-compressed fronts, consuming 1.3%.
The remaining 1,885 F22 blocks, 1,957 blocks of each rectangular orientation
and 1,977 F11 blocks use device evaluation.

For example, a 5,625-square F11 image fits the cap while its
5,625-by-16,875 rectangular neighbors and 16,875-square F22 do not. The
slowest observed F22 host construction was front 3130574, size
16,428-square on three ranks, taking 169.065 seconds. Such front calls may
run concurrently; summing their maxima would overstate elapsed time.

Rank imbalance and preparation
------------------------------

Host-compression work ranges from 239.833 to 607.204 seconds per rank.
Uninstrumented factor-scope time ranges from 8.109 to 385.576 seconds. The
existing process-tree timer reports 197.0 mean rank seconds, with a range
of approximately 0.103 to 378.4 seconds. It encloses collective communicator
and BLACS-grid creation after child processing, so it includes waits for
late-arriving subtrees.

The gap before compression of parent front 1665224 is 0.881 to 241.485
seconds across its four ranks. This supports arrival imbalance as a major
part of the setup remainder. These gaps also include hierarchy creation
and other uninstrumented work; the data do not isolate pure MPI wait time.
Do not add the apparent setup saving to a compression saving as though
they were independent.

Entry preparation is a smaller component: mean rank time is 11.061 seconds
in sparse/child extraction, 8.869 in materializing the current Schur term,
5.533 in other child extraction, 0.649 in packing, 0.639 in replication and
0.307 in the resident-image upload call. Maximum logical resident image
payload is 2.661 GiB per rank; this excludes native factors, cuSOLVER
allocations, scratch and reserved pool capacity.

Next targets
------------

1. **Bound cuSOLVER resource ownership.** Reuse or lease handles and parameter
   objects per process/device/stream instead of retaining separate handles
   in every construction, SVD and distributed-QR object. Preserve explicit
   stream ordering and safe concurrent ownership. This removes the default
   configuration's observed failure before pursuing further scaling.

2. **Extend device evaluation to large F22 blocks first, then large F12/F21.**
   Their host compression consumes 56.5% of mean rank factor time. Replace
   full replicated entry images with bounded distributed pieces or tiles.
   Prepare sparse entries, child Schur contributions and the current
   negative Schur product on their owner ranks before local evaluator
   callbacks. Define communication ordering, buffer leases and stream events;
   callbacks cannot introduce unscheduled collectives. Raising the image cap
   alone would increase retained memory and replication without addressing
   this design limit.

3. **Reassess subtree balance using measured compression costs.** The slow
   host F22 branches correlate with large arrival gaps at parent collectives.
   Use F22 update sizes, ranks and observed construction cost in placement
   decisions, and measure arrival times around hierarchy creation. Recheck
   this after device evaluation changes the work distribution.

GPU LU kernel tuning and further fusion inside the compound Schur operator
have small measured budgets in this run. The quiet solve occupies only
0.67% of factor-plus-solve time. Solve optimization and host randomized
Schur compression are secondary to large-block construction and balance.

Selected-rank CUDA evidence
--------------------------

Trace job 60399073 completed the solver with factorization 716.697 seconds,
solve 5.8823 seconds, relative error 7.55519e-11 and scaled residual
1.81176e-8. All 32 ranks completed their reorder, factor and solve scopes;
factor and solve exclusive accounting passed on every rank.

.. list-table:: Whole-process CUDA trace totals
   :header-rows: 1
   :widths: 10 20 25 20 25

   * - Rank
     - Kernels
     - GPU activity union (s)
     - Host to device (GiB)
     - Device to host (GiB)
   * - 0
     - 723,387
     - 20.602
     - 6.425
     - 6.158
   * - 31
     - 1,390,362
     - 38.957
     - 10.645
     - 15.991

Every kernel was attributed through its correlated launch API to a
diagnostic scope. These totals cover startup, factorization, solve and
shutdown. The GPU activity union is per process; an untraced process shares
each GPU, so this is not a whole-device utilization measurement.

There is a smaller opportunity to batch GPU-construction metadata and
staging. ``cudaMemcpyAsync`` has 123,297 calls and 24.781 seconds of summed
host API duration on rank zero, and 245,881 calls and 40.733 seconds on rank
31. Device-evaluated F22 construction accounts for 13.604 and 23.763 seconds
of those host API costs. By comparison, total traced host/device copy-engine
durations are 1.049 and 2.143 seconds. API duration includes blocking and
ordering costs and is not interchangeable with device transfer time. This
supports reducing call/staging overhead after addressing the much larger
host fallback and balance costs, rather than prioritizing transfer bandwidth.

The trace batch status remains **FAILED**: ``srun --wait=60`` killed rank
31 during CPU-only SQLite/report conversion after the other ranks exited.
Both ``.nsys-rep`` files were already saved. Rank 31 was exported and
analyzed offline, with numerical, all-rank accounting and CUDA attribution
checks recorded in ``profile_results/60399073/postprocess_recovery.json``.
Original exit codes and the original failed summary were preserved. The
profiling harness now performs report conversion after the MPI step and
records target and report statuses separately (commit 1565498a).

Measurement limits
------------------

There is one quiet sample per backend. Instrumented and traced samples are
diagnostic measurements, not additional benchmark trials. Per-rank monotonic
timestamps are used for local durations; cross-node absolute timestamps
are not assumed synchronized. CPU and GPU constructions may produce
different compressed factors and refinement iteration counts at the same
tolerance, so correctness and iteration counts must accompany timings.

CUDA/NVTX tracing is limited to ranks zero and 31, each sharing a GPU with
an untraced peer. Per-process GPU activity cannot be interpreted as
whole-device utilization. The trace deliberately avoids MPI interposition.
