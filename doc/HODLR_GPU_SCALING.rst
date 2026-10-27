Large-grid Poisson CPU and GPU comparison
=========================================

The test uses the final bounded-evaluator executable from source b2b0a48a,
with subsequent harness-only commits. The numerical configuration is the same
as the successful 40^3 validation: three RHS, HODLR leaf size 128, compression
thresholds 1000, relative compression tolerance 1e-5, refinement tolerance
1e-10, element extraction and zero butterfly levels. STRUMPACK's CUDA dense
front backend remains OFF.

Each case uses 32 MPI ranks on four A100 GPU nodes, eight ranks per node and
two ranks per GPU. OMP_NUM_THREADS and OPENBLAS_NUM_THREADS are both one;
MPI transfers use host staging. CPU mode, hybrid GPU mode and the bounded
GPU evaluator are compared using the same executable. The evaluator's cap
is 8 MiB per F11/F22 block, with explicit host fallback for oversized images
and distributed fronts.

100^3 has 1,000,000 unknowns; 150^3 has 3,375,000; 200^3 has 8,000,000.
The driver constructs the original matrix on rank zero and redistributes it
before factorization; factor/solve times exclude that initial construction.

Execution and evidence
-----------------------

HODLR_GPU_SCALING.json is the incremental evidence record. RUNNING means
some requested cases have not completed. It separates each allocation's
scheduler state from individual numerical-case results and checks that both
allocations use the same executable hash.

Job 60351838 runs three fresh-process repetitions of each 100^3 variant,
alternating variant order. After all nine cases pass, its controller releases
the allocation, so its scheduler state can be CANCELLED even though every
retained 100^3 case completed with exit zero. The original broader driver
plan is superseded by the recorded scope_change.json in that job folder.

Job 60353138 continues one initial comparison of each 150^3 and 200^3
variant in a four-hour allocation. It reuses the completed CPU 150^3 step
from job 60352297 after correcting an analysis-only failure: the parser had
read the printed count 1,980 as 1 HODLR front. The solver itself exited zero
with forward error 2.68729e-10 and scaled residual 2.00859e-8. Reuse checks
the executable hash and every solver argument before copying any evidence. Measured 100^3 factorization took roughly 140--150
seconds per run, making three repetitions at every larger size unsuitable
for the initial two-hour allocation. Single-trial large-grid times are
preliminary and do not establish a speed improvement or timing uncertainty.

Every completed case must have finite forward error below 1e-8, scaled
residual below 1e-6, multiple HODLR fronts and positive phase times. Binding
records must identify sixteen GPU UUIDs, each used by exactly two ranks.
Evaluator cases must prepare device F11 blocks; F22 counts and host fallback
counts are recorded rather than assuming every front fits the cap. A successful
registered construction internally verifies that its device callback ran.

These are uninstrumented Release runs with ButterflyPACK's expensive native
verbose entry/product checks disabled. GPU preparation messages remain enabled.
The case summaries include refinement counts because changing iterative work
can affect solve-time comparisons. Timings cover sparse factorization and
solve, not isolated kernels or SM occupancy. Job accounting separately records
sampled maximum resident CPU memory per MPI job step; the launcher's
/usr/bin/time RSS is not the MPI ranks' memory usage.

Reproduce
----------

Run test/scale_hodlr_gpu_perlmutter.sbatch with the evaluator-enabled build.
HODLR_SCALE_GRIDS selects the grid sides and HODLR_SCALE_REPEATS selects one
to three repetitions. For example, the larger-grid initial pass is::

  HODLR_SCALE_GRIDS='150 200' HODLR_SCALE_REPEATS=1 \
    salloc -A m4872_g -C gpu -q interactive -N 4 --gpus-per-node=4 \
    --ntasks-per-node=8 -c 8 --mem=0 -t 04:00:00 \
    bash test/scale_hodlr_gpu_perlmutter.sbatch

Collect the completed evidence with::

  python3 test/collect_hodlr_gpu_scale.py \
    profile_results/60351838 profile_results/60353138 \
    --output doc/HODLR_GPU_SCALING.json --require-complete

Raw logs, commands, input configuration snapshots, memory accounting and job
folders remain under the locally ignored profile_results directories.


Hardware and reuse qualification
--------------------------------

The completed 100^3 runs and the original CPU 150^3 run used A100 40 GB
nodes. Continuation job 60353138 has eight A100 40 GB and eight A100 80 GB
GPUs. All 200^3 variants will use that same mixed allocation, while the
150^3 CPU baseline originates in the earlier all-40-GB allocation. The
initial large-size timings therefore do not establish a CPU/GPU speedup;
the GPU inventory and reused-case origin remain in the JSON record.

Job 60353035 received no resources and timed out while waiting for four
40 GB nodes. It ran no solver tests. Job 60352297's FAILED allocation state
comes from the grouped-number parser, not a solver failure; its CPU solver
step is COMPLETED with exit zero. Its successful result is preserved.

The continuation now owns a read-only solver_snapshot with the identical
recorded executable hash. Subsequent ranks execute that job-local copy so
unrelated builds in the shared worktree cannot change later trials. The
source build path remains in recorded command arguments, and the snapshot
checksum and rank-wrapper copy document the actual launch selection.

.. BEGIN GENERATED SCALING RESULTS

Measured results
-----------------

Numerical status: FAIL. The JSON record contains individual
trial accuracy, device-construction coverage and allocation accounting.

.. list-table:: Sparse factor / solve times in seconds
   :header-rows: 1

   * - Grid
     - CPU
     - Hybrid GPU
     - Evaluator GPU
     - Samples per variant
   * - 100^3
     - 141.261 / 0.592
     - 146.255 / 1.231
     - 144.706 / 1.230
     - 3/3/3 (CPU/hybrid/evaluator)
   * - 150^3
     - 845.854 / 3.730
     - 843.150 / 6.951
     - 851.115 / 4.185
     - 1/1/1 (CPU/hybrid/evaluator)
   * - 200^3
     - failed
     - not run
     - not run
     - 0/0/0 (CPU/hybrid/evaluator)

100^3 rows use medians when repetitions have completed; 150^3 and 200^3
remain single-trial observations. Pending entries are not passing tests.

Allocation states:

* Job 60351838: scheduler CANCELLED by 115486, numerical plan PASS.

* Job 60353138: scheduler TIMEOUT, numerical plan FAIL.

Failures:

* n200_cpu_r1: MPICH communicator contexts exhausted (0/4096 free); see n200_cpu_r1.rank31.log
