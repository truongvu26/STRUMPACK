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

Job 60352297 uses a four-hour allocation for one initial comparison of each
150^3 and 200^3 variant. Measured 100^3 factorization took roughly 140--150
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
    profile_results/60351838 profile_results/60352297 \
    --output doc/HODLR_GPU_SCALING.json --require-complete

Raw logs, commands, input configuration snapshots, memory accounting and job
folders remain under the locally ignored profile_results directories.
