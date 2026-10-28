GPU HODLR profiling after the SVD accuracy fix
=============================================

The completed efficiency follow-up is in HODLR_GPU_DEVICE_WY.rst and its
JSON record. The device WY change gives a 1.095x matched 150-cubed speedup;
the measurements below preserve the earlier QR-SVD profiling baseline.

The 150-cubed Poisson comparison uses 3,375,000 unknowns, three RHS,
32 MPI ranks on four A100 40-GB nodes (16 GPUs reserved), and one OpenMP
thread per rank. Quiet CPU and GPU runs use the identical Release executable
and ButterflyPACK library. The GPU cases enable rectangular HODLR operations,
collective host-prepared panels and QR-based SVD. Residual certification is
specified per case.
STRUMPACK's CUDA dense-front backend remains disabled.

The native fix is ButterflyPACK commit 88e859d. Subsequent STRUMPACK changes
are test/report changes; the production solver sources are unchanged since
the quiet executable was built. Exact binary hashes and settings are in
``HODLR_GPU_QR_SVD_PROFILE.json`` and the retained job manifests.
Nsight uses the Release diagnostic executable with stage instrumentation
enabled, built from the same production source and linked to the same
ButterflyPACK library.

Validated evidence
------------------

* CPU 150-cubed job 60423044 passed: factor 838.071 s, solve 3.31297 s,
  relative solution error 6.92735e-10 and scaled residual 4.40636e-8.
* The 100-cubed sparse validation, job 60423305, passed on the target
  32-rank/16-GPU layout: factor 132.915 s, solve 0.605141 s,
  solution error 1.63058e-13 and scaled residual 1.43934e-11.
* Native tests passed 288 independent real/complex recompression comparisons
  and 12 direct SVD checks (60422460). Replaying captured exact frontal
  factors also passed (60422461).
* The original polar-SVD reproducer (60422299) reported info=0 but a
  perturbation of 0.50266009782 for a matrix of norm 2.26377982924.
  Its reconstruction error was 5.80721996296; QR-based SVD reduced that to
  4.59525662666e-15. NVIDIA documents the perturbation used near zero
  singular values in its `cuSOLVER reference
  <https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdnxgesvdp>`_.

Matched 150-cubed timings
------------------------

.. list-table:: One quiet run per case; seconds
   :header-rows: 1

   * - Backend
     - Factor
     - Solve
     - Total
     - Status/job
   * - CPU
     - 838.071
     - 3.31297
     - 841.38397
     - PASS / 60423044
   * - Default GPU (certification off)
     - 379.926
     - 4.84846
     - 384.77446
     - PASS / 60424358
   * - Certified GPU
     - 1174.06
     - 1.59843
     - 1175.65843
     - PASS / 60423201

The default GPU path is 2.19x faster for factor plus solve in this pair.
It uses nine refinement iterations, as does CPU; their maximum HODLR
ranks are 989 and 988. GPU solution error is 3.21923e-10 and scaled
residual is 2.47194e-8. GPU solve alone is slower than CPU and accounts
for only 1.26 percent of the default GPU total.

Certification produces a more accurate preconditioner (three refinement
iterations here) but costs 3.06x the default GPU total. It is a separate,
opt-in numerical diagnostic, not part of the default timing result.

Nsight attribution and next targets
----------------------------------

Job 60425000 passed the complete 150-cubed sparse solve: factor 391.597 s,
solve 7.90356 s, relative error 5.31009e-10 and scaled residual 3.11648e-8.
All four Nsight reports and their CPU/CUDA summaries were retained. All
32 rank CSV envelopes partition correctly. The traced solve uses 15
refinement iterations versus nine in the quiet pair, so the timing
difference includes randomized compression variation as well as tracing.

.. list-table:: Selected processes; factor envelope approximately 391.60 s
   :header-rows: 1

   * - MPI rank
     - Kernel union (s)
     - Kernel/copy/memset union (s)
     - CPU samples with ScaLAPACK/BLACS frame
     - host_wy leaf samples
   * - 0
     - 26.54
     - 28.86
     - 2.24%
     - 3.70%
   * - 11
     - 31.80
     - 40.90
     - 1.77%
     - 10.91%
   * - 15
     - 32.00
     - 37.77
     - 2.00%
     - 20.06%
   * - 31
     - 46.99
     - 52.73
     - 1.00%
     - 7.79%

These process traces point to host construction and coordination as the
leading optimization targets. They do not measure whole-device utilization:
each GPU has a second MPI rank, and only selected processes were traced.
The ScaLAPACK percentages are observed-frame lower bounds, not exclusive
wall-time shares. Most backtraces do not reach main, so an unobserved
ScaLAPACK caller cannot be ruled out for incomplete BLAS or MPI stacks.

Across all 32 ranks, exclusive factor rank-time is distributed as follows:

* ``prepare.host_panels``: 37.17 percent, including extraction, MPI and
  synchronization. CPU stacks show ButterflyPACK extraction/all-to-all
  routines and STRUMPACK's child-inclusive extraction callbacks.
* Uninstrumented remainder in ``solver.factor``: 26.49 percent. For rank 0,
  this scope is dominated by MPI/transport samples; it must not be
  interpreted as useful CPU arithmetic or GPU compute.
* F22 GPU construction: 13.62 percent across panel and resident-image paths.
  This is host-scope wall work, including launches and synchronization.
* F12/F21 GPU construction: 8.74 percent across both paths.
* Host randomized Schur compression: 4.02 percent; this is where sampled
  ``pdgesvd``/PBLAS/BLACS work is concentrated.

Host-panel counters report mean/max preparation time of 145.72/215.41 s
per rank, versus 0.34/0.56 s for uploading the prepared panels. Those
upload counters are a subset of all transfers. Across each selected
process's whole solver trace, device-to-host payload is 13.44--82.38 GB
and host-to-device payload is 8.73--14.28 GB. Their recorded copy-engine
durations are 1.08--7.32 s and 1.18--1.96 s, respectively. In contrast,
``cudaMemcpyAsync`` host API duration is 33.81--60.88 s per selected
process, over 465,850--814,195 calls. These overlapping measurements
identify request/host overhead as a target; they do not form an additive
wall-time breakdown.

Peak per-rank resident-image accounting is 2,856,853,608 bytes across
simultaneously retained images; peak panel staging is 67,026,240 bytes,
within the 67,108,864-byte cap. Neither is a total device-memory peak.

The next targets, in measured order, are:

1. Reduce host-panel assembly and distributed child extraction. Design
   device-resident child factors/contributions and collective extraction
   before implementing a device frontal evaluator. It must assemble the
   original sparse entries plus every child Schur contribution, with
   explicit buffer ownership, rank participation and stream ordering.
2. Move the native zero-tau ``host_wy`` recompression fallback onto the GPU.
   It currently downloads Gram/tau/RHS arrays, executes nested host loops,
   then uploads the result. This is a bounded native optimization; retain
   zero-column, rank-deficient, real/complex and captured-front tests.
3. Batch small metadata/panel transfers and reduce synchronization.
   Hundreds of thousands of ``cudaMemcpyAsync`` calls and millions of
   kernel launches appear per selected process. API host duration is
   substantial even where copy-engine duration is small. Persistent
   pinned workspaces and coalescing are candidates; dependencies must
   remain explicit.
4. Reassess host Schur compression after reducing construction costs.
   This evidence does not justify prioritizing a broad ScaLAPACK
   replacement or solve-only work: the quiet solve is 1.26 percent of
   default GPU factor-plus-solve time.

Trace quality and scope
-----------------------

Nsight warns that some NVTX events might not have been collected and
reports four incomplete CUPTI events dropped at stop for each selected
process. Present factor NVTX envelope durations agree with the independent
CSV envelopes within 0.02 s; complete fine-grained event collection is not
established. User process-tree CPU sampling succeeded, while kernel-space
CPU stacks were unavailable at perf_event_paranoid=2. Broken/incomplete
backtraces limit exact caller attribution. The JSON records these warnings,
sample coverage, transfer counts, image/staging memory counters and the
per-rank measurements. No total device working-set peak is inferred from
image or staging counters.

Interpretation
--------------

Quiet runs supply the primary timing comparison; tracing adds overhead.
Stage sums represent local rank work, not global elapsed time. Kernel
unions for four selected processes do not establish device utilization:
each GPU is shared by two MPI ranks. Recorded resident-image bytes exclude
ButterflyPACK numeric factors, temporary buffers and pool reservations.

The earlier unsuccessful GPU studies remain in ``HODLR_GPU_CPU_NSYS.json``.
Their timings do not demonstrate a valid speedup. Four-rank/2-GPU 100-cubed
checks hit a six-minute validation limit; they are recorded as timeouts,
while the target-layout end-to-end case above passed. Residual certification
is enabled only for the certified cases; default GPU timing and Nsight use
certification disabled.
