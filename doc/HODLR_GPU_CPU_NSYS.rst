150-cubed matched CPU control and Nsight profile
=================================================

The CPU control passed, but the quiet GPU repeat and both profiled GPU attempts failed numerical accuracy. There is no new valid matched GPU speed ratio. These failures take priority over further feature optimization; the cause has not been established.

Matched quiet run
-----------------

Job 60416297 ran the same frozen Release executable and ButterflyPACK library, CPU first and GPU panels second, on the same four-node allocation. Both used 32 MPI ranks, eight allocated CPUs per rank, one OpenMP/OpenBLAS thread, three RHS, leaf size 128, compression tolerance 1e-5, refinement tolerance 1e-10 and butterfly level zero. STRUMPACK dense CUDA was disabled. The GPU case used 16 A100 40 GB GPUs, two ranks per GPU, an 85% native pool, 256 MiB image cap and 64 MiB panel staging budget.

The CPU command selected ``--hodlr_use_gpu 0 --hodlr_gpu_rectangular 0 --hodlr_gpu_front_max_bytes 0``. GPU panels were disabled. CPU factorization took 854.156 seconds and solve 2.6076 seconds, totaling 856.7636 seconds. Seven refinement iterations produced relative error 4.74426e-10 and scaled residual 3.11758e-8.

The quiet GPU factorization took 558.144 seconds, followed by 204.398 seconds of divergent refinement ending in NaNs after 391 iterations. This is a failed solve, not a faster successful result. The overall batch status is FAILED because the numerical validator rejected the GPU result; the CPU case remains PASS. Earlier passing 428.507-second GPU timing is preserved as historical evidence, not substituted for this failed matched repeat.

Nsight run and validity
-----------------------

Corrected profile job 60417148 retained and exported reports for ranks 0, 11, 15 and 31. It traced CUDA, NVTX, cuBLAS and OS runtime APIs, sampled process CPU stacks with DWARF at period 8,000,000, and collected process thread scheduling. Nsight Systems was version 2026.2.1.210. No MPI API tracing was enabled.

The profiled factorization took 729.443 seconds. Refinement was capped at 30 iterations to bound failed runs; it diverged, with final relative error 1.63561e10 and residual 1. This profile therefore describes a numerically failed factorization/solve. Trace export succeeded, and the numerical failure remains recorded as FAIL. Neither instrumentation overhead nor the differing random compression work can be inferred solely from its difference from the quiet timing.

.. list-table:: Selected-process factor CUDA activity
   :header-rows: 1

   * - Rank
     - Factor wall, s
     - Kernel interval union, s
     - Kernel/copy/memset union, s
   * - 0
     - 729.452
     - 25.839
     - 27.708
   * - 11
     - 729.451
     - 31.078
     - 42.362
   * - 15
     - 729.451
     - 32.168
     - 51.783
   * - 31
     - 729.451
     - 44.965
     - 50.297

Kernel activity is approximately 3.5% to 6.2% of the local factor envelope in these traced processes. Each shares a GPU with an untraced peer: these are process activity intervals, not whole-device utilization or SM occupancy. Host and device work can overlap and must not be added. The most expensive observed kernel was the small batched Jacobi eigensolver, followed by QR and recompression kernels.

What the CPU samples show
-------------------------

Fractions below use collected running-CPU backtraces inside factorization. They are not wall-time percentages. The ButterflyPACK host category excludes visible CUDA, MPI and BLAS/ScaLAPACK frames. ScaLAPACK includes PBLAS and BLACS, with MPI below it separated.

.. list-table:: Collected factor CPU backtraces, percent
   :header-rows: 1

   * - Rank
     - ButterflyPACK host
     - CUDA/library host
     - BLAS caller incomplete
     - ScaLAPACK outside MPI
     - host_wy leaf
   * - 0
     - 21.10
     - 19.88
     - 9.18
     - 1.35
     - 2.71
   * - 11
     - 47.09
     - 17.87
     - 13.63
     - 1.11
     - 16.47
   * - 15
     - 62.51
     - 14.04
     - 12.70
     - 0.90
     - 28.90
   * - 31
     - 31.80
     - 24.47
     - 13.54
     - 0.65
     - 7.04

The visible hotspots point to ButterflyPACK host processing, extraction and communication, with a substantial host QR/WY fallback. They do not identify ScaLAPACK arithmetic as the dominant cost. Incomplete BLAS caller stacks prevent a complete attribution of all BLAS work to ScaLAPACK versus other callers.

``bpack::gpu::HodlrConstruct<double>::host_wy`` is directly visible as the leaf in 16.47% and 28.90% of collected factor backtraces on ranks 11 and 15. Source inspection locates it in ``GPU_BACKEND/hodlr_gpu/hodlr_construct.hpp`` around line 1388. A zero Householder tau selects this fallback: it synchronously downloads Gram/tau/RHS data, forms the compact WY triangular factor in serial CPU loops, and uploads the result. It does not call ScaLAPACK. Its work includes cubic loops in the QR basis size.

Other visible functions include ``d_hodlr_gpu_fill_rows``, ``d_lr_block_extraction``, ``d_lr_all2all_extraction`` and ``d_bpack_all2all_inters_optimized``. These identify GPU-factor row staging, host low-rank reconstruction and extraction/communication preparation as concrete costs. The most visible ScaLAPACK function was ``pdgesvd_``, appearing in 0.48% to 0.99% of collected factor backtraces across the selected ranks. Inclusive function samples overlap.

CUDA API host costs are also visible: ``cudaMemcpyAsync`` sums approximately 35 to 66 seconds per selected process over the complete trace. These are API host durations, including ordering or staging costs, not device transfer durations; they overlap other measurements.

Independent CSV scopes still partition the factor envelope on all 32 ranks. In this failed profile, mean exclusive rank wall time is 220.792 seconds in host panel callbacks and 353.492 seconds in the outer factor remainder, which includes hierarchy/setup, dense fronts and subtree waits. These categories are not isolated CPU arithmetic. Do not add them to sampled function fractions or CUDA intervals.

Limits and failure history
--------------------------

Only 1.18% to 25.48% of factor backtraces reach a known program entry frame; unresolved leaf fractions range from 20.3% to 63.6%. Caller attribution is therefore incomplete. Kernel-space CPU stacks are unavailable under the node settings. Nsight also warns that some NVTX events might be missing and reports four or five incomplete CUPTI events dropped at shutdown per report. All selected outer factor NVTX durations agree with direct CSV scopes within 0.001 second, but finer nested attribution remains approximate.

First profile job 60416427 independently diverged after a finite first correction, ultimately reaching NaNs after 210 refinement iterations. Its factorization took 550.992 seconds. A launcher error, ``srun --wait=30``, then killed Nsight report generation after untraced ranks exited, so no nsys-rep files from that attempt were retained. The launcher now uses ``--wait=0`` while retaining the outer timeout and Slurm allocation limit. The complete repeat above uses that correction.

The quiet GPU failure rules out Nsight as the sole cause. The original numerical failure preceded the report-generation kill. No numerical source or tolerances were changed in this study. The exact failure mechanism has not been localized, and the host WY hotspot is not proof of its cause.

Next work
---------

1. Localize the first inaccurate front or construction/recompression block and establish numerical correctness. Keep failed runs visible; do not optimize or publish a speedup based on their timings.
2. After correctness is established, investigate the zero-tau-safe device QR/WY application, host child/current-Schur extraction and row-staging/communication path. The trace provides evidence for these targets ahead of generic GPU kernel tuning.

Complete settings, hashes, selected CPU samples and trace diagnostics are in ``HODLR_GPU_CPU_NSYS.json``. Raw reports and logs are retained in locally ignored ``profile_results/60417148/`` and ``profile_results/60416297/`` folders. The raw Nsight reports are named ``rank0.nsys-rep``, ``rank11.nsys-rep``, ``rank15.nsys-rep`` and ``rank31.nsys-rep``.
