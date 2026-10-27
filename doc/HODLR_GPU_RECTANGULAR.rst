Experimental rectangular GPU and distributed construction path
==============================================================

This branch extends the hybrid interface at butterfly level zero. It is
opt-in, requires new ButterflyPACK symbols, and is not a speedup claim.
The original interface and its dependency remain usable with the extension
disabled.

Build and selection
-------------------

Use ButterflyPACK branch feature/gpu-rectangular-schur and configure with
STRUMPACK_USE_HODLR_GPU_RECTANGULAR=ON. Configure-time link checks require
both real and complex rectangular activation, evaluator registration,
resident-image upload and compound Schur-product entry points. Add
STRUMPACK_USE_HODLR_GPU_EVALUATOR=ON for device entry construction.
STRUMPACK_USE_CUDA may remain OFF.

Select --hodlr_use_gpu 1 --hodlr_gpu_rectangular 1. With a zero
--hodlr_gpu_front_max_bytes, F12/F21 still compress on the host and upload
low-rank factors for GPU multiplies. A nonzero byte cap also enables GPU
BACA construction of eligible F11/F12/F21/F22 blocks. The cap includes
prepared numerical data, CSR offsets and alignment. Nonzero butterfly
levels and concurrent OpenMP backend calls remain unsupported.

On Perlmutter::

  HODLR_BPACK_SOURCE="$PWD/../ButterflyPACK-gpu-rectangular" \
    bash build_butterflypack_hodlr_gpu.sh
  HODLR_SKIP_BPACK=1 HODLR_GPU_RECTANGULAR=ON HODLR_GPU_EVALUATOR=ON \
    HODLR_BPACK_INSTALL="$PWD/../ButterflyPACK-gpu-rectangular/install_hodlr_gpu" \
    HODLR_STRUMPACK_BUILD="$PWD/build_rectangular_hodlr" \
    bash config_perlmutter_gpu_hodlr.sh
  cmake --build build_rectangular_hodlr --target test_HODLR_gpu_rectangular
  sbatch test/validate_hodlr_gpu_rectangular_perlmutter.sbatch

Rectangular backend
-------------------

ButterflyPACK standalone BF handles own separate GPU state, activated after
construct_init. This is not a cast from a BF handle to a Bmatrix handle.
Only level-zero blocks are accepted. The implementation reuses the HODLR
split/BACA/TSQR/redistribution algorithm for rectangular dimensions; the
factorization representation remains U times V-transpose, including complex
arithmetic. Existing BF extraction retains valid host mirrors.

The registered rectangular evaluator receives merged tree slots: rows
0..M-1 and columns M..M+N-1. STRUMPACK subtracts the column base exactly
once. Square HODLR callbacks retain their application-index convention.
GPU-built factors are reused from matrix-owned device mirrors, avoiding a
second upload. Rectangular multiplies support N/T/C and preserve padded multi-RHS views.
Each rank multiplies its factor rows; only rank-by-RHS coefficients need a
collective sum. The initial implementation stages that sum through host
memory. It does not assume GPU-aware MPI or create MPI_COMM_WORLD grids.

Device Schur intermediates
---------------------------

The compound backend call performs F21 * inverse(F11) * F12 and its
transpose/conjugate transpose on the backend stream. It uploads the input once and downloads
the output once. F11-sized intermediate vectors remain in the backend GPU
heap, including across its inverse application. Internal MPI reductions
can still use host staging. Complex conjugation is applied on the device around the transposed
pipeline, so complex C applications also retain device intermediates.
The compound interface assumes the unit scaling used by STRUMPACK.

ButterflyMatrix::backend_statistics reports construction, multiply and
compound Schur call times, outer host payload bytes and intermediate device
payload peak. These are synchronous API times, not isolated kernel times.
The intermediate peak excludes the inverse implementation's own scratch.
The legacy host_prepare_seconds field measures preparation wall time,
including synchronous evaluator registration and resident-image upload.
Resident uploads occur before compress(), so they are included in the
sparse factor time and preparation interval, not the native construction
API interval.
Per-front GPU Schur reports expose the executed call count and payloads.
Compound work is recorded in ButterflyMatrix's Schur counters; it bypasses
the individual HODLRMatrix inverse counters and legacy FLOP accounting.
Reported Schur boundary bytes cover outer RHS payload only. Resident-byte
report totals cover printed rectangular images, not complete solver device
memory or all replicas. Use whole-solver wall times for speed comparisons.
The temporary Schur matrix still uses the existing host randomized
compression, with its operator applications on the device.

Collective preparation and ownership
------------------------------------

For an eligible distributed block, all ranks of the front prepare the same
bounded entry image before entering adaptive construction. Existing
extract_2d routes the original entries and child contributions across their
process grids. A collective gather then replicates that image on the
front ranks. F22 deliberately skips original A(u,u) and includes children
plus this front's negative Schur product. Rank-local GPU callbacks issue
no MPI calls and perform no nested child extraction.

Images are uploaded once into matrix-owned leases from ButterflyPACK's
resident heap. All callbacks reuse them. Upload completes before native
construction begins; callback events join streams and finish before child
work is released. The backend frees the image when its matrix is deleted.
Prepared host data and callbacks survive C++ owner moves. Reported resident
bytes are distinct from per-callback scratch bytes and upload traffic.

This is a bounded distributed implementation, not a scalable tiled data
service. Preparation and replication still run on the host, and images
remain for matrix lifetime. The byte cap is per block, not an aggregate
solver memory budget. Oversized blocks explicitly retain host compression.
An unbounded adaptive path needs collective tile-request rounds, child
factor-row routing, leases with eviction and per-GPU memory accounting.

Validation record
-------------------

Job 60390291 PASSED on the final numerical source (STRUMPACK a450b185,
ButterflyPACK 8285e65): device contract tests, zero Compute Sanitizer memory
errors, square-evaluator regression, real/complex direct checks at one,
three and four ranks, scoped MPI_SELF work while other ranks wait, and all
six 40^3 sparse variants with three RHS and 24 HODLR fronts. The four-rank
constructed case reports 46 rectangular GPU constructions, nine collective
prepared images and 69 device Schur calls across front roots. Its relative
error is 2.64e-11 and scaled residual 3.42e-9. Raw logs remain under the
ignored profile_results/60390291 folder. Exact executable and library hashes
are retained in HODLR_GPU_RECTANGULAR.json.

The initial job 60387091 also passed before device-factor reuse and the
conjugate-transpose extension. Failed integration attempts 60386850 and
60386973 remain in the JSON with their causes and corrections; neither
reached the sparse tests. Jobs 60389146/60389147 were cancelled while
pending to finish the complex path and used no compute allocation.
The direct regression compares real and
complex rectangular GPU BACA and N/T/C multiplies against independent dense
references, tests one/three RHS with padding and moved owners, and compares
normal/transposed/conjugate-transposed device Schur products. The sparse test compares hybrid,
rectangular-only and constructed variants at one and four ranks, with three
RHS and multiple fronts. The constructed four-rank case must show collective
preparation and actual rectangular GPU evaluator calls. Raw logs and builds
are local ignored artifacts; the JSON record retains results and provenance.

A separate 60^3 profiling harness runs three alternating fresh-process trials
of CPU, original hybrid, rectangular multiplies/device Schur, and bounded
GPU construction on one allocation with four ranks sharing two GPUs. Its
256 MiB per-block cap and tolerance 1e-5 are recorded separately from the
40^3 validation. Run test/profile_hodlr_rectangular_perlmutter.sbatch and
analyze with test/analyze_hodlr_rectangular_profile.py. Timings from one
trial are not used to claim a performance improvement.

The controlled profiling run is job 60390292. Its twelve-case plan is still
running; incomplete timing evidence is not a passing profiling record.

Repeated 60^3 comparison
------------------------

Job 60390292 PASSED (9m18s) with three fresh-process trials per variant,
four MPI ranks sharing two GPUs, three RHS, 113 HODLR fronts and relative
compression tolerance 1e-5. All twelve trials pass the numerical criteria
and use six refinement iterations. The same uninstrumented Release
executable and one allocation were used for every variant, alternating order.

.. list-table:: Median factor / solve / total seconds
   :header-rows: 1

   * - Path
     - Factor
     - Solve
     - Total
   * - CPU
     - 44.623
     - 0.469
     - 45.095
   * - Original hybrid
     - 45.846
     - 0.680
     - 46.523
   * - GPU rectangular multiplies and device Schur
     - 45.292
     - 0.911
     - 46.202
   * - Plus bounded GPU construction
     - 30.274
     - 0.917
     - 31.190

The constructed path reduces median factor-plus-solve time by 33.0 percent
relative to the original hybrid and 30.8 percent relative to CPU in this
layout. Its observed total-time range is 31.045--32.577 s; the hybrid range
is 45.643--46.807 s. Rectangular multiplies alone give little end-to-end
improvement, and solve times increase. The useful improvement here comes
with device construction and resident entry images. This is a small,
controlled regime, not evidence of a 100^3--200^3 speedup or an asymptotic
crossover. All constructed trials report 224 rectangular GPU constructions
and nine collective prepared images, with finite relative errors below
1.74e-9 and scaled residuals below 2.16e-8.

The timing executable predates only the final registration shape/type
validators; its numerical implementation is unchanged. Its SHA256 and
source/dependency commits are recorded separately in the JSON. Final-code
sparse validation is rerun after those guards. Compute Sanitizer covers
the evaluator device-contract test, not the full native sparse solver.
