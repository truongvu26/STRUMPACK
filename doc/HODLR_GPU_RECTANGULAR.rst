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

Job 60387091 passed the initial one-/four-rank direct checks and all six
40^3 sparse variants. Final checks after factor-buffer reuse are pending.
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
