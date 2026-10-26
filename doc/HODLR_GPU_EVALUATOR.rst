Experimental bounded GPU frontal evaluator
==========================================

This implementation is the single-rank correctness bridge from
HODLR_GPU_DESIGN.rst. It adds device assembly and ButterflyPACK GPU BACA for
eligible F11/F22 blocks while retaining host preparation of child data and
the current-front Schur product. It is opt-in and makes no speed claim.
The hybrid path remains the default and the reference for large fronts.

Build and select
----------------

Configure with STRUMPACK_USE_HODLR_GPU_EVALUATOR=ON. This separately probes
bpack_gpu.h and both d_/z_ block registration symbols and enables a small
CUDA object target. STRUMPACK_USE_CUDA can remain OFF: the dense-front
backend and ordinary installed C++ consumers do not need CUDA compilation.
CUDA runtime dependencies still come from the exported package target.

On Perlmutter, reusing the metadata-fixed ButterflyPACK installation::

  HODLR_SKIP_BPACK=1 HODLR_GPU_EVALUATOR=ON \
  HODLR_BPACK_INSTALL="$PWD/../ButterflyPACK-gpu-profile-support/install_hodlr_gpu" \
  HODLR_STRUMPACK_BUILD="$PWD/build_optimized_hodlr" \
    bash config_perlmutter_gpu_hodlr.sh

Run with --hodlr_use_gpu 1 (or 2) and, for example,
--hodlr_gpu_front_max_bytes 8388608. A zero cap disables the evaluator.
The nonzero cap requires lr_leaf=4/5 and BF_entry_n15=0. Existing precision,
OpenMP and butterfly-level checks still apply. The cap includes the prepared
CSR, the dense contribution image, and device alignment slack; the square
contribution image and its index count are checked before allocation.

A distributed front uses host construction and reports
reason=distributed_front_requires_collective_preparation. An oversized
single-rank block reports reason=prepared_image_exceeds_byte_cap. Eligibility
is independent for F11 and F22, so one front can use different construction
paths for the two blocks. Rectangular F12/F21 remain unchanged.

Correct assembly and ordering
------------------------------

F11 preparation uses the sparse matrix's virtual count/set_front_elements
interface and sorts the returned triplets into local APPLICATION CSR indices.
This handles both ordinary CSR and the distributed solver's proportionally
mapped CSC storage without treating raw pointers as global CSR rows. It
separately embeds the children's contributions using existing
extract_CB_sub_matrix_blocks routing. The device kernel adds those sparse
entries to the requested contribution entries.

F22 preparation has an EMPTY original sparse matrix. Its contribution image
is the children's update contribution plus the already-constructed negative
Schur product -F21*inverse(F11)*F12. Including A(u,u) here would double count
original entries, so preparation deliberately uses skip_sparse=true. Update
indices follow upd[] and, when enabled, CB_iperm[], exactly like host extraction.

Register after construct_init and before construct_element_compute. The
backend supplies zero-based application row/column ids; the kernel applies
no second clustering permutation. It supports repeated or noncontiguous ids,
complex asymmetric entries, padded leading dimensions and empty blocks.
The evaluator registration sets neither the symmetric nor host-id flags.

No child extraction, MPI operation or nested ButterflyPACK operation occurs
inside the rank-local device callback. All required contributions have
already been prepared. This avoids the mismatched collective participation
that prevents a general distributed evaluator with the current ABI.

Memory and stream ownership
----------------------------

A shared evaluator object owns the prepared host inputs. HODLRMatrix retains
it across moves and destroys it after the backend matrix. Prepared images
remain available for the lifetime of that matrix: the current registration
ABI does not offer unregistering an evaluator. The byte cap is PER BLOCK;
it is not a global solver memory limit. Retained F11 images can accumulate
throughout factorization and solve, so this mode is experimental and bounded
front-by-front rather than a scalable residency strategy.

For each callback, payload scratch comes from batch.allocate; the prototype
uploads the CSR and full contribution image for that batch, then launches
assembly on batch.stream and returns scratch through batch.release. No
persistent cudaMalloc competes with ButterflyPACK's large reserved heap.
The extra repeated uploads are intentional in this correctness bridge and
are exposed through evaluator_upload_bytes_local. A persistent bounded
cache requires an explicit backend buffer-lease and memory-budget contract.

An event joins successive callbacks if their supplied streams differ.
Construction fences the last callback before child work is released;
destruction also fences outstanding reads. Callback descriptors and output
pointers are borrowed only for that batch. CUDA or allocation failure aborts
the job, without throwing through a C/Fortran callback boundary.

Execution reports distinguish gpu_blocks_host_prepared_schur from the
existing host_callbacks_plus_upload path and count evaluator calls, actual
host-to-device payload bytes, and peak per-batch payload scratch. Separate
host_prepare_seconds reports cover the contribution preparation. Existing
construction wall time includes callback uploads, GPU assembly and BACA;
it is not an isolated kernel time or a full front time.

Validation
-----------

Build test_HODLR_gpu_front_evaluator and test_HODLR_gpu_front_integration,
then submit test/validate_hodlr_gpu_front_perlmutter.sbatch. The focused device
contract test includes real/complex child data, repeated/permuted ids,
padded outputs, empty blocks, two nondefault streams and destruction with
pending work. It is also run under compute-sanitizer memcheck. The direct
HODLR test compares asymmetric child-inclusive matrices to dense references
for N/T/C products and three-RHS inverse application after moving the owner.

The sparse check runs 40^3 Poisson with three RHS on one and four ranks,
using the previously validated fixed two-ranks-per-GPU binding for the
four-rank cases, and comparing the default hybrid path, eligible F11/F22
device construction and a deliberately tiny cap forcing host fallback. The sparse configuration uses
compression thresholds of 1000 and HODLR leaf size 128. It checks finite forward
error and scaled residual, actual evaluator use and explicit distributed
fallback. Completed evidence is recorded in HODLR_GPU_EVALUATOR.json.

Initial diagnostic job 60340028 passed both focused tests and memcheck, but
its sparse baseline failed before device evaluation. A deliberately low
compression threshold of 64 created 281 HODLR fronts on one rank and reached
MPICH's 4096-context limit ("Too many communicators (0/4096 free)"). This is
retained as FAILED evidence; it is not classified as an evaluator failure or
a passing end-to-end run. The replacement sparse configuration uses the
previously validated 40^3 problem and normal compression thresholds.

General distributed construction, persistent device child storage and
cross-rank request preparation remain the next implementation stage. The
collective preparation protocol in HODLR_GPU_DESIGN.rst is still required.
OMP_NUM_THREADS>1, randomized matvec construction and nonzero butterfly
levels remain guarded for the dependency reasons documented there.

Completed validation
---------------------

PASS: interactive allocation 60341136, source b2b0a48a, node nid002313,
two A100 40 GB GPUs, completed in 1m44s with exit 0:0. All focused tests and
all six sparse cases pass. The source patch recorded by the job is empty.

.. list-table:: Evaluator-enabled sparse cases, three RHS and 24 HODLR fronts
   :header-rows: 1
   :widths: 16 17 17 24 26

   * - MPI ranks
     - Device F11 blocks
     - Device F22 blocks
     - Relative forward error
     - Scaled residual
   * - 1
     - 23
     - 1
     - 1.40345e-13
     - 2.02270e-11
   * - 4
     - 21
     - 1
     - 3.11010e-11
     - 8.74812e-9

The four-rank case reports three distributed-front host fallbacks. Oversized
images also fall back explicitly. The cap=1 cases report zero evaluator calls.
The four-rank runs verify exactly two GPU UUID groups of two ranks and the
backend's matching share count. Both the ordinary evaluator-disabled build
and an installed host consumer without CUDA compilation build successfully.
Compute Sanitizer reports zero device memory errors.

The direct real/complex regression also checks padded sequential rectangular
Butterfly extraction. Job 60340993 exposed an existing bug in
ButterflyMatrix::set_extraction_meta_1grid: after setting Nalldat_loc from the
requested output size, it reset that value to zero while clearing the row
and column totals. F22 preparation then allocated a zero-sized output buffer
and wrote past it. Commit b2b0a48a preserves the output count and adds the
child-tile regression; both now pass in the full sparse path.

Per-callback uploads total 801375696 bytes on one rank and 526767744 bytes
across the four-rank case. Maximum payload scratch is about 5.2 MB per
callback. These transfers demonstrate the cost of the bounded preparation
bridge. Single verbose runs, adaptive compression and varying refinement
iterations do not establish a speed benefit. Use the uninstrumented repeated
workflow from HODLR_GPU_PROFILE.rst before making performance claims.

Debug jobs 60340270 and 60340660 were cancelled while pending and ran no
tests. The interactive queue was used after the debug queue delayed the
validation; the completed computation itself took less than two minutes.
Raw job folders, logs and build directories remain locally ignored.

The same validation can run through an interactive allocation when the debug
queue is busy; all device work, including device discovery, uses srun::

  salloc -A m4872_g -C gpu -q interactive -N 1 --gpus-per-node=2 \
    -n 4 -c 8 --mem=64G -t 00:05:00 \
    bash test/validate_hodlr_gpu_front_perlmutter.sbatch
