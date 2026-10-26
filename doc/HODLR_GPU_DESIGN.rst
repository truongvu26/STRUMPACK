GPU frontal evaluation and capability extensions
===============================================

Status: design for the next implementation; the hybrid path remains the
correctness reference. This document does not enable device construction or
remove any runtime guard. The separate profiling report determines priority.
The dependency audited here is ButterflyPACK gpu_hodlr plus 8d8b38b.

The operator the evaluator must assemble
---------------------------------------

Use the ordering and arithmetic of FrontHODLR::compress_extraction and
FrontHODLRMPI::compress_extraction. For a front with separator s and update
indices u, let C be the sum of its children's embedded Schur contributions::

  F11 = A(s,s) + C(s,s)
  F12 = A(s,u) + C(s,u)
  F21 = A(u,s) + C(u,s)
  F22 = C(u,u) - F21 * inverse(F11) * F12

The last line deliberately has NO original A(u,u). The existing extraction
uses skip_sparse=true for F22; the appropriate ancestor introduces those
original entries. Adding A(u,u) in a device evaluator would double count.
The current-front negative Schur product is required in addition to the
children's contributions. It currently comes from a temporary rectangular
ButterflyMatrix called Schur. Children can be dense CPU fronts as well as
HODLR fronts because of the compression thresholds.

Indices and registration
-------------------------

ButterflyPACK's GPU block ABI is in GPU_INTERFACE/bpack_gpu.h. Register the
d_/z_c_bpack_set_gpu_block_evaluator after construct_init and before
construct_element_compute, independently for F11 and F22. Check the header
and both registration symbols in a separate configure-time capability probe.
Retain the existing hybrid GPU ABI probe. Device construction should have
its own explicit selection; HODLR_use_gpu=1/2 continues to select arithmetic
and factor/solve placement rather than implicitly promising GPU construction.

GPU row_ids and col_ids are int64, zero-based APPLICATION point ids, already
mapped by ButterflyPACK from its internal cluster ordering. Do not apply
HODLRMatrix::perm() a second time. F11 maps i to sep_begin+i. An update block
maps i to upd[i], or upd[CB_iperm[i]] when STRUMPACK_PERMUTE_CB is enabled,
matching the existing host callbacks. Keep the reverse child-update maps
and parent destination offsets separately. Check conversion to the current
32-bit ButterflyPACK interface and refuse overflowing problem sizes.

The callback fills out[r+c*ld], including noncompact leading dimensions.
It must support empty blocks, duplicate indices, unsymmetric matrices and
complex values. Avoid setting BPACK_GPU_SYMMETRIC merely because the test
problem is Poisson. CPU extraction is the entry-by-entry reference.

Data ownership and residency
----------------------------

Introduce a heap-owned FrontDeviceState with an explicit matrix/assembly
epoch. A moved HODLRMatrix retains a shared reference to the same state;
callback user pointers must never point to a stack lambda or movable inline
state. The registration's user object outlives its ButterflyPACK matrix.

.. list-table:: Ownership contract
   :header-rows: 1
   :widths: 24 38 38

   * - Data
     - Owner
     - Required lifetime
   * - Permuted sparse matrix and distribution
     - Solver; immutable during factorization
     - Through all front preparation and dependent transfers
   * - Front-to-global and child-update maps
     - FrontDeviceState; copied to device
     - Through the final evaluator completion event
   * - Child F22 forward representation
     - Child front; parent holds a lease
     - Through parent F11/F12/F21/F22 construction and outstanding extraction
   * - Current-front Schur representation
     - Parent front construction state
     - Through F22 construction and queued tile reads
   * - Requested contribution tiles
     - Bounded per-front cache with explicit byte accounting
     - Through the last assembly kernel that reads each tile
   * - Callback output, ids and block descriptors
     - ButterflyPACK; borrowed
     - Callback-enqueued work on the supplied stream only
   * - Batch scratch
     - ButterflyPACK callback allocator
     - Return through release after its final queued use

Do not silently retain an opaque batch allocator past its documented
lifetime. Persistent device inputs need either an explicitly supported
ButterflyPACK buffer-lease API or an application allocation budget negotiated
before its heap is reserved. The backend reserves 85 percent of available
device memory by default; raw cudaMalloc inside a callback is not a memory
strategy. Account separately for payload allocations, heap reservations,
MPI exchange space and host staging. Multiple ranks sharing one GPU must
negotiate the budget on the full solver communicator before any allocation.

Cache only sparse entries incident on a separator, not a dense original
front. A first distributed implementation may replicate these sparse
triplets over the front communicator under a byte cap. Sort/combine them
with the same duplicate and symmetry semantics as extract_separator_2d.
A later solver-level distributed CSR cache can avoid repeated uploads.
Never silently materialize an unbounded dense child or parent matrix.

Child extraction across ranks
------------------------------

The current GPU block callback is rank-local: it has a CUDA stream and
blocks, but no collective participation guarantee. FrontMPI::extract_2d
and child HODLR extraction contain collectives. Calling them directly from
this callback can deadlock when other ranks have different adaptive batches.
BPACK_GPU_HOST_IDS only supplies host copies and synchronizes their copy;
it does NOT make the callback collective. Recursive backend calls from the
callback also share the backend's mutable queue, metadata and allocator.

Two explicit stages are needed.

1. A bounded initial prototype can support MPI_SELF fronts whose necessary
   child and current-Schur tiles have been prepared before construction.
   Dense materialization is permitted only below an explicit byte cap, with
   a reported host-construction fallback above it. This is a correctness
   bridge, not the scalable distributed implementation or a speed claim.
2. Distributed adaptive construction needs a ButterflyPACK preparation hook
   at a collective construction round, before rank-local GPU evaluation.
   That hook is NOT in the current public ABI and is a prerequisite to the
   general interface.

The distributed hook's protocol must be::

  all front ranks generate a local batch (possibly empty)
    -> agree on a round and the union of requested blocks and destination owners
    -> visit left child, then right child, in the same collective order
    -> participating child ranks extract using the child's own process tree
    -> redistribute requested tiles to parent block owners
    -> stage received data; record tiles_ready on each destination GPU
    -> run local sparse-plus-contribution assembly and GPU BACA step
    -> agree on completion or start the next adaptive round

Existing batched extraction and extend-add routing provide the reference
for the first host-staged exchange. Keep child and parent BLACS grids and
communicators distinct; shared trees are valid within a front, not a license
to reuse a child's ownership map for its parent. Nonmembers do not call child
collectives; every parent rank still participates in the parent round.
Include no-request ranks in termination/progress detection. Do not implement
an all-to-all inside an otherwise rank-local evaluator.

Device-aware MPI is a later transport optimization. First use explicit pinned
host staging and completion before MPI reads sends or the GPU reads receives.
Moving the arithmetic onto the device does not by itself make child extraction
or rectangular Butterfly compression device-native.

Streams and completion
----------------------

All assembly launches use batch.stream. Sparse/map uploads and received
child tiles record readiness events; the supplied stream waits on them.
ButterflyPACK's later work is ordered after that stream. If a helper stream
is used, make the supplied stream wait on the helper's final event before
returning. Default-stream assumptions and host-only completion flags are
insufficient.

Do not retain batch.blocks, output pointers or ids for another callback.
The callback allocator may recycle scratch for subsequent launches on the
same stream. Each parent records a final use event before returning child
leases or releasing work memory. The current release_work_memory immediately
resets child F22 after construction; the device extension must defer that
reset until all users complete. Matrix destruction and exceptional paths
also drain or fence their outstanding work before freeing payloads.

The callback ABI returns void. Preflight eligibility and allocations before
collective construction. Never throw through its C/Fortran boundary; define
a backend-compatible error mechanism for allocation or launch failures that
cannot leave other ranks blocked in an unrelated collective.

Eligibility and correctness gates
---------------------------------

Agree on eligibility over the entire front communicator. GPU BACA requires
RecLR_leaf 4/5, LR_BLK_NUM=1 and forwardN15flag=0, in addition to the current
precision, format and butterfly-level restrictions. Mixed eligible/ineligible
ranks may not silently take different collective paths. Report host fallback
and its reason. Fully device-only selection should reject an unsupported
front instead of claiming GPU construction.

Before an end-to-end timing claim, check GPU-evaluated tiles against host
extraction for both F11 and F22, including nonzero child contributions and
the current-front Schur term. Include permuted/noncontiguous update sets,
dense and HODLR children, asymmetric real and complex problems, noncompact
output strides, empty/unequal rank batches and a parent/child communicator
mismatch. A sparse-matrix-only evaluator must fail these tests. Then rerun
single/multiple RHS residual checks and stress early destruction and allocator
reuse while streams have pending work. Use compute-sanitizer on a small case.

Current limits: actual dependency work
--------------------------------------

.. list-table:: Capability assessment
   :header-rows: 1
   :widths: 19 45 36

   * - Limit
     - Why it exists
     - Requirement before enabling
   * - OpenMP threads greater than one
     - Process-wide Context/MAGMA queue, DeviceHeap, pinned metadata pool,
       tensor-core mode and warm-up state are mutable; external callbacks
       can reenter them. A mutex held across MPI collectives can itself
       deadlock when ranks service overlapping fronts in different orders.
     - Per-execution/thread backend contexts and allocator/stream ownership,
       or a globally ordered serialized GPU scheduler; MPI thread-level
       validation and concurrent-front/callback stress tests.
   * - Randomized matvec construction
     - BPACK_construction_Matvec does not bind/upload the resulting HODLR
       forward matrix, unlike element construction. Removing the C++ guard
       does not establish a GPU representation and can yield a CPU fallback.
     - ButterflyPACK host-matvec-construction-to-GPU bind/upload bridge,
       collective initialization before traversal, per-front native-GPU
       coverage checks, and real/complex N/T/C multi-RHS sparse tests.
   * - Several MPI ranks per GPU
     - ButterflyPACK supports equal-share heaps, but share_device caches the
       first communicator's UUID-based sharing count for the process lifetime.
       A later larger group or changed binding invalidates that assumption.
     - Validate fixed full-solver groups and explicit bindings first; record
       UUIDs, share counts and memory budgets. Expansion/rebinding requires
       a backend lifecycle/reinitialization contract, not a documentation edit.
   * - Nonzero butterfly levels
     - ButterflyPACK explicitly requires LRlevel=0 and uploads simple U/V
       low-rank blocks. HODBF factors have a different representation.
     - A ButterflyPACK GPU HODBF representation and construction/factor/
       multiply/extraction/inverse kernels, or an explicitly labeled CPU
       fallback. Keep the existing error until such backend support exists.

Implement capability changes separately so their numerical and concurrency
coverage can be reviewed independently. None of these conditions is resolved
by changing a command-line validator alone.
