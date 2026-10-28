Bounded distributed frontal evaluation: implementation design
============================================================

The first hybrid milestone is implemented through bounded BACA panel
rendezvous; see HODLR_GPU_HOST_PANELS.rst. Whole-piece device preparation
and device-side child assembly below remain a design, not an enabled
implementation.
The earlier 150-cubed profile attributes 34.2% of mean rank factor time to
host F22 construction and 22.4% to host F12/F21 construction. Only 94 F22
and 22 fronts of each rectangular orientation account for those costs.
The initial target is those large blocks, rather than the already small
GPU factorization and compound Schur operator budgets.

The implemented host-panel milestone subsequently passed 150-cubed job
60412989 and reduced paired factor-plus-solve time from 695.620 to 428.507
seconds. Profile job 60412991 now attributes 36.0% of the mean rank factor
envelope to host panel callbacks. The remaining device-child design below
targets that preparation cost. Whole-piece scheduling remains an alternative
design; it is not a prerequisite already implemented by the panel protocol.

Why the current callback is insufficient
---------------------------------------

``GPUFrontEvaluator`` receives a complete immutable entry image. Distributed
fronts form that image with ``extract_2d``, include child contributions,
all-gather it onto every participating rank and upload it once. A per-block
256 MiB cap excludes the largest blocks. Raising it preserves quadratic
replication and does not impose an aggregate device-memory budget.

``bpack_gpu_block_evaluator`` is rank-local. Its panel batches differ across
ranks and BACA iterations. It cannot call STRUMPACK's collective extraction
routine directly: ranks would enter different collective sequences. This
also prevents an on-demand MPI request inside the callback from being a
safe general solution.

Collective preparation followed by local evaluation
--------------------------------------------------

Add an optional preparation/release protocol alongside the existing block
evaluator, without changing existing callback behavior. Preparation runs
on the full front communicator, before native construction descends into
construction subgroups. Ranks with no requested entries participate with
empty descriptors. No new operation uses MPI_COMM_WORLD.

1. **Plan native pieces.** Separate planning from execution in
   ``HODLR_gpu_construct_level`` and ``GPU_construct_shared_block``.
   ``HODLR_gpu_piece_list`` already describes split regions and owners.
   Produce a canonical front-level plan for local and shared blocks before
   either path starts compression. Descriptors contain block and window
   identifiers, row/column sets, dimensions, front-communicator owner ranks
   and an ownership epoch. Balanced-piece reassignment must update the plan
   before preparation; it cannot silently change owners afterwards.

2. **Bound windows.** Split large single-rank pieces as well as distributed
   pieces. Use the native split/merge machinery, rather than changing the
   matrix formula. Schedule the same number of preparation windows on all
   front ranks. Each owner's total active image bytes, metadata and scratch
   must fit its window budget. Account for retained compressed factors and
   merge scratch separately. Query the backend heap's available capacity;
   physical ``cudaMemGetInfo`` alone does not describe free space inside its
   already reserved pool.

3. **Prepare correct frontal entries.** Convert clustering slots through
   ButterflyPACK's permutation to application ordering. Map separator ids
   through ``sep_begin`` and update ids through the existing update/CB
   permutations. Rectangular merged-tree column slots require their row
   base to be removed first. Use 64-bit ids in the protocol.

   F11, F12 and F21 contain original sparse entries plus embedded child
   Schur contributions. F22 contains embedded child contributions and this
   front's negative Schur product; it does not add original sparse A(u,u).
   Preserve the batched ``extract_2d`` formula as the reference implementation.

4. **Compute with local leases.** The preparation callback returns device
   leases covering every region that the next local BACA window may read.
   The existing evaluator only gathers entries from those leases and
   launches on ``batch.stream``. It performs no MPI or synchronous host
   extraction. An ownership epoch prevents reuse of stale windows. Native
   BACA work itself has no MPI collectives, so independent owners can finish
   their local window before rejoining the next preparation rendezvous.

5. **Finish and merge.** Retain each piece's compressed U/V ids across
   windows, then use the existing hierarchical merge and layout conversion.
   End-window events fence queued evaluator reads before releasing the
   entry images. Release images and host staging after construction, rather
   than retaining full entry images for the factor lifetime. Keep child
   factors alive until all parent windows and merges are complete.

The present native shared-block path combines splitting, local BACA,
collective merging and layout conversion in one routine. The preparation
hook must not simply be inserted inside that routine: its subgroup may be
smaller than the communicator required by STRUMPACK's child extraction.
Refactoring the front-level schedule is a required part of this extension.

Device-accessible child contributions
------------------------------------

Use two explicit implementation milestones, with separate counters:

* First establish bounded owner-local windows using the existing collective
  host extraction as a correctness reference and upload each owner's image
  once. This enables GPU BACA of large blocks but remains host preparation.

* Then add device extraction leases for child factors and the current
  rectangular Schur representation. Existing native mirror/download helpers
  do not constitute a distributed device-entry extraction service. Define
  which child ranks own the needed U/V panels, gather requested panels or
  computed contribution tiles in a collective preparation phase, and
  reconstruct/add contributions on the destination GPU. CPU-resident child
  factors need an explicit import or bounded host fallback.

Initially retain the proven host-staged MPI transport. A producer event
precedes staging or device communication; a receiver event precedes
evaluation. Device pointers are local-process resources and are never sent
as remotely dereferenceable addresses. The parent owns assembled windows;
children retain ownership of their factors, with leases protecting reads.
Shared process trees and compatible BLACS grids remain unchanged.

Diagnostics that revisit entries after construction need a new preparation
epoch or a supported host reference. They cannot invoke an evaluator whose
window has been released. Callback registration and release must therefore
have an explicit construction lifetime, including native correctness checks.

Acceptance and measurements
--------------------------

Force several windows with a very small cap. Compare every assembled tile
against an independent sparse-plus-two-children reference, including the
current negative Schur term, repeated/permuted ids, asymmetric complex
entries and rectangular column offsets. Cover one, three and four ranks,
empty requests, strict subcommunicators, moves and child-release ordering.
Check stream ordering with delayed producers and memory checking.

Then repeat the full 150-cubed solve at the same numerical tolerances and
layout. Report F22/F12/F21 host versus device construction coverage, image
and factor memory separately, per-window maximum bytes, preparation and
MPI staging times, transfer bytes, native GPU BACA time and parent-arrival
gaps. Preserve N/T/C and padded multi-RHS regressions. Reassess subtree
placement after the compression cost distribution changes.
