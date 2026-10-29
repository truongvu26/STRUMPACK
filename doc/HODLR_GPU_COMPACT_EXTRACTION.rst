Compact child extraction for HODLR construction
==============================================

``--hodlr_compact_extraction 0|1`` defaults to ``0``. Mode 1 routes compact
child intersections to the owner-local panels used by batched HODLR entry
callbacks. It works with the existing ButterflyPACK extraction ABI and
retains CPU extraction of original entries, both child contributions and
the current negative Schur term. No ButterflyPACK source or CUDA dense-front
backend change is required.

On the matched 200-cubed layout, mode 1 reduces the three-trial median factor
time from 104.575 to 85.4017 seconds (18.33%). Median factor plus solve falls
from 141.7304 to 127.1683 seconds (10.27%). The complete record is in
HODLR_GPU_COMPACT_EXTRACTION.json.

The preceding profile, job 60475371, measured 27.00 mean-rank seconds in
native result redistribution and STRUMPACK collectives, plus 5.07 seconds
of host packing. These account for 59.5% of entry callback wall time. See
HODLR_GPU_OPTIMIZATION.rst for that profile and its timing limitations.

Data path
---------

The legacy path constructs a compact child intersection, redistributes it
into a full query-sized child-grid matrix filled with zeros, then packs and
redistributes that matrix again to the parent panel. The new path keeps the
intersection compact until the final parent owner:

1. The native callback already supplies each panel's 1-by-1 process map.
   STRUMPACK reuses this owner metadata and avoids rediscovering it with
   the parent metadata allreduce.
2. Global query lists are intersected with each child's update set. Output
   position maps retain unsorted order and every duplicate occurrence.
   Child members compute native requests; a final owner outside the child
   computes only the scatter maps it needs. Other inactive parent ranks
   avoid the index searches.
3. HODLR children use the existing native 1-by-1 output-map API. The native
   request contains child-local relay ranks only. A parent owner inside a
   child relays to itself; an outside owner uses a deterministic floor-mod
   translation to a valid child rank. All child members participate,
   including ranks with zero owned output.
4. Only relay ranks hold dense compact results. Exact sequential children
   use their existing extraction method on pruned queries. Other MPI child
   types retain their existing distributed extraction on those queries.
5. Compact local columns are appended directly to final parent-owner send
   buffers. One parent-communicator exchange carries original sparse values
   and both child contributions. The receiver adds through the saved output
   position maps in sparse/left/right order.

The child full-query zero matrices and their intermediate child exchange
are removed for owned HODLR results. ButterflyPACK's row fetch, reconstruction,
barrier and final native redistribution remain. With all output maps 1-by-1,
its existing whole-intersection redistribution is selected instead of the
general 2D output route. Performance therefore needs measurement; this is
not a native-redistribution bypass.

Mathematics, ownership and bounds
---------------------------------

F11, F12 and F21 still contain original sparse entries plus both child
Schur contributions. F22 extraction skips original update/update entries,
then adds the existing current-front negative Schur product through the
unchanged Butterfly extraction call. Pivoting, RHS layout, inverse
applications and the opt-in GPU exact local Schur GEMM are unchanged.

Parent ranks outside a child communicator never enter that child's native
collectives or appear in its process map. Transport to those owners occurs
only on the parent communicator. Existing shared process-tree ownership
within each front and full-solver GPU initialization are retained. There
are no new MPI_COMM_WORLD collectives or MPI calls in device evaluators.

The native owned-output wrapper validates query counts, indices, child-local
owners, dimension products and conservative native integer message bounds
before extraction. Nonowners receive empty matrices. Zero-sized intersections
remain in nonempty collective batches; ranks cannot skip a call based on
local empty work. The parent route also checks replicated query/message
bounds before entering child collectives. Messages retain the existing
MPI wrapper's scalar datatypes and size_t buffer offsets.

All compact buffers are owned by their extraction call. Parent exchange
completion precedes reads and buffer destruction. Final panel data stays
in the native callback's existing host storage for the subsequent GPU upload.
This introduces no additional asynchronous device-data lifetime.

Non-owner-local output layouts retain the legacy extraction path, including
the bounded full-image preparation option. Enabling compact extraction with
randomized matvec construction or non-HODLR solver compression is rejected.
The measured GPU configuration keeps OMP_NUM_THREADS=1 and STRUMPACK_USE_CUDA=OFF.

Focused regression
------------------

``test_HODLR_compact_extraction_mpi`` constructs actual HODLR children and
checks assembled parent panels against an independent long-double Gaussian
elimination Schur oracle. A grandparent extraction also checks the parent's
current negative Schur term. It runs both callback option settings and both
``skip_sparse`` choices, with nonzero original entries, duplicate/unsorted
queries crossing 32-entry tiles, empty panels, output guards, inactive ranks
and owners outside child communicators. The six-rank case uses 4/2 child
groups and explicitly tests negative floor-mod owner translation.

Job 60476615 passes this fixture at MPI counts 2, 4 and 6. Legacy and compact
panel values agree exactly, with maximum dense-oracle error 1.78e-15. The
same job runs three-RHS 100-cubed smoke comparisons and a 200-cubed diagnostic.

The expanded fixture in job 60477041 gives the children distinct, partially
overlapping update sets, including a noncontiguous set and nonempty requests
with an empty child intersection. It checks HODLR, distributed DenseMPI and
sequential Dense children, including the generic distributed fallback and
the sequential owned-output default. MPI counts 2, 4 and 6 each pass 370560
entry checks, with exact legacy/compact agreement and maximum oracle error
1.78e-15. The fixture is also registered as a two-process CPU CTest.

Matched performance and validation
----------------------------------

Job 60476615 uses one four-node allocation, 16 MPI ranks and 16 A100 GPUs,
with one rank per GPU and 16 physical CPU cores per rank. It runs compact
flags 0,1,1,0,0,1 with identical executable/native/BLAS hashes, bindings and
solver settings. Each flag has three passing trials. OMP_NUM_THREADS=1,
pthread OpenBLAS workers=16 and OPENBLAS_THREAD_TIMEOUT=28 are retained.

.. list-table:: Quiet 200-cubed medians, seconds
   :header-rows: 1
   :widths: 50 25 25

   * - Measurement
     - Compact off
     - Compact on
   * - Factor
     - 104.575
     - 85.4017
   * - Solve
     - 38.7919
     - 39.8545
   * - Factor plus solve, paired per trial
     - 141.7304
     - 127.1683
   * - GNU whole-process wall time
     - 153.49
     - 138.42

Factor ranges are 101.956--105.681 seconds off and 85.1553--89.6437 seconds
on. Paired totals are medians of each trial's factor-plus-solve, not sums of
the separate medians. Solve medians are reported separately above; the
measured whole-process improvement is 9.82%.

Both sides enable mode 2, GPU rectangular/host-panel paths and GPU local
dense Schur updates. They retain leaf size 1024, RecLR_leaf=5, BACA batch 64,
KNN=0/0, butterfly levels 0, minimum separator/front=8000/100000, compression
tolerance 1e-2, full-image cap 0, panel cap 64 MiB, QR SVD, heap fraction
0.85, GPU-aware MPI off, solver relative/absolute tolerances 1e-12 and at
most 100 refinement iterations. Add ``--hodlr_compact_extraction 1`` to that
previously validated command; the option remains off by default.

The two 100-cubed smoke runs, one diagnostic 200-cubed run and six quiet
200-cubed runs all pass with three RHS. Maximum relative solution error is
1.9892e-10 and maximum scaled residual is 3.78716e-10, below retained limits
1e-8 and 1e-6. The 100-cubed smoke timing is a single trial per flag and is
not promoted as a repeated performance result.

Diagnostic explanation
----------------------

The compact diagnostic has all 16 stage/native-counter files and consistent
exclusive wall accounting. Compared descriptively with the earlier profile
60475371, mean-rank GPU host-panel callback scope wall changes from 53.87 to
39.67 seconds. Native child result redistribution inside those callbacks
changes from 13.54 to 4.84 seconds; reconstruction plus V-row exchange stays
near 3.54--3.59 seconds. Current-Schur redistribution stays near 5.37--5.46
seconds, consistent with leaving that path unchanged.

Parent metadata allreduce (3.22 seconds) is replaced by compact mapping
(1.01 seconds). The intermediate child exchange/pack/unpack scopes disappear.
The final parent exchange increases from 3.37 to 5.48 seconds, so some
waiting moves there. These diagnostics use different executables/allocations
and slightly different query counts. They explain the change; the matched
quiet comparisons establish its performance benefit. Local timings include
communication and waiting and cannot be summed as an assured global saving.

Native and framework totals outside GPU host-panel callback intervals are
kept separate in the JSON. Compact callback exchange payload is 60.91 GB
across ranks. Baseline payload bytes were not instrumented, so a transfer-byte
reduction is not claimed. STRUMPACK adds no CUDA allocations or asynchronous
device buffers in this route. Native redistribution still remains; a future
local-fragment/device-output API would be a separate ButterflyPACK change.

Logs, job folders, frozen binaries and build directories stay locally ignored
and are excluded from the source/report commit.
