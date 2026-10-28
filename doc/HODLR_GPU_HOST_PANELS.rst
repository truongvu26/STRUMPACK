Bounded host-prepared panels for GPU HODLR construction
======================================================

The opt-in host-panel path removes the full-image requirement for GPU BACA
compression. It applies to HODLR F11/F22 and level-zero rectangular F12/F21
blocks that do not have a registered full-image device evaluator. Original
matrix entries, child Schur contributions and the current negative Schur
term still come from STRUMPACK's existing batched host extraction callbacks.
This is hybrid GPU construction, not device-side frontal-entry assembly.

The previous 256 MiB image cap caused 94 large F22 and 22 blocks of each
rectangular orientation to use host compression at 150 cubed. The new path
prepares only requested BACA panels and uploads them to existing native
panel buffers. Full small images remain available when they fit the cap.
No complete large frontal image is all-gathered or retained by this path.

Build and select
----------------

Use ButterflyPACK branch ``feature/gpu-host-panels``, based on the validated
cuSOLVER resource-reuse branch. Configure STRUMPACK with
``STRUMPACK_USE_HODLR_GPU_HOST_PANELS=ON``. Configure-time link checks require
all real/complex registration, completion and statistics symbols. The flag
is off by default and STRUMPACK's own dense CUDA backend can remain off.
The panel interface itself needs no application CUDA compilation; existing
full-image GPU evaluation has its separate CUDA build flag.

For the sparse Poisson driver, use the existing GPU settings plus::

   --hodlr_use_gpu 1 --hodlr_gpu_rectangular 1 --hodlr_butterfly_levels 0
   --hodlr_gpu_front_max_bytes 268435456
   --hodlr_gpu_host_panels 1 --hodlr_gpu_panel_max_bytes 67108864
   --hodlr_quiet

The image cap controls full prepared images; the panel cap controls the sum
of local host values staged in one collective window. It excludes native
compressed factors, BACA workspaces and request metadata. Set the image cap
to zero to exercise panels for every eligible block. The default 85% native
pool is retained, with one shared cuSOLVER resource per process.

Collective sequence and ownership
--------------------------------

Native BACA iterations rendezvous on the full front communicator until all
ranks have finished their local blocks. Empty owners continue participating.
Column panels, row panels, the optional second column phase and KNN seeding
therefore execute in a compatible order across independent native subgroups.
Merging and factor redistribution retain the existing native algorithms.
Dense leaves continue on the existing host extraction path.

For each phase, the backend gathers the requested application ids from the
device, partitions local requests into bounded windows, and exchanges only
request metadata over the front communicator. Every rank invokes the same
host callback metadata. Each owner receives only its requested values, with
one-by-one BLACS grid maps matching the existing STRUMPACK extraction
interface. Rectangular column ids have their merged-tree row base removed
before entering the host callback.

The callback computes the complete frontal entries. F11/F12/F21 include
original sparse entries and child contributions; F22 includes child
contributions and this front's negative Schur product without adding sparse
A(u,u). The same extraction lambda used by the CPU path is called, avoiding
a separate stencil-only formula.

Each owner uploads its tiles to the native BACA output buffers on the
backend stream. A stream fence precedes reuse/destruction of host staging.
Construction-only callback leases are cleared before their captured Schur
objects and child work can be released. Native factors remain resident for
factorization, multiplies and solves. Shared process trees and compatible
BLACS grids are preserved, and no hidden world-communicator initialization
is added. Existing GPU block callbacks keep their rank-local contract.

The initial interface retains one OpenMP thread, butterfly level zero and
BACA variants 4/5 with BF_entry_n15 zero. Adaptive balanced-piece construction
and native entry diagnostics are rejected. Kernel constructors require their
separate path; panel construction uses batched extraction callbacks.

Evidence and profiling
----------------------

Focused job 60412476 passed on A100 40 GB GPUs, two ranks per GPU:

* Real/complex N/T/C products and inverse applications against independent
  asymmetric references containing sparse and child/Schur terms.
* One/three/four ranks, strict subcommunicators while other ranks wait,
  matrix moves, padded one/three RHS and forced multiple windows.
* A 1,024-byte staging cap was respected in the direct tests. A 64-square
  complex case used 192 windows and a 96-square case used 320, demonstrating
  that requests were split rather than silently materialized in full.
* Full 40-cubed three-RHS sparse solves with the baseline image path and
  both panel BACA variants passed. Panel variant 5 had relative error
  8.26439e-12 and scaled residual 3.35673e-10; variant 4 had 4.67681e-11
  and 7.85427e-9 respectively.

Job 60411461 stopped at configuration validation because direct fixtures
initially retained verbose diagnostics. It is preserved as a failed setup
attempt, and was corrected before the successful run.

The intermediate 100-cubed check, job 60414054, passed on four ranks sharing
two A100 40 GB GPUs. It exercised 528 HODLR fronts and three RHS, with
factorization 250.040 seconds, solve 4.48121 seconds, relative error
1.76318e-10 and scaled residual 2.86074e-8. All-rank panel counters and
staging checks passed. This diagnostic case is not a paired speed comparison.

The 150-cubed stage-profile job 60412991 passed with factorization
421.617 seconds, solve 5.92057 seconds, relative error 5.64166e-10 and
scaled residual 4.05087e-8. All 32 ranks passed numerical, binding and
exclusive-scope accounting checks. All 94 large F22 blocks, 22 blocks of
each rectangular orientation and three large F11 blocks used panels;
the remaining blocks retained full-image GPU evaluation. No measured
HODLR construction scope used the host-compression fallback.

Paired timing job 60412989 passed. It used the current frozen GPU executable
and the new Release executable on the same allocation, linked to the same
frozen ButterflyPACK library. Both cases used four nodes, 32 ranks and 16
A100 40 GB GPUs, two ranks per GPU, three RHS, and the same tolerances.

.. list-table:: Quiet 150-cubed timing, seconds
   :header-rows: 1

   * - Backend
     - Factor
     - Solve
     - Factor plus solve
   * - Current GPU image/host fallback
     - 690.838
     - 4.78237
     - 695.62037
   * - GPU image/panels
     - 423.214
     - 5.29268
     - 428.50668

Factor-plus-solve time decreased 38.4%, a 1.623x ratio. This is one paired
trial, not a repeated benchmark. Baseline relative error was 6.15349e-10
and residual 4.12792e-8; panels gave 2.28022e-10 and 1.72091e-8. The panel
case needed ten refinement iterations versus nine for the baseline, and
its solve time increased. Both passed the same accuracy checks. The
diagnostic executable adds scopes and task timers and is not a quiet
performance sample.

``backend_statistics()`` exposes panel calls, windows, upload bytes, peak
host staging and preparation/metadata/upload wall times. During stage
profiling these native counters are also printed on every front rank.
``prepare.host_panels`` scopes separate callback work from the enclosing
``*.construct.gpu_panels`` scopes. Native device-entry timing excludes host
preparation; transfer counters are reported separately.

The diagnostic profile partitions mean local factor wall time as follows.
These are exclusive rank wall times, including waits, not isolated GPU
execution times or a global critical path:

* Host panel callbacks: 151.714 seconds, 36.0%.
* Other full-image preparation: 27.736 seconds, 6.6%.
* GPU BACA panel construction outside host callbacks: 44.351 seconds, 10.5%.
* Full-image device-evaluator compression: 51.601 seconds, 12.2%.
* Traversal, dense fronts, setup and remaining work: 128.584 seconds, 30.5%.
* Host randomized Schur compression outside the device operator:
  15.462 seconds, 3.7%.
* Native GPU factorization and compound device Schur applications together:
  2.188 seconds, 0.5%.

Native counters recorded 73.899 GiB of uploaded panel values over all ranks,
including repeated requested values. This excludes child-extraction MPI
traffic. Maximum local host-value staging was 63.921 MiB against the 64 MiB
cap. Metadata processing, index download, synchronization and exchange took
3.210 mean rank seconds; uploads and staging-release fences took 0.330.
These native timers overlap the phase budget and must not be added again.
Maximum logical retained full-image payload was 2.661 GiB per rank; this
does not measure native factors, workspaces or pool reservations.

The next targets are host child/current-Schur entry extraction and subtree
balance. Inclusive task timers report 91.28 mean rank seconds in
``get_submatrix_2d`` and 67.70 in ``extract_Schur``; their nested values are
not an additional exclusive budget. Before parent front 1254450, local
gaps from the last completed compression scope range from 0.904 to
157.386 seconds. These include hierarchy setup and subtree-arrival waits,
not solely MPI time. The report computes gaps from local time differences
without assuming synchronized clocks across nodes.

GPU LU tuning, panel metadata exchange and upload bandwidth have much
smaller measured budgets. Device-side child extraction needs a separate
ownership and stream-lifetime implementation; the current counters provide
its correctness reference and a measured reason to pursue it.

Device-side child contribution extraction remains the next extension.
It requires explicit device leases and rank ownership for child factor
panels, preserving the same collective preparation order. The present path
provides a correctness reference and measurements for that work.

Provenance and hashes are in ``HODLR_GPU_HOST_PANELS.json``. Raw logs,
job folders, immutable executable/library snapshots and builds are retained
locally and excluded from commits.
