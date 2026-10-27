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

The 150-cubed paired timing job is 60412989 and the stage-profile job is
60412991. Results are pending. The paired run uses the current frozen GPU
executable and the new Release executable on the same allocation, linked
to the same frozen ButterflyPACK library. The diagnostic executable adds
scopes and task timers and is not a quiet performance sample.

``backend_statistics()`` exposes panel calls, windows, upload bytes, peak
host staging and preparation/metadata/upload wall times. During stage
profiling these native counters are also printed on every front rank.
``prepare.host_panels`` scopes separate callback work from the enclosing
``*.construct.gpu_panels`` scopes. Native device-entry timing excludes host
preparation; transfer counters are reported separately.

Device-side child contribution extraction remains the next extension.
It requires explicit device leases and rank ownership for child factor
panels, preserving the same collective preparation order. The present path
provides a correctness reference and measurements for that work.

Provenance and hashes are in ``HODLR_GPU_HOST_PANELS.json``. Raw logs,
job folders, immutable executable/library snapshots and builds are retained
locally and excluded from commits.
