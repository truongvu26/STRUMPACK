// Bounded FP64 Schur update for an exact local front. The caller retains all
// host matrix ownership; no MPI or STRUMPACK dense GPU backend is involved.
#ifndef STRUMPACK_GPU_DENSE_SCHUR_HPP
#define STRUMPACK_GPU_DENSE_SCHUR_HPP

#include <cstddef>

namespace strumpack { namespace HODLR {

struct DenseSchurGPUStats {
  bool used_gpu = false;
  int panel_columns = 0, panels = 0;
  // Explicit buffers only; cuBLAS/context workspace and BPACK pools excluded.
  std::size_t upload_bytes = 0, download_bytes = 0, device_peak_bytes = 0;
  double wall_seconds = 0., device_gemm_seconds = 0.;
  const char* fallback_reason = "";
};

// Compute C -= A*B using column-major host matrices, preserving nonzero C and
// leading-dimension padding. A is m-by-k, B k-by-n, and C m-by-n. Upload A once
// and stream bounded columns of B and C through the caller's current device.
// The 2-GiB cap is further limited by cudaMemGetInfo with a safety reserve.
//
// false means explicit buffer admission/allocation failed BEFORE changing any
// host data, so the caller may perform the complete CPU update. Other CUDA or
// cuBLAS resource and execution errors throw:
// earlier output panels may already be committed and MUST NOT be updated a
// second time. Successful return fences all output before parent extraction.
// A zero dimension is a successful no-op, allowing null pointers and no CUDA.
// Other scalar types remain on the existing CPU path at the front call site.
bool gpu_dense_schur_update
  (int m, int n, int k, const double* A, int lda, const double* B, int ldb,
   double* C, int ldc, DenseSchurGPUStats& stats,
   std::size_t budget_bytes = std::size_t(2) << 30);

}} // namespace strumpack::HODLR
#endif
