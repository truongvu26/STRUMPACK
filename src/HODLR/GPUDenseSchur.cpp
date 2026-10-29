#include "GPUDenseSchur.hpp"

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>

namespace strumpack { namespace HODLR {
namespace {
using Clock = std::chrono::steady_clock;

void checked(cudaError_t status, const char* operation) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("GPU dense Schur ") + operation +
                             ": " + cudaGetErrorString(status));
}
void checked(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string("GPU dense Schur ") + operation +
                             ": cuBLAS status " + std::to_string(int(status)));
}

struct Resources {
  double* data = nullptr;
  cudaStream_t stream = nullptr;
  cublasHandle_t handle = nullptr;
  cudaEvent_t begin = nullptr, end = nullptr;
  ~Resources() {
    // This also fences any copy queued before an execution error. Destruction
    // never masks that error, and never reruns an update on host output.
    if (stream) cudaStreamSynchronize(stream);
    if (handle) cublasDestroy(handle);
    if (begin) cudaEventDestroy(begin);
    if (end) cudaEventDestroy(end);
    if (data) cudaFree(data);
    if (stream) cudaStreamDestroy(stream);
  }
};

bool update(int m, int n, int k, const double* A, int lda,
            const double* B, int ldb, double* C, int ldc,
            DenseSchurGPUStats& stats, std::size_t budget) {
  if (m < 0 || n < 0 || k < 0)
    throw std::invalid_argument("GPU dense Schur dimensions must be nonnegative");
  if (!m || !n || !k) return true;
  if (!A || !B || !C || lda < m || ldb < k || ldc < m)
    throw std::invalid_argument("Invalid GPU dense Schur host matrices or leading dimensions");
  const auto host_limit = std::size_t(std::numeric_limits<std::ptrdiff_t>::max())/sizeof(double);
  const auto invalid_span = [host_limit](int rows, int columns, int ld) {
    return std::size_t(rows)>host_limit ||
      std::size_t(columns-1)>(host_limit-std::size_t(rows))/std::size_t(ld);
  };
  if (invalid_span(m,k,lda) || invalid_span(k,n,ldb) || invalid_span(m,n,ldc))
    throw std::invalid_argument("GPU dense Schur host matrix span overflows addressable storage");

  const auto limit = std::numeric_limits<std::size_t>::max()/sizeof(double);
  if (std::size_t(m) > limit/std::size_t(k) ||
      std::size_t(m) > limit-std::size_t(k)) {
    stats.fallback_reason = "size_overflow"; return false;
  }
  const auto a_bytes = std::size_t(m)*k*sizeof(double);
  const auto column_bytes = (std::size_t(m)+k)*sizeof(double);
  // Bound the requested footprint before querying or initializing CUDA. This
  // is also a deterministic, host-preserving fallback for a zero/tiny budget.
  budget = std::min(budget,std::size_t(2) << 30);
  if (a_bytes > budget || column_bytes > budget-a_bytes) {
    stats.fallback_reason = "configured_budget"; return false;
  }
  std::size_t free = 0, total = 0;
  checked(cudaMemGetInfo(&free,&total),"memory query");
  const auto reserve = std::max(std::size_t(256) << 20,free/8);
  budget = std::min(budget,free > reserve ? free-reserve : std::size_t(0));
  if (a_bytes > budget || column_bytes > budget-a_bytes) {
    stats.fallback_reason = "available_device_memory"; return false;
  }
  const int width = int(std::min(std::size_t(n),(budget-a_bytes)/column_bytes));
  const auto b_bytes = std::size_t(k)*width*sizeof(double);
  const auto c_bytes = std::size_t(m)*width*sizeof(double);
  const auto bytes = a_bytes+b_bytes+c_bytes;
  Resources resource;
  const auto allocation = cudaMalloc(reinterpret_cast<void**>(&resource.data),bytes);
  if (allocation == cudaErrorMemoryAllocation) {
    // Clear the recoverable allocation error before subsequent BPACK calls.
    cudaGetLastError();
    stats.fallback_reason = "device_allocation"; return false;
  }
  checked(allocation,"buffer allocation");
  checked(cudaStreamCreateWithFlags(&resource.stream,cudaStreamNonBlocking),"stream creation");
  checked(cublasCreate(&resource.handle),"handle creation");
  checked(cublasSetStream(resource.handle,resource.stream),"stream selection");
  checked(cublasSetPointerMode(resource.handle,CUBLAS_POINTER_MODE_HOST),"pointer mode");
  checked(cublasSetMathMode(resource.handle,CUBLAS_DEFAULT_MATH),"FP64 math mode");
  checked(cudaEventCreate(&resource.begin),"begin event creation");
  checked(cudaEventCreate(&resource.end),"end event creation");
  auto* dA = resource.data;
  auto* dB = dA+std::size_t(m)*k;
  auto* dC = dB+std::size_t(k)*width;
  stats.panel_columns = width;
  stats.device_peak_bytes = bytes;
  checked(cudaMemcpy2DAsync(dA,std::size_t(m)*sizeof(double),A,std::size_t(lda)*sizeof(double),
                            std::size_t(m)*sizeof(double),k,cudaMemcpyHostToDevice,resource.stream),"A upload");
  stats.upload_bytes += a_bytes;
  const double alpha = -1., beta = 1.;
  for (int first = 0; first < n; ) {
    const int columns = std::min(width,n-first);
    checked(cudaMemcpy2DAsync(dB,std::size_t(k)*sizeof(double),B+std::size_t(first)*ldb,
                              std::size_t(ldb)*sizeof(double),std::size_t(k)*sizeof(double),
                              columns,cudaMemcpyHostToDevice,resource.stream),"B panel upload");
    checked(cudaMemcpy2DAsync(dC,std::size_t(m)*sizeof(double),C+std::size_t(first)*ldc,
                              std::size_t(ldc)*sizeof(double),std::size_t(m)*sizeof(double),
                              columns,cudaMemcpyHostToDevice,resource.stream),"C panel upload");
    checked(cudaEventRecord(resource.begin,resource.stream),"GEMM begin record");
    checked(cublasDgemm(resource.handle,CUBLAS_OP_N,CUBLAS_OP_N,m,columns,k,
                        &alpha,dA,m,dB,k,&beta,dC,m),"DGEMM");
    checked(cudaEventRecord(resource.end,resource.stream),"GEMM end record");
    checked(cudaMemcpy2DAsync(C+std::size_t(first)*ldc,std::size_t(ldc)*sizeof(double),
                              dC,std::size_t(m)*sizeof(double),std::size_t(m)*sizeof(double),
                              columns,cudaMemcpyDeviceToHost,resource.stream),"C panel download");
    // Host output becomes visible before this helper returns or a parent
    // samples child contributions. No buffer outlives its owner or stream.
    checked(cudaStreamSynchronize(resource.stream),"panel completion");
    float milliseconds = 0.;
    checked(cudaEventElapsedTime(&milliseconds,resource.begin,resource.end),"GEMM timing");
    stats.device_gemm_seconds += milliseconds/1000.;
    stats.upload_bytes += std::size_t(columns)*(std::size_t(k)+m)*sizeof(double);
    stats.download_bytes += std::size_t(columns)*m*sizeof(double);
    ++stats.panels;
    first += columns;
  }
  stats.used_gpu = true;
  return true;
}
} // namespace

bool gpu_dense_schur_update
  (int m, int n, int k, const double* A, int lda, const double* B, int ldb,
   double* C, int ldc, DenseSchurGPUStats& stats, std::size_t budget_bytes) {
  stats = DenseSchurGPUStats{};
  const auto started = Clock::now();
  const bool completed = update(m,n,k,A,lda,B,ldb,C,ldc,stats,budget_bytes);
  stats.wall_seconds = std::chrono::duration<double>(Clock::now()-started).count();
  return completed;
}

}} // namespace strumpack::HODLR
