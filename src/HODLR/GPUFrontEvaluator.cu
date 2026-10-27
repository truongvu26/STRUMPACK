#include "GPUFrontEvaluator.hpp"
#include <bpack_gpu.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace strumpack { namespace HODLR {
namespace {
// A callback cannot throw through the C/Fortran boundary. A device failure is
// fatal to this MPI job; never leave other front ranks waiting for a callback.
void checked(cudaError_t status) noexcept {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "GPU front evaluator: %s\n", cudaGetErrorString(status));
    std::abort();
  }
}
__device__ double plus(double a, double b) { return a+b; }
__device__ cuDoubleComplex plus(cuDoubleComplex a, cuDoubleComplex b) { return cuCadd(a,b); }

template<typename V> __global__ void assemble
(bpack_gpu_block b, int rows, int cols, int col_base, const std::int64_t* ptr,
 const std::int64_t* col, const V* sparse, const V* contribution) {
  const auto entries = static_cast<std::int64_t>(b.m)*b.n;
  for (auto k = static_cast<std::int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
       k < entries; k += static_cast<std::int64_t>(gridDim.x)*blockDim.x) {
    const auto r = k%b.m, c = k/b.m;
    const auto i = b.row_ids[r], j = b.col_ids[c]-col_base;
    // Inputs are checked at preparation; the backend guarantees valid ids.
    assert(i >= 0 && i < rows && j >= 0 && j < cols);
    V value = contribution[i+j*rows];
    auto lo = ptr[i], hi = ptr[i+1];
    while (lo < hi) {
      const auto mid = lo+(hi-lo)/2;
      if (col[mid] < j) lo = mid+1;
      else hi = mid;
    }
    if (lo < ptr[i+1] && col[lo] == j) value = plus(value, sparse[lo]);
    static_cast<V*>(b.out)[r+c*b.ld] = value;
  }
}

template<typename T> class Evaluator final : public GPUFrontEvaluator {
  using V = typename std::conditional<std::is_same<T,double>::value,
                                      double,cuDoubleComplex>::type;
public:
  explicit Evaluator(GPUFrontData<T>&& data) : data_(std::move(data)) {
    const auto n = data_.rows;
    if (!data_.cols) data_.cols=n;
    if (n <= 0 || data_.ptr.size() != std::size_t(n)+1 || data_.ptr[0] != 0 ||
        data_.cols<=0 || data_.contribution.size() != std::size_t(n)*data_.cols ||
        data_.sparse.size() != data_.col.size() ||
        data_.ptr.back() != static_cast<std::int64_t>(data_.col.size()))
      throw std::invalid_argument("Invalid GPU front image dimensions");
    for (int i=0; i<n; ++i) {
      if (data_.ptr[i] < 0 || data_.ptr[i] > data_.ptr[i+1] ||
          data_.ptr[i+1] > static_cast<std::int64_t>(data_.col.size()))
        throw std::invalid_argument("Invalid GPU front CSR offsets");
      for (auto k=data_.ptr[i]; k<data_.ptr[i+1]; ++k)
        if (data_.col[k] < 0 || data_.col[k] >= data_.cols ||
            (k>data_.ptr[i] && data_.col[k-1]>=data_.col[k]))
          throw std::invalid_argument("GPU front CSR must have sorted unique columns");
    }
    // No device payload allocation: the backend already reserves most memory.
    // All per-batch payloads come from its stream-ordered scratch allocator.
    const auto status = cudaEventCreateWithFlags(&done_,cudaEventDisableTiming);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
  }
  ~Evaluator() override { finish(); if (done_) checked(cudaEventDestroy(done_)); }
  bool complex_values() const override { return !std::is_same<T,double>::value; }
  int rows() const override { return data_.rows; }
  std::size_t calls() const override { return calls_; }
  std::size_t upload_bytes() const override { return uploads_; }
  std::size_t scratch_peak_bytes() const override { return peak_; }
  std::size_t resident_bytes() const override { return resident_bytes_; }
  void register_matrix(void** matrix) override {
#if defined(STRUMPACK_USE_HODLR_GPU_RECTANGULAR)
    upload_image(matrix,false);
#endif
    const int flags = 0; // The kernel consumes device ids; no symmetry assumption.
    if constexpr (std::is_same<T,double>::value)
      d_c_bpack_set_gpu_block_evaluator(matrix,callback,this,&flags);
    else z_c_bpack_set_gpu_block_evaluator(matrix,callback,this,&flags);
  }
  void register_rectangular(void** matrix) override {
#if defined(STRUMPACK_USE_HODLR_GPU_RECTANGULAR)
    col_base_=data_.rows;
    upload_image(matrix,true);
    const int flags=0;
    if constexpr (std::is_same<T,double>::value)
      d_c_bf_set_gpu_block_evaluator(matrix,callback,this,&flags);
    else z_c_bf_set_gpu_block_evaluator(matrix,callback,this,&flags);
#else
    throw std::invalid_argument("Rectangular GPU evaluator was not configured");
#endif
  }
  void finish() override {
    if (pending_) { checked(cudaEventSynchronize(done_)); pending_=false; }
  }
private:
  static void callback(const bpack_gpu_block_batch* batch, void* user) noexcept {
    static_cast<Evaluator*>(user)->evaluate_blocks(*batch);
  }
public:
  void evaluate_blocks(const bpack_gpu_block_batch& batch) noexcept override {
    if (!batch.count) return;
    if (!batch.allocate || !batch.release || !batch.blocks) std::abort();
    const auto stream = static_cast<cudaStream_t>(batch.stream);
    // A later callback may use a different stream. Join its predecessor before
    // either reads the same state or the backend recycles our scratch.
    if (pending_) checked(cudaStreamWaitEvent(stream,done_,0));
    const auto aligned=[](std::size_t n) { return (n+15)/16*16; };
    const auto ptr_bytes=data_.ptr.size()*sizeof(std::int64_t);
    const auto col_bytes=data_.col.size()*sizeof(std::int64_t);
    const auto sparse_bytes=data_.sparse.size()*sizeof(T);
    const auto dense_bytes=data_.contribution.size()*sizeof(T);
    const auto col_offset=aligned(ptr_bytes);
    const auto sparse_offset=aligned(col_offset+col_bytes);
    const auto dense_offset=aligned(sparse_offset+sparse_bytes);
    const auto bytes=dense_offset+dense_bytes;
    auto* storage=device_image_?static_cast<char*>(device_image_):static_cast<char*>(batch.allocate(batch.allocator,bytes));
    if (!storage) {
      std::fprintf(stderr,"GPU front evaluator: backend scratch allocation failed (%zu bytes)\n",bytes);
      std::abort();
    }
    auto* ptr=reinterpret_cast<std::int64_t*>(storage);
    auto* col=reinterpret_cast<std::int64_t*>(storage+col_offset);
    auto* sparse=reinterpret_cast<V*>(storage+sparse_offset);
    auto* dense=reinterpret_cast<V*>(storage+dense_offset);
    // Input vectors remain owned through finish(). Pageable uploads are staged
    // by CUDA; no borrowed callback descriptors survive this call.
    if (!device_image_) {
      checked(cudaMemcpyAsync(ptr,data_.ptr.data(),ptr_bytes,cudaMemcpyHostToDevice,stream));
      if (col_bytes) checked(cudaMemcpyAsync(col,data_.col.data(),col_bytes,cudaMemcpyHostToDevice,stream));
      if (sparse_bytes) checked(cudaMemcpyAsync(sparse,data_.sparse.data(),sparse_bytes,cudaMemcpyHostToDevice,stream));
      checked(cudaMemcpyAsync(dense,data_.contribution.data(),dense_bytes,cudaMemcpyHostToDevice,stream));
    }
    for (int k=0; k<batch.count; ++k) {
      const auto b=batch.blocks[k];
      if (b.m<=0 || b.n<=0) continue;
      if (b.ld<b.m || !b.out || !b.row_ids || !b.col_ids) std::abort();
      const auto count=static_cast<std::int64_t>(b.m)*b.n;
      const int blocks=static_cast<int>(std::min<std::int64_t>(65535,(count+255)/256));
      assemble<V><<<blocks,256,0,stream>>>(b,data_.rows,data_.cols,col_base_,ptr,col,sparse,dense);
      checked(cudaGetLastError());
    }
    checked(cudaEventRecord(done_,stream)); pending_=true;
    // The BPACK ABI promises reuse on this same ordered stream.
    if (!device_image_) batch.release(batch.allocator,storage);
    ++calls_;
    if (!device_image_) {
      uploads_+=ptr_bytes+col_bytes+sparse_bytes+dense_bytes;
      peak_=std::max(peak_,bytes);
    }
  }
private:
#if defined(STRUMPACK_USE_HODLR_GPU_RECTANGULAR)
  void upload_image(void** matrix,bool rectangular) {
    const auto aligned=[](std::size_t n) { return (n+15)/16*16; };
    const auto p=data_.ptr.size()*sizeof(std::int64_t),c=data_.col.size()*sizeof(std::int64_t);
    const auto z=data_.sparse.size()*sizeof(T),d=data_.contribution.size()*sizeof(T);
    const auto co=aligned(p),so=aligned(co+c),dd=aligned(so+z),bytes=dd+d;
    std::vector<char> image(bytes,0);
    std::memcpy(image.data(),data_.ptr.data(),p);
    if (c) std::memcpy(image.data()+co,data_.col.data(),c);
    if (z) std::memcpy(image.data()+so,data_.sparse.data(),z);
    if (d) std::memcpy(image.data()+dd,data_.contribution.data(),d);
    if constexpr (std::is_same<T,double>::value) {
      if (rectangular) d_c_bf_upload_entry_image(matrix,image.data(),&bytes,&device_image_);
      else d_c_bpack_upload_entry_image(matrix,image.data(),&bytes,&device_image_);
    } else {
      if (rectangular) z_c_bf_upload_entry_image(matrix,image.data(),&bytes,&device_image_);
      else z_c_bpack_upload_entry_image(matrix,image.data(),&bytes,&device_image_);
    }
    uploads_+=p+c+z+d; resident_bytes_=bytes;
  }
#endif
  std::size_t resident_bytes_=0;
  void* device_image_=nullptr; // backend-owned until matrix deletion
  GPUFrontData<T> data_;
  int col_base_=0;
  cudaEvent_t done_=nullptr;
  bool pending_=false;
  std::size_t calls_=0,uploads_=0,peak_=0;
};
} // namespace

template<typename T> std::shared_ptr<GPUFrontEvaluator>
make_gpu_front_evaluator(GPUFrontData<T>&& data) {
  if constexpr (std::is_same<T,double>::value || std::is_same<T,std::complex<double>>::value)
    return std::make_shared<Evaluator<T>>(std::move(data));
  else throw std::invalid_argument("GPU front evaluation requires double or complex double");
}
template std::shared_ptr<GPUFrontEvaluator> make_gpu_front_evaluator(GPUFrontData<double>&&);
template std::shared_ptr<GPUFrontEvaluator> make_gpu_front_evaluator(GPUFrontData<std::complex<double>>&&);
template std::shared_ptr<GPUFrontEvaluator> make_gpu_front_evaluator(GPUFrontData<float>&&);
template std::shared_ptr<GPUFrontEvaluator> make_gpu_front_evaluator(GPUFrontData<std::complex<float>>&&);
}} // namespace strumpack::HODLR
