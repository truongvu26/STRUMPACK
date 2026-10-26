/* Device callback contract: nondefault streams, repeated/permuted ids, empty
 * blocks, complex asymmetry, noncompact output, and backend scratch ownership. */
#include "HODLR/GPUFrontEvaluator.hpp"
#include <bpack_gpu.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>
using namespace strumpack::HODLR;
void check(cudaError_t s) { if (s!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(s)); }
struct Scratch {
  std::vector<void*> live;
  static void* allocate(void* context,std::size_t bytes) {
    void* p=nullptr;
    if (cudaMalloc(&p,bytes)!=cudaSuccess) return nullptr;
    static_cast<Scratch*>(context)->live.push_back(p); return p;
  }
  static void release(void*,void*) {} // Reclaim only after the test's final fence.
  ~Scratch() { for (auto p:live) cudaFree(p); }
};
template<typename T> T value(double re,double im) { return T(re); }
template<> std::complex<double> value(double re,double im) { return {re,im}; }
template<typename T> void run() {
  constexpr int n=7,m=5,cols=4,ld=8;
  GPUFrontData<T> data; data.rows=n; data.ptr.push_back(0);
  std::vector<T> reference(n*n);
  for (int i=0;i<n;++i) {
    for (int j=0;j<n;++j) if ((i+2*j)%3==0) {
      data.col.push_back(j); data.sparse.push_back(value<T>(i+2*j+1,.3*i-.2*j));
      reference[i+j*n]=data.sparse.back();
    }
    data.ptr.push_back(data.col.size());
  }
  for (int j=0;j<n;++j) for (int i=0;i<n;++i) {
    const auto child=value<T>(.17*(i+1)*(j+2),.11*(i-j));
    data.contribution.push_back(child); reference[i+j*n]+=child;
  }
  auto evaluator=make_gpu_front_evaluator(std::move(data));
  cudaStream_t streams[2];
  for (auto& s:streams) check(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));
  std::int64_t ri[m]={6,1,1,4,0},ci[cols]={3,0,6,3};
  std::int64_t *dr,*dc; T* out;
  check(cudaMalloc(&dr,sizeof(ri))); check(cudaMalloc(&dc,sizeof(ci)));
  check(cudaMalloc(&out,ld*cols*sizeof(T)));
  check(cudaMemcpy(dr,ri,sizeof(ri),cudaMemcpyHostToDevice));
  check(cudaMemcpy(dc,ci,sizeof(ci),cudaMemcpyHostToDevice));
  std::vector<T> result(ld*cols,T(-999));
  check(cudaMemcpy(out,result.data(),result.size()*sizeof(T),cudaMemcpyHostToDevice));
  bpack_gpu_block blocks[2]{};
  blocks[0].m=m; blocks[0].n=cols; blocks[0].ld=ld;
  blocks[0].out=out; blocks[0].row_ids=dr; blocks[0].col_ids=dc;
  // The second block is empty, with NULL pointers deliberately left in place.
  Scratch scratch;
  bpack_gpu_block_batch batch{}; batch.count=2; batch.blocks=blocks;
  batch.allocate=Scratch::allocate; batch.release=Scratch::release; batch.allocator=&scratch;
  for (auto s:streams) { batch.stream=s; evaluator->evaluate_blocks(batch); }
  evaluator->finish();
  check(cudaMemcpy(result.data(),out,result.size()*sizeof(T),cudaMemcpyDeviceToHost));
  for (int j=0;j<cols;++j) for (int i=0;i<ld;++i) {
    const auto expected=i<m?reference[ri[i]+ci[j]*n]:T(-999);
    if (std::abs(result[i+j*ld]-expected)>1.e-12) throw std::runtime_error("Incorrect child-inclusive tile or overwritten padding");
  }
  if (evaluator->calls()!=2 || !evaluator->upload_bytes() || !evaluator->scratch_peak_bytes())
    throw std::runtime_error("Missing evaluator resource counters");
  // State destruction with a pending callback must fence its reads.
  batch.stream=streams[0]; evaluator->evaluate_blocks(batch); evaluator.reset();
  check(cudaFree(out)); check(cudaFree(dr)); check(cudaFree(dc));
  for (auto s:streams) check(cudaStreamDestroy(s));
}
int main() {
  try {
    run<double>(); run<std::complex<double>>();
    std::puts("GPU front evaluator device contract: PASS");
  } catch (const std::exception& e) { std::fprintf(stderr,"%s\n",e.what()); return 1; }
}
