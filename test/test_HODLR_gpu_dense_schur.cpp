#include "HODLR/GPUDenseSchur.hpp"
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
using strumpack::HODLR::DenseSchurGPUStats;
using strumpack::HODLR::gpu_dense_schur_update;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

bool unchanged(const std::vector<double>& a, const std::vector<double>& b) {
  return a.size() == b.size() &&
    std::memcmp(a.data(), b.data(), a.size()*sizeof(double)) == 0;
}

struct Matrices {
  static constexpr int guard = 5;
  int m, n, k, lda, ldb, ldc;
  std::vector<double> a, b, c, a0, b0, c0;

  Matrices(int rows, int columns, int inner)
    : m(rows), n(columns), k(inner), lda(m+7), ldb(k+5), ldc(m+9),
      a(std::size_t(lda)*k+2*guard, -98765.125),
      b(std::size_t(ldb)*n+2*guard, -87654.25),
      c(std::size_t(ldc)*n+2*guard, -76543.5) {
    for (int p=0; p<k; ++p)
      for (int i=0; i<m; ++i)
        A()[i+p*lda] = std::sin(.071*(i+1)*(p+3)) + .01*(i-p);
    for (int j=0; j<n; ++j)
      for (int p=0; p<k; ++p)
        B()[p+j*ldb] = std::cos(.091*(p+2)*(j+1)) - .03*(p+j);
    for (int j=0; j<n; ++j)
      for (int i=0; i<m; ++i)
        C()[i+j*ldc] = .5 + double(i-2*j)/13.;
    a0=a; b0=b; c0=c;
  }

  double* A() { return a.data()+guard; }
  double* B() { return b.data()+guard; }
  double* C() { return c.data()+guard; }

  void verify() const {
    require(unchanged(a,a0) && unchanged(b,b0), "Schur update changed an input or its padding");
    const auto eps = std::numeric_limits<double>::epsilon();
    for (int j=0; j<n; ++j)
      for (int i=0; i<m; ++i) {
        long double expected=c0[guard+i+j*ldc], scale=std::abs(expected);
        for (int p=0; p<k; ++p) {
          const long double product =
            static_cast<long double>(a0[guard+i+p*lda])*b0[guard+p+j*ldb];
          expected -= product;
          scale += std::abs(product);
        }
        const long double actual=c[guard+i+j*ldc];
        require(std::isfinite(actual) &&
                std::abs(actual-expected) <= 128*eps*(1+scale),
                "GPU Schur update differs from independent long-double reference");
      }
    // Check every output byte outside the logical matrix, including both
    // end guards and the unused rows of each padded leading dimension.
    for (std::size_t pos=0; pos<c.size(); ++pos) {
      const bool in_storage=pos>=guard && pos<std::size_t(guard)+std::size_t(ldc)*n;
      const bool active=in_storage && (pos-guard)%ldc<std::size_t(m);
      if (!active)
        require(std::memcmp(&c[pos],&c0[pos],sizeof(double))==0,
                "GPU Schur update overwrote output padding or an end guard");
    }
  }
};

template<typename F> void invalid(F call) {
  bool rejected=false;
  try { call(); }
  catch (const std::invalid_argument&) { rejected=true; }
  require(rejected, "Invalid Schur dimensions, leading dimension or pointer accepted");
}

void check_argument_contract() {
  DenseSchurGPUStats stats;
  // These are host-only no-ops even without a CUDA device or valid pointers.
  for (const auto& shape : std::vector<std::vector<int>>{{0,3,2},{3,0,2},{3,2,0}}) {
    stats.used_gpu=true; stats.panels=9;
    require(gpu_dense_schur_update(shape[0],shape[1],shape[2],
            nullptr,0,nullptr,0,nullptr,0,stats), "Zero-dimension Schur update rejected");
    require(!stats.used_gpu && stats.panels==0, "Zero-dimension Schur update reported GPU work");
  }
  Matrices x(3,4,2);
  auto call=[&](int m,int n,int k,const double* a,int lda,
                const double* b,int ldb,double* c,int ldc) {
    gpu_dense_schur_update(m,n,k,a,lda,b,ldb,c,ldc,stats);
  };
  invalid([&] { call(-1,0,0,nullptr,0,nullptr,0,nullptr,0); });
  invalid([&] { call(0,-1,0,nullptr,0,nullptr,0,nullptr,0); });
  invalid([&] { call(0,0,-1,nullptr,0,nullptr,0,nullptr,0); });
  // Nonnull small buffers are sufficient: impossible padded host spans must
  // be rejected before either pointer access or a CUDA admission query.
  const int huge=std::numeric_limits<int>::max();
  invalid([&] { call(1,huge,1,x.A(),x.lda,x.B(),huge,x.C(),x.ldc); });
  invalid([&] { call(1,1,huge,x.A(),huge,x.B(),huge,x.C(),x.ldc); });
  invalid([&] { call(1,huge,1,x.A(),x.lda,x.B(),x.ldb,x.C(),huge); });
  invalid([&] { call(x.m,x.n,x.k,x.A(),x.m-1,x.B(),x.ldb,x.C(),x.ldc); });
  invalid([&] { call(x.m,x.n,x.k,x.A(),x.lda,x.B(),x.k-1,x.C(),x.ldc); });
  invalid([&] { call(x.m,x.n,x.k,x.A(),x.lda,x.B(),x.ldb,x.C(),x.m-1); });
  invalid([&] { call(x.m,x.n,x.k,nullptr,x.lda,x.B(),x.ldb,x.C(),x.ldc); });
  invalid([&] { call(x.m,x.n,x.k,x.A(),x.lda,nullptr,x.ldb,x.C(),x.ldc); });
  invalid([&] { call(x.m,x.n,x.k,x.A(),x.lda,x.B(),x.ldb,nullptr,x.ldc); });
  require(unchanged(x.a,x.a0) && unchanged(x.b,x.b0) && unchanged(x.c,x.c0),
          "Rejected argument check mutated host data");
}

void check_gpu_updates() {
  Matrices x(37,31,29);
  const std::size_t a_bytes=std::size_t(x.m)*x.k*sizeof(double);
  const std::size_t column_bytes=std::size_t(x.k+x.m)*sizeof(double);
  const std::size_t budget=a_bytes+3*column_bytes;
  DenseSchurGPUStats stats;
  require(gpu_dense_schur_update(x.m,x.n,x.k,x.A(),x.lda,x.B(),x.ldb,
          x.C(),x.ldc,stats,budget), "Small GPU Schur fixture was not admitted");
  require(stats.used_gpu && stats.panel_columns>0 && stats.panel_columns<x.n &&
          x.n%stats.panel_columns!=0 && stats.panels>1,
          "GPU fixture did not exercise multiple panels and an odd final panel");
  require(stats.panels==(x.n+stats.panel_columns-1)/stats.panel_columns &&
          stats.device_peak_bytes<=budget, "GPU Schur panel count or device budget incorrect");
  x.verify();

  Matrices single(9,7,17);
  DenseSchurGPUStats single_stats;
  require(gpu_dense_schur_update(single.m,single.n,single.k,single.A(),single.lda,
          single.B(),single.ldb,single.C(),single.ldc,single_stats),
          "Default-budget GPU Schur fixture was not admitted");
  require(single_stats.used_gpu, "Nonempty Schur fixture did not use the GPU");
  single.verify();

  Matrices rejected(37,31,29);
  DenseSchurGPUStats rejected_stats;
  require(!gpu_dense_schur_update(rejected.m,rejected.n,rejected.k,
          rejected.A(),rejected.lda,rejected.B(),rejected.ldb,rejected.C(),rejected.ldc,
          rejected_stats,a_bytes+column_bytes-1), "Insufficient device budget was accepted");
  require(!rejected_stats.used_gpu && unchanged(rejected.c,rejected.c0) &&
          unchanged(rejected.a,rejected.a0) && unchanged(rejected.b,rejected.b0),
          "GPU admission fallback changed host data");
  std::cout << "GPU Schur fixture: panels=" << stats.panels
            << " panel_columns=" << stats.panel_columns
            << " device_peak_bytes=" << stats.device_peak_bytes << '\n';
}
}

int main() {
  try {
    check_argument_contract();
    int devices=0;
    const auto status=cudaGetDeviceCount(&devices);
    if (status!=cudaSuccess || devices<1)
      throw std::runtime_error("GPU Schur fixture requires an available CUDA device");
    check_gpu_updates();
    std::cout << "PASS GPU dense Schur update\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL GPU dense Schur update: " << error.what() << '\n';
    return 1;
  }
}
