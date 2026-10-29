// GPU exact subtrees must return complete, additive child Schur entries to
// HODLR callbacks. References below use independent long-double elimination.
#include "StrumpackOptions.hpp"
#include "sparse/PropMapSparseMatrix.hpp"
#include "sparse/fronts/FrontGPU.hpp"
#include "sparse/fronts/FrontDense.hpp"
#include "sparse/fronts/FrontHODLR.hpp"
#include "sparse/fronts/FrontHODLRMPI.hpp"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

using namespace strumpack;
namespace {
using FrontT=Front<double,int>;
using GPUFront=FrontGPU<double,int>;
using HFront=FrontHODLR<double,int>;
using MFront=FrontMPI<double,int>;
using HMFront=FrontHODLRMPI<double,int>;
using Matrix=DenseMatrix<double>;
using Wrapper=DenseMatrixWrapper<double>;
using Queries=std::vector<std::vector<std::size_t>>;
constexpr int order=128;
constexpr double guard=-987654321.25;

void require(bool condition,const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
std::vector<int> range(int first,int last) {
  std::vector<int> out(last-first);std::iota(out.begin(),out.end(),first);return out;
}
double entry(int r,int c) {
  if ((r<32 && c>=32 && c<64) || (c<32 && r>=32 && r<64)) return 0.;
  if ((r<32 && c>=112) || (c<32 && r>=112)) return 0.;
  if ((r>=32 && r<64 && c>=96 && c<104) ||
      (c>=32 && c<64 && r>=96 && r<104)) return 0.;
  // Force pivoting in one exact child, rather than testing diagonal LU only.
  if (r==0 && c==0) return .0001;
  if (r==1 && c==0) return .2;
  if (r==0 && c==1) return .3;
  if (r==c) return 12.+.01*r;
  return .007*(1+r%5)*(1+c%3)/15.+.003*std::sin(.13*(r+1))*
    std::cos(.17*(c+2))+.002*(r-c)/order;
}
class SparseFixture final: public PropMapSparseMatrix<double,int> {
public:
  SparseFixture() {
    this->n_=order;this->symm_sparse_=false;this->ptr_.push_back(0);
    for (int c=0;c<order;++c) {
      this->global_col_.push_back(c);
      for (int r=0;r<order;++r) if (entry(r,c)!=0.) {
        this->ind_.push_back(r);this->val_.push_back(entry(r,c));
      }
      this->ptr_.push_back(this->ind_.size());
    }
    this->local_cols_=order;this->nnz_=this->ind_.size();
  }
};
// Row-major pivoted elimination; no STRUMPACK or BLAS/LAPACK is used.
std::vector<long double> solve_reference(std::vector<long double> a,
                                        std::vector<long double> x,
                                        int n,int nrhs) {
  for (int k=0;k<n;++k) {
    int p=k;
    for (int r=k+1;r<n;++r) if (std::abs(a[r*n+k])>std::abs(a[p*n+k])) p=r;
    require(std::abs(a[p*n+k])>1.e-20L,"singular independent oracle");
    if (p!=k) {
      for (int c=0;c<n;++c) std::swap(a[p*n+c],a[k*n+c]);
      for (int c=0;c<nrhs;++c) std::swap(x[p*nrhs+c],x[k*nrhs+c]);
    }
    for (int r=k+1;r<n;++r) {
      const auto f=a[r*n+k]/a[k*n+k];a[r*n+k]=0.;
      for (int c=k+1;c<n;++c) a[r*n+c]-=f*a[k*n+c];
      for (int c=0;c<nrhs;++c) x[r*nrhs+c]-=f*x[k*nrhs+c];
    }
  }
  for (int r=n-1;r>=0;--r) for (int c=0;c<nrhs;++c) {
    for (int k=r+1;k<n;++k) x[r*nrhs+c]-=a[r*n+k]*x[k*nrhs+c];
    x[r*nrhs+c]/=a[r*n+r];
  }
  return x;
}
std::vector<long double> contribution(int eliminated) {
  const int e=eliminated,u=order-e;
  std::vector<long double> a(e*e),x(e*u),s(u*u,0.);
  for (int r=0;r<e;++r) {
    for (int c=0;c<e;++c) a[r*e+c]=entry(r,c);
    for (int c=0;c<u;++c) x[r*u+c]=entry(r,e+c);
  }
  x=solve_reference(std::move(a),std::move(x),e,u);
  // Contributions deliberately exclude the original sparse A(upd,upd).
  for (int r=0;r<u;++r) for (int c=0;c<u;++c)
    for (int k=0;k<e;++k) s[r*u+c]-=entry(e+r,k)*x[k*u+c];
  return s;
}
void queries(int first,Queries& I,Queries& J) {
  const auto f=std::size_t(first);
  I={{127,f,111,f,104,126},{},{f,127},{},{112,127,112},{96,103,96}};
  J={{f+1,127,f+1,104},{f,127},{},{},{96,103,111},{96,103,127}};
}
// Keep exactly the factors checked by the oracle when the parent is later
// assembled. This test adapter avoids refactoring away the exercised path.
class CachedGPUFront final: public GPUFront {
public:
  using GPUFront::GPUFront;
  ReturnCode multifrontal_factorization(const CompressedSparseMatrix<double,int>& A,
      const SPOptions<double>& opts,int etree_level=0,int task_depth=0) override {
    if (factored_) return ReturnCode::SUCCESS;
    const auto code=GPUFront::multifrontal_factorization(A,opts,etree_level,task_depth);
    factored_=code==ReturnCode::SUCCESS;return code;
  }
private:
  bool factored_=false;
};
std::unique_ptr<GPUFront> local_tree(bool mixed) {
  auto upd=range(96,128),left_upd=range(64,112),right_upd=range(64,96);
  auto tail=range(104,128);right_upd.insert(right_upd.end(),tail.begin(),tail.end());
  auto root=std::make_unique<CachedGPUFront>(2,64,96,upd);
  std::unique_ptr<FrontT> left,right;
  if (mixed) {
    left=std::make_unique<FrontDense<double,int>>(0,0,32,left_upd);
    right=std::make_unique<HFront>(1,32,64,right_upd);
  } else {
    left=std::make_unique<GPUFront>(0,0,32,left_upd);
    right=std::make_unique<GPUFront>(1,32,64,right_upd);
  }
  root->set_lchild(std::move(left));root->set_rchild(std::move(right));return root;
}
void check_local_entries(GPUFront& front,int first=96) {
  const auto oracle=contribution(first);const int span=order-first;
  Queries I,J;queries(first,I,J);
  // Missing update indices must leave pre-existing panel data unchanged.
  I[0].insert(I[0].begin()+1,first-1);J[0].insert(J[0].begin()+2,first-1);
  std::vector<std::vector<double>> storage(I.size());std::vector<Wrapper> blocks;
  blocks.reserve(I.size());
  for (std::size_t q=0;q<I.size();++q) {
    const auto ld=I[q].size()+3;
    storage[q].assign(4+ld*J[q].size()+4,guard);
    blocks.emplace_back(I[q].size(),J[q].size(),storage[q].data()+4,ld);
    blocks.back().fill(.375);
  }
  front.extract_CB_sub_matrix_blocks(I,J,blocks,0);
  for (std::size_t q=0;q<I.size();++q) {
    const auto rows=I[q].size(),ld=rows+3,cols=J[q].size();
    for (std::size_t p=0;p<storage[q].size();++p) {
      const bool body=p>=4 && p<4+ld*cols && (p-4)%ld<rows;
      if (!body) require(storage[q][p]==guard,"GPU extraction overwrote padding");
    }
    for (std::size_t c=0;c<cols;++c) for (std::size_t r=0;r<rows;++r) {
      double expected=.375;
      if (I[q][r]>=std::size_t(first) && J[q][c]>=std::size_t(first))
        expected+=double(oracle[(I[q][r]-first)*span+J[q][c]-first]);
      require(std::isfinite(blocks[q](r,c)) && std::abs(blocks[q](r,c)-expected)<2.e-8,
              "additive GPU Schur extraction differs from independent oracle");
    }
  }
  // The owned local-child wrapper must also work with no output on inactive ranks.
  std::vector<Matrix> output;std::vector<int> owners(I.size(),0);
  require(front.get_submatrix_2d_owned(I,J,owners,output,true),"GPU local owned layout rejected");
  for (std::size_t q=0;q<I.size();++q)
    for (std::size_t c=0;c<J[q].size();++c) for (std::size_t r=0;r<I[q].size();++r)
      require(std::abs(output[q](r,c)-(blocks[q](r,c)-.375))<2.e-10,"owned GPU entries differ from additive wrapper");
  output.clear();
  require(front.get_submatrix_2d_owned(I,J,owners,output,false) && output.empty(),
          "inactive local GPU child produced output");
}
void check_solve_after_release(std::unique_ptr<GPUFront> child,const SparseFixture& A,
                               const SPOptions<double>& opts,int first=96) {
  auto empty=range(128,128);
  FrontDense<double,int> root(3,first,128,empty);root.set_lchild(std::move(child));
  require(root.multifrontal_factorization(A,opts)==ReturnCode::SUCCESS,"full mixed-tree factor failed");
  Matrix rhs(order,3),exact(order,3);
  for (int c=0;c<3;++c) for (int r=0;r<order;++r) exact(r,c)=.1+.003*r+.02*c+std::sin(.07*(r+2*c));
  for (int c=0;c<3;++c) for (int r=0;r<order;++r) {
    long double b=0.;for (int k=0;k<order;++k) b+=entry(r,k)*exact(k,c);rhs(r,c)=double(b);
  }
  std::vector<long double> dense(order*order),reference_rhs(order*3);
  for (int r=0;r<order;++r) {
    for (int c=0;c<order;++c) dense[r*order+c]=entry(r,c);
    for (int c=0;c<3;++c) reference_rhs[r*3+c]=rhs(r,c);
  }
  const auto reference=solve_reference(std::move(dense),std::move(reference_rhs),order,3);
  // Dense parent extend-add has released the GPU child's Schur. The GPU
  // LU/coupling/pivot storage must remain valid for all three RHS.
  root.multifrontal_solve(rhs);
  long double error=0.,norm=0.,known_error=0.;
  for (int c=0;c<3;++c) for (int r=0;r<order;++r) {
    error+=std::pow(static_cast<long double>(rhs(r,c))-reference[r*3+c],2);
    norm+=std::pow(reference[r*3+c],2);
    known_error+=std::pow(reference[r*3+c]-exact(r,c),2);
  }
  require(std::isfinite(error) && std::sqrt(error/norm)<2.e-8 && std::sqrt(known_error/norm)<2.e-11,
          "GPU factors invalid after Schur release or independent three-RHS solve");
}
void check_memory_fallback(const SparseFixture& A,const SPOptions<double>& opts) {
  std::unique_ptr<GPUFront> leaf;
  std::string trace;
  {
    struct Ballast {
      std::vector<void*> allocations;
      ~Ballast() { for (auto* p:allocations) cudaFree(p); }
    } ballast;
    std::size_t free=0,total=0;
    require(cudaMemGetInfo(&free,&total)==cudaSuccess,"GPU ballast memory query failed");
    const std::size_t target=std::size_t(128)<<20;
    require(free>target,"GPU fixture has too little memory for controlled ballast");
    // Bounded chunks avoid requiring one very large contiguous allocation.
    while (free>target+(std::size_t(1)<<20)) {
      const auto bytes=std::min(std::size_t(1)<<30,free-target);
      void* data=nullptr;
      const auto status=cudaMalloc(&data,bytes);
      if (status!=cudaSuccess) {
        cudaGetLastError();
        throw std::runtime_error("Unable to reserve GPU ballast for deterministic admission test");
      }
      ballast.allocations.push_back(data);
      require(cudaMemGetInfo(&free,&total)==cudaSuccess,"GPU ballast followup query failed");
    }
    auto upd=range(32,128);leaf=std::make_unique<CachedGPUFront>(0,0,32,upd);
    std::ostringstream output;
    struct Capture {
      std::streambuf* old;
      explicit Capture(std::ostream& stream):old(std::cout.rdbuf(stream.rdbuf())) {}
      ~Capture() { std::cout.rdbuf(old); }
    };
    {
      Capture capture(output);
      require(leaf->multifrontal_factorization(A,opts)==ReturnCode::SUCCESS,
              "memory-admission CPU fallback factor failed");
    }
    trace=output.str();
    require(trace.find("cpu_fallback_nodes=1")!=std::string::npos &&
            trace.find("split_gpu_nodes=0")!=std::string::npos,
            "controlled memory pressure did not exercise the CPU admission fallback");
  } // Return all ballast before extraction, parent assembly, or solve.
  std::cout<<trace;
  check_local_entries(*leaf,32);
  check_solve_after_release(std::move(leaf),A,opts,32);
}
void check_mpi_entries(const MPIComm& world,const MFront& parent,const SparseFixture& A,
                       int first,int sep_end,const std::vector<long double>& oracle) {
  Queries I,J;queries(first,I,J);const int u=order-first;
  std::vector<int> owners={world.size()-1,0,0,0,world.size()/2,0};
  BLACSGrid active(MPIComm(MPI_COMM_SELF),1),inactive(MPIComm(MPI_COMM_NULL),1);
  for (bool skip_sparse:{false,true}) {
    std::vector<std::vector<double>> old_values(I.size()),new_values(I.size());
    std::vector<DistributedMatrixWrapper<double>> old_blocks,new_blocks;
    old_blocks.reserve(I.size());new_blocks.reserve(I.size());
    for (std::size_t q=0;q<I.size();++q) {
      const auto size=I[q].size()*J[q].size();old_values[q].assign(size+8,guard);new_values[q].assign(size+8,guard);
      const auto* grid=world.rank()==owners[q]?&active:&inactive;
      old_blocks.emplace_back(grid,I[q].size(),J[q].size(),old_values[q].data()+4);
      new_blocks.emplace_back(grid,I[q].size(),J[q].size(),new_values[q].data()+4);
    }
    parent.extract_2d(A,I,J,old_blocks,skip_sparse);
    parent.extract_2d_compact(A,I,J,new_blocks,owners,skip_sparse);
    double error=0.;
    for (std::size_t q=0;q<I.size();++q) {
      const auto size=I[q].size()*J[q].size();
      for (std::size_t p=0;p<size+8;++p) {
        if (world.rank()!=owners[q] || p<4 || p>=size+4)
          require(old_values[q][p]==guard && new_values[q][p]==guard,"MPI GPU extraction changed nonowned output/guard");
      }
      if (world.rank()!=owners[q]) continue;
      for (std::size_t c=0;c<J[q].size();++c) for (std::size_t r=0;r<I[q].size();++r) {
        const int gr=I[q][r],gc=J[q][c];
        const auto expected=double(oracle[(gr-first)*u+gc-first])+
          (!skip_sparse && (gr<sep_end || gc<sep_end)?entry(gr,gc):0.);
        const auto p=4+r+c*I[q].size();
        require(std::isfinite(old_values[q][p]) && std::isfinite(new_values[q][p]),"nonfinite MPI GPU child entries");
        error=std::max(error,std::abs(old_values[q][p]-expected));
        error=std::max(error,std::abs(new_values[q][p]-expected));
      }
    }
    require(world.all_reduce(error,MPI_MAX)<2.e-8,"MPI GPU child legacy/compact entries differ from oracle");
  }
}
void check_mpi_parent(const MPIComm& world,const SparseFixture& A,SPOptions<double> opts) {
  if (world.size()<2) return;
  for (bool compact:{false,true}) {
    // Front constructors take ownership of these update-index vectors.
    auto upd=range(96,128),left_upd=range(64,112),right_upd=range(64,96);
    auto tail=range(104,128);right_upd.insert(right_upd.end(),tail.begin(),tail.end());
    opts.HODLR_options().set_compact_extraction(compact);
    auto parent=std::make_unique<HMFront>(2,64,96,upd,world,world.size());
    auto left=std::make_unique<GPUFront>(0,0,32,left_upd),right=std::make_unique<GPUFront>(1,32,64,right_upd);
    auto* l=left.get();auto* r=right.get();parent->set_lchild(std::move(left));parent->set_rchild(std::move(right));
    auto order_map=range(0,order);parent->partition_fronts(opts,A,order_map.data(),true,0);
    if (world.rank()==0) require(l->multifrontal_factorization(A,opts)==ReturnCode::SUCCESS,"MPI left GPU child factor failed");
    if (world.rank()==world.size()-1) require(r->multifrontal_factorization(A,opts)==ReturnCode::SUCCESS,"MPI right GPU child factor failed");
    check_mpi_entries(world,*parent,A,64,96,contribution(64));
    require(parent->multifrontal_factorization(A,opts)==ReturnCode::SUCCESS,"HODLR parent of GPU children factor failed");
    auto grand_upd=range(112,128);MFront grand(3,96,112,grand_upd,world,world.size());grand.set_lchild(std::move(parent));
    check_mpi_entries(world,grand,A,96,112,contribution(96));
  }
}
}
int main(int argc,char** argv) {
  int supplied=0;MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&supplied);
  try {
    {
      MPIComm world(MPI_COMM_WORLD);int devices=0;
      require(supplied==MPI_THREAD_MULTIPLE,"GPU exact fixture requires MPI_THREAD_MULTIPLE");
      require(cudaGetDeviceCount(&devices)==cudaSuccess && devices==1,
              "GPU exact fixture requires one visible CUDA GPU per MPI rank");
      SPOptions<double> opts;opts.set_compression(CompressionType::HODLR);opts.set_verbose(false);opts.disable_openmp_tree();
      auto& h=opts.HODLR_options();h.set_use_gpu(1);h.set_gpu_dense_fronts(true);
      h.set_gpu_rectangular(true);h.set_gpu_host_panels(true);
      h.set_leaf_size(8);h.set_rel_tol(1.e-12);h.set_abs_tol(1.e-14);
      h.set_knn_hodlrbf(0);h.set_knn_lrbf(0);h.set_verbose(false);h.set_compression_algorithm(HODLR::CompressionAlgorithm::ELEMENT_EXTRACTION);
      HODLR::HODLRMatrix<double>::initialize_gpu(world);
      SparseFixture A;
      for (bool mixed:{false,true}) {
        auto tree=local_tree(mixed);auto order_map=range(0,order);tree->partition_fronts(opts,A,order_map.data(),true,0);
        require(tree->multifrontal_factorization(A,opts)==ReturnCode::SUCCESS,"GPU subtree factorization failed");
        check_local_entries(*tree);check_solve_after_release(std::move(tree),A,opts);
      }
      check_mpi_parent(world,A,opts);
      const bool memory_test=argc==2 && std::string(argv[1])=="--check_cpu_fallback";
      require(argc==1 || memory_test,"unknown GPU exact fixture argument");
      if (memory_test) check_memory_fallback(A,opts);
      if (world.is_root()) std::cout << "GPU exact front: PASS ranks=" << world.size()
        << " independent_Schur=1 mixed_subtrees=1 additive_and_owned=1 three_RHS_after_release=1\n";
    }
    scalapack::Cblacs_exit(1);MPI_Finalize();return 0;
  } catch (const std::exception& e) {
    std::cerr << "GPU exact front: FAIL " << e.what() << '\n';MPI_Abort(MPI_COMM_WORLD,1);return 1;
  }
}
