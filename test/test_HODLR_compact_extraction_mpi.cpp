// Exercise the production frontal extraction paths against an independent
// dense Schur-complement oracle.  All index requests are global indices.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "StrumpackOptions.hpp"
#include "HODLR/HODLRMatrix.hpp"
#include "sparse/PropMapSparseMatrix.hpp"
#include "sparse/fronts/FrontHODLRMPI.hpp"
#include "sparse/fronts/FrontDenseMPI.hpp"
#include "sparse/fronts/FrontDense.hpp"
#include "sparse/fronts/FrontMPI.hpp"

using namespace strumpack;
using Queries = std::vector<std::vector<std::size_t>>;
using Panel = DistributedMatrixWrapper<double>;
using MPIFront = FrontMPI<double,int>;
using HODLRFront = FrontHODLRMPI<double,int>;

namespace {
constexpr int order=128;
constexpr double guard=-987654321.25;
constexpr std::size_t guard_size=8;

double entry(int r, int c) {
  // Independent sibling separators have no direct coupling, as in a valid
  // elimination tree.  Both children couple to the parent and its updates.
  if ((r<32 && c>=32 && c<64) || (c<32 && r>=32 && r<64))
    return 0.;
  // Child update sets differ and contain holes.  Parent requests therefore
  // include rows/columns absent from either child; pruning must preserve
  // their requested positions, including duplicate positions.
  if ((r<32 && c>=112) || (c<32 && r>=112)) return 0.;
  if ((r>=32 && r<64 && c>=96 && c<104) ||
      (c>=32 && c<64 && r>=96 && r<104)) return 0.;
  if (r==c) return 12.+.01*r;
  return .007*(1+r%5)*(1+c%3)/15.
    +.003*std::sin(.13*(r+1))*std::cos(.17*(c+2))
    +.002*(r-c)/order;
}

class SparseFixture final : public PropMapSparseMatrix<double,int> {
public:
  SparseFixture() {
    this->n_=order;
    this->symm_sparse_=false;
    this->ptr_.push_back(0);
    for (int c=0;c<order;c++) {
      this->global_col_.push_back(c);
      for (int r=0;r<order;r++) {
        const auto a=entry(r,c);
        if (a!=0.) {
          this->ind_.push_back(r);
          this->val_.push_back(a);
        }
      }
      this->ptr_.push_back(this->ind_.size());
    }
    this->local_cols_=this->global_col_.size();
    this->nnz_=this->ind_.size();
  }
};

// Row-major Gaussian elimination, partial pivoting, and substitution in
// long double.  It deliberately uses neither STRUMPACK nor BLAS/LAPACK.
std::vector<long double> schur_contribution(int eliminated) {
  const int e=eliminated,u=order-e;
  std::vector<long double> a(e*e),x(e*u),out(u*u,0.);
  for (int r=0;r<e;r++) {
    for (int c=0;c<e;c++) a[r*e+c]=entry(r,c);
    for (int c=0;c<u;c++) x[r*u+c]=entry(r,e+c);
  }
  for (int k=0;k<e;k++) {
    int p=k;
    for (int r=k+1;r<e;r++)
      if (std::abs(a[r*e+k])>std::abs(a[p*e+k])) p=r;
    if (std::abs(a[p*e+k])<1.e-20L)
      throw std::runtime_error("singular independent dense oracle");
    if (p!=k) {
      for (int c=0;c<e;c++) std::swap(a[p*e+c],a[k*e+c]);
      for (int c=0;c<u;c++) std::swap(x[p*u+c],x[k*u+c]);
    }
    for (int r=k+1;r<e;r++) {
      const auto f=a[r*e+k]/a[k*e+k];
      a[r*e+k]=0.;
      for (int c=k+1;c<e;c++) a[r*e+c]-=f*a[k*e+c];
      for (int c=0;c<u;c++) x[r*u+c]-=f*x[k*u+c];
    }
  }
  for (int r=e-1;r>=0;r--)
    for (int c=0;c<u;c++) {
      for (int k=r+1;k<e;k++) x[r*u+c]-=a[r*e+k]*x[k*u+c];
      x[r*u+c]/=a[r*e+r];
    }
  for (int r=0;r<u;r++)
    for (int c=0;c<u;c++)
      for (int k=0;k<e;k++) out[r*u+c]-=entry(e+r,k)*x[k*u+c];
  return out;
}

std::vector<int> range(int first,int last) {
  std::vector<int> v(last-first);
  std::iota(v.begin(),v.end(),first);
  return v;
}

void make_queries(int first,Queries& I,Queries& J) {
  const int span=order-first;
  I.resize(10); J.resize(10);
  // Both dimensions cross the default block-cyclic tile boundary.  Request
  // order and duplicates must survive pruning, routing, and reconstruction.
  for (int q=0;q<4;q++) {
    for (int i=0;i<span+11-q;i++) I[q].push_back(first+(i*13+q*7)%span);
    for (int j=0;j<span+9+q;j++) J[q].push_back(first+(j*19+q*3)%span);
    I[q].insert(I[q].end(),{std::size_t(first),std::size_t(first)});
    J[q].insert(J[q].end(),{std::size_t(order-1),std::size_t(order-1)});
  }
  // Zero row, zero column, and zero-by-zero queries remain in the metadata.
  J[4]={std::size_t(first),std::size_t(order-1)};
  I[5]={std::size_t(order-1),std::size_t(first)};
  // Nonempty requests can still have an empty child intersection.  The last
  // block misses the left child by rows and the right child by columns.
  I[7]={112,127,112}; J[7]={std::size_t(first),111,127};
  I[8]={96,103,96}; J[8]={96,103,111};
  I[9]={112,127}; J[9]={96,103};
}

std::uint64_t check_extraction
  (const MPIComm& world,const MPIFront& front,const SparseFixture& A,
   int eliminated,int separator_end,const std::vector<long double>& oracle,
   bool skip_sparse,const std::string& label) {
  Queries I,J;
  make_queries(eliminated,I,J);
  // Owner 1 is outside the right child's [4,6) group on the six-rank test;
  // this specifically exercises negative floor-mod rank translation.
  std::vector<int> owners={0,world.size()-1,world.size()>1?1:0,
                          world.size()/2,0,world.size()-1,0,
                          world.size()-1,0,world.size()>1?1:0};
  BLACSGrid local(MPIComm(MPI_COMM_SELF),1);
  BLACSGrid inactive(MPIComm(MPI_COMM_NULL),1);
  std::vector<std::vector<double>> old_storage(I.size()),new_storage(I.size());
  std::vector<Panel> old_panels,new_panels;
  old_panels.reserve(I.size()); new_panels.reserve(I.size());
  for (std::size_t q=0;q<I.size();q++) {
    const auto n=I[q].size()*J[q].size();
    old_storage[q].assign(n+2*guard_size,guard);
    new_storage[q].assign(n+2*guard_size,guard);
    const auto* g=world.rank()==owners[q]?&local:&inactive;
    old_panels.emplace_back(g,I[q].size(),J[q].size(),
                            old_storage[q].data()+guard_size);
    new_panels.emplace_back(g,I[q].size(),J[q].size(),
                            new_storage[q].data()+guard_size);
  }
  front.extract_2d(A,I,J,old_panels,skip_sparse);
  front.extract_2d_compact(A,I,J,new_panels,owners,skip_sparse);
  double error=0.,route_error=0.;
  std::uint64_t checked=0;
  const int u=order-eliminated;
  for (std::size_t q=0;q<I.size();q++) {
    const auto n=I[q].size()*J[q].size();
    for (std::size_t k=0;k<guard_size;k++)
      if (old_storage[q][k]!=guard || new_storage[q][k]!=guard ||
          old_storage[q][guard_size+n+k]!=guard ||
          new_storage[q][guard_size+n+k]!=guard)
        throw std::runtime_error(label+": output guard modified");
    if (world.rank()!=owners[q]) {
      if (!std::all_of(old_storage[q].begin(),old_storage[q].end(),
                       [](double v){return v==guard;}) ||
          !std::all_of(new_storage[q].begin(),new_storage[q].end(),
                       [](double v){return v==guard;}))
        throw std::runtime_error(label+": nonowner output modified");
      continue;
    }
    for (std::size_t c=0;c<J[q].size();c++)
      for (std::size_t r=0;r<I[q].size();r++) {
        const auto gr=I[q][r],gc=J[q][c];
        const auto expected=double(oracle[(gr-eliminated)*u+gc-eliminated])
          +(!skip_sparse && (gr<std::size_t(separator_end) ||
                             gc<std::size_t(separator_end))?entry(gr,gc):0.);
        const auto k=guard_size+r+c*I[q].size();
        if (!std::isfinite(old_storage[q][k]) || !std::isfinite(new_storage[q][k]))
          throw std::runtime_error(label+": nonfinite output");
        error=std::max(error,std::abs(new_storage[q][k]-expected));
        error=std::max(error,std::abs(old_storage[q][k]-expected));
        route_error=std::max(route_error,std::abs(new_storage[q][k]-old_storage[q][k]));
        checked++;
      }
  }
  error=world.all_reduce(error,MPI_MAX);
  route_error=world.all_reduce(route_error,MPI_MAX);
  if (error>2.e-8 || route_error>2.e-10)
    throw std::runtime_error(label+": oracle error="+std::to_string(error)+
                             " route error="+std::to_string(route_error));
  if (world.is_root())
    std::cout << label << " PASS max_oracle_error=" << error
              << " max_route_error=" << route_error << '\n';
  return world.all_reduce(checked,MPI_SUM);
}

std::uint64_t check_tree(const MPIComm& world,const SparseFixture& A,
                         SPOptions<double> opts,bool compact,int kind) {
  opts.HODLR_options().set_compact_extraction(compact);
  const int P=world.size();
  const int Pl=kind==2?1:(P==6?4:P/2),Pr=P-Pl;
  auto left_comm=world.sub(0,Pl),right_comm=world.sub(Pl,Pr);
  auto parent_upd=range(96,128),left_upd=range(64,112),right_upd=range(64,96);
  auto right_tail=range(104,128);
  right_upd.insert(right_upd.end(),right_tail.begin(),right_tail.end());
  auto parent=std::make_unique<HODLRFront>(2,64,96,parent_upd,world,P);
  std::unique_ptr<Front<double,int>> left;
  if (kind==0) left=std::make_unique<HODLRFront>(0,0,32,left_upd,left_comm,Pl);
  else if (kind==1)
    left=std::make_unique<FrontDenseMPI<double,int>>(0,0,32,left_upd,left_comm,Pl);
  else left=std::make_unique<FrontDense<double,int>>(0,0,32,left_upd);
  auto right=std::make_unique<HODLRFront>(1,32,64,right_upd,right_comm,Pr);
  auto* l=left.get(); auto* r=right.get();
  parent->set_lchild(std::move(left)); parent->set_rchild(std::move(right));
  std::vector<int> sorder=range(0,order);
  parent->partition_fronts(opts,A,sorder.data(),true,0);
  if (!left_comm.is_null() &&
      l->multifrontal_factorization(A,opts)!=ReturnCode::SUCCESS)
    throw std::runtime_error("left child factorization failed");
  if (!right_comm.is_null() &&
      r->multifrontal_factorization(A,opts)!=ReturnCode::SUCCESS)
    throw std::runtime_error("right child factorization failed");
  world.barrier();
  const auto c64=schur_contribution(64),c96=schur_contribution(96);
  const auto suffix=std::string(" kind=")+
    (kind==0?"HODLR":kind==1?"DenseMPI":"Dense")+" compact="+(compact?"1":"0");
  std::uint64_t checked=0;
  for (bool skip:{false,true})
    checked+=check_extraction(world,*parent,A,64,96,c64,skip,
      "two children"+suffix+" skip_sparse="+std::to_string(skip));
  // Re-factor the actual parent using each callback dispatch choice.  Its
  // negative current Schur update must remain in F22 alongside both children.
  if (parent->multifrontal_factorization(A,opts)!=ReturnCode::SUCCESS)
    throw std::runtime_error("parent factorization failed");
  auto grand_upd=range(112,128);
  MPIFront grandparent(3,96,112,grand_upd,world,P);
  grandparent.set_lchild(std::move(parent));
  for (bool skip:{false,true})
    checked+=check_extraction(world,grandparent,A,96,112,c96,skip,
      "parent Schur"+suffix+" skip_sparse="+std::to_string(skip));
  return checked;
}
} // namespace

int main(int argc,char** argv) {
  int supplied=0;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_MULTIPLE,&supplied);
  try {
    {
      MPIComm world(MPI_COMM_WORLD);
      if (world.size()<2)
        throw std::runtime_error("compact extraction fixture requires >=2 MPI ranks");
      SPOptions<double> opts;
      opts.set_compression(CompressionType::HODLR);
      opts.set_verbose(false);
      auto& h=opts.HODLR_options();
      h.set_leaf_size(8); h.set_rel_tol(1.e-11); h.set_abs_tol(1.e-13);
      h.set_knn_hodlrbf(0); h.set_knn_lrbf(0); h.set_verbose(false);
      opts.set_from_command_line(argc,argv);
      h.set_verbose(false);
      if (h.use_gpu()) {
        h.set_gpu_rectangular(true); h.set_gpu_host_panels(true);
        HODLR::HODLRMatrix<double>::initialize_gpu(world);
      }
      SparseFixture A;
      std::uint64_t checked=0;
      for (bool compact:{false,true})
        for (int kind=0;kind<3;kind++) checked+=check_tree(world,A,opts,compact,kind);
      if (world.is_root())
        std::cout << "HODLR compact extraction MPI: PASS ranks=" << world.size()
                  << " checked=" << checked << " gpu=" << h.use_gpu() << '\n';
    }
    scalapack::Cblacs_exit(1);
    MPI_Finalize();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "HODLR compact extraction MPI: FAIL " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD,1);
    return 1;
  }
}
