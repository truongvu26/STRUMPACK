/* Compare a registered, child-inclusive evaluator with an asymmetric dense
 * reference, including move ownership and real/complex N/T/C operations. */
#include "HODLR/HODLRMatrix.hpp"
#include "HODLR/GPUFrontEvaluator.hpp"
#include "HODLR/ButterflyMatrix.hpp"
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
using namespace strumpack;
template<typename T> T entry(double re,double im) { return T(re); }
template<> std::complex<double> entry(double re,double im) { return {re,im}; }
template<typename T> T conjugate(T x) { return x; }
template<> std::complex<double> conjugate(std::complex<double> x) { return std::conj(x); }
template<typename T> void run(const MPIComm& comm,bool panels=false,std::size_t budget=1024) {
  constexpr int n=64;
  HODLR::GPUFrontData<T> data; data.rows=n; data.ptr.push_back(0);
  DenseMatrix<T> reference(n,n); reference.zero();
  for (int i=0;i<n;++i) {
    for (int j=0;j<n;++j) if (i==j || j==i+1 || j==i-2) {
      const auto a=entry<T>(i==j?8.:.1*(1+i%3),i==j?0.:.03*(i-j));
      data.col.push_back(j); data.sparse.push_back(a); reference(i,j)=a;
    }
    data.ptr.push_back(data.col.size());
  }
  for (int j=0;j<n;++j) for (int i=0;i<n;++i) {
    const auto a=entry<T>(.0001*(i+1)*(j+2),.00003*(i-j));
    data.contribution.push_back(a); reference(i,j)+=a;
  }
  HODLR::HODLROptions<T> opts;
  opts.set_use_gpu(1); opts.set_leaf_size(8); opts.set_rel_tol(1.e-10);
  if (panels) opts.set_verbose(false);
  opts.set_knn_hodlrbf(0); opts.set_gpu_host_panels(panels); opts.set_gpu_panel_max_bytes(budget);
  structured::ClusterTree tree(n); tree.refine(8);
  HODLR::HODLRMatrix<T> source(comm,tree,opts), matrix;
  if (!panels) source.set_gpu_front_evaluator(HODLR::make_gpu_front_evaluator(std::move(data)));
  matrix=std::move(source); // callback user pointer must survive this move.
  typename HODLR::HODLRMatrix<T>::elem_blocks_t host =
    [&](auto& I,auto& J,auto& B,auto&) {
      for (std::size_t b=0;b<B.size();++b)
        for (std::size_t j=0;j<J[b].size();++j)
          for (std::size_t i=0;i<I[b].size();++i) B[b](i,j)=reference(I[b][i],J[b][j]);
    };
  matrix.compress(host);
  if (panels ? !matrix.backend_statistics().host_panel_calls : !matrix.backend_statistics().evaluator_calls)
    throw std::runtime_error("GPU evaluator not used in HODLR construction");
  DenseMatrix<T> x(n,3),rhs(n,3),actual(n,3),solved(n,3);
  for (int j=0;j<3;++j) for (int i=0;i<n;++i) x(i,j)=entry<T>((i+1.)/(n+j),.002*(i-j));
  for (auto op : {Trans::N,Trans::T,Trans::C}) {
    matrix.mult(op,x,actual);
    double diff=0.,norm=0.;
    for (int j=0;j<3;++j) for (int i=0;i<n;++i) {
      T sum=0.;
      for (int k=0;k<n;++k) sum+=(op==Trans::N?reference(i,k):
        (op==Trans::T?reference(k,i):conjugate(reference(k,i))))*x(k,j);
      if (op==Trans::N) rhs(i,j)=sum;
      diff+=std::norm(actual(i,j)-sum); norm+=std::norm(sum);
    }
    if (!std::isfinite(diff) || std::sqrt(diff/norm)>1.e-8)
      throw std::runtime_error("GPU-constructed HODLR multiply disagrees with child-inclusive reference");
  }
  // F22 preparation uses the sequential rectangular BF extraction helper.
  // Its local output count must survive metadata initialization: resetting it
  // to zero allocated a zero-length extraction buffer and corrupted the heap.
  HODLR::ButterflyMatrix<T> schur(matrix,matrix);
  typename HODLR::HODLRMatrix<T>::elem_blocks_t schur_entries =
    [](auto& I,auto& J,auto& B,auto&) {
      for (std::size_t b=0;b<B.size();++b)
        for (std::size_t j=0;j<J[b].size();++j)
          for (std::size_t i=0;i<I[b].size();++i)
            B[b](i,j)=entry<T>(.01*(I[b][i]+1)*(J[b][j]+2),.003*(double(I[b][i])-double(J[b][j])));
    };
  schur.compress(schur_entries);
  std::vector<std::vector<std::size_t>> rows{{5,0,5}},cols{{n-1,1}};
  DenseMatrix<T> padded(6,2); padded.fill(T(-999));
  std::vector<DenseMatrixWrapper<T>> tiles;
  tiles.emplace_back(3,2,padded.data(),padded.ld()); tiles[0].fill(T(7));
  schur.extract_add_elements(rows,cols,tiles);
  for (int j=0;j<2;++j) for (int i=0;i<6;++i) {
    const auto expected=i<3?T(7)+entry<T>(.01*(rows[0][i]+1)*(cols[0][j]+2),
      .003*(double(rows[0][i])-double(cols[0][j]))):T(-999);
    if (std::abs(padded(i,j)-expected)>1.e-8)
      throw std::runtime_error("Wrong child Schur extraction or overwritten padding");
  }
  matrix.factor(); matrix.solve(rhs,solved);
  double diff=0.,norm=0.;
  for (int j=0;j<3;++j) for (int i=0;i<n;++i) {
    diff+=std::norm(solved(i,j)-x(i,j)); norm+=std::norm(x(i,j));
  }
  if (!std::isfinite(diff) || std::sqrt(diff/norm)>1.e-8)
    throw std::runtime_error("GPU-constructed HODLR inverse disagrees with reference");
}
// Distributed callback metadata and owner-local staging, independent of
// the native BACA arithmetic: sparse A + two child terms - current Schur.
template<typename T> T front_entry(int i,int j) {
  return entry<T>(i==j?8.:(i==j+1?-.1:0.),0.)
    +entry<T>(.0001*(i+1)*(j+2),.00001*(i-j))
    +entry<T>(.0003*(1+i%3)*(1+j%5),.00002*(i+2*j))
    -entry<T>(.00003*(i+3)*(j+1),.000005*(i-3*j));
}
template<typename T> void distributed(const MPIComm& comm,std::size_t budget) {
  const int n=96;
  HODLR::HODLROptions<T> opts; opts.set_verbose(false); opts.set_use_gpu(1); opts.set_gpu_host_panels(true);
  opts.set_gpu_panel_max_bytes(budget); opts.set_leaf_size(8); opts.set_knn_hodlrbf(0); opts.set_rel_tol(1.e-10);
  structured::ClusterTree tree(n); tree.refine(8);
  HODLR::HODLRMatrix<T> original(comm,tree,opts),matrix(std::move(original));
  typename HODLR::HODLRMatrix<T>::delem_blocks_t entries=[](auto& I,auto& J,auto& B,auto&) {
    for (std::size_t b=0;b<B.size();++b) for (std::size_t j=0;j<B[b].lcols();++j)
      for (std::size_t i=0;i<B[b].lrows();++i)
        B[b](i,j)=front_entry<T>(I[b][B[b].rowl2g(i)],J[b][B[b].coll2g(j)]);
  };
  matrix.compress(entries);
  const auto& st=matrix.backend_statistics();
  if (!st.host_panel_calls || !st.host_panel_upload_bytes || st.host_panel_peak_bytes>budget ||
      comm.all_reduce(st.host_panel_windows,MPI_MAX)<=comm.all_reduce(st.host_panel_calls,MPI_MAX))
    throw std::runtime_error("Host panels did not exercise bounded multiple windows");
  const int local=matrix.lrows(),first=matrix.begin_row();
  auto xvalue=[](int i,int j){return entry<T>(.01*(i+1)*(j+1),.002*(i-j));};
  DenseMatrix<T> x(local,3),y(local,3),rhs(local,3),answer(local,3);
  for (int j=0;j<3;++j) for(int i=0;i<local;++i)x(i,j)=xvalue(first+i,j);
  for (auto op:{Trans::N,Trans::T,Trans::C}) {
    matrix.mult(op,x,y); double diff=0,norm=0;
    for (int j=0;j<3;++j) for(int i=0;i<local;++i) {
      T expected=0;
      for(int k=0;k<n;++k)expected+=(op==Trans::N?front_entry<T>(first+i,k):
          (op==Trans::T?front_entry<T>(k,first+i):conjugate(front_entry<T>(k,first+i))))*xvalue(k,j);
      if(op==Trans::N)rhs(i,j)=expected;
      diff+=std::norm(y(i,j)-expected);norm+=std::norm(expected);
    }
    const auto error=std::sqrt(comm.all_reduce(diff,MPI_SUM)/comm.all_reduce(norm,MPI_SUM));
    if(!std::isfinite(error)||error>1.e-8)throw std::runtime_error("Distributed host-panel construction has incorrect child-inclusive entries");
  }
  matrix.factor(); matrix.solve(rhs,answer); double diff=0,norm=0;
  for(int j=0;j<3;++j)for(int i=0;i<local;++i){diff+=std::norm(answer(i,j)-x(i,j));norm+=std::norm(x(i,j));}
  const auto error=std::sqrt(comm.all_reduce(diff,MPI_SUM)/comm.all_reduce(norm,MPI_SUM));
  if(!std::isfinite(error)||error>1.e-8)throw std::runtime_error("Distributed host-panel inverse failed");
}
int main(int argc,char** argv) {
  MPI_Init(&argc,&argv);
  try {
    MPIComm world(MPI_COMM_WORLD),self(MPI_COMM_SELF);
    HODLR::HODLRMatrix<double>::initialize_gpu(world);
    HODLR::HODLROptions<double> options; options.set_from_command_line(argc,argv);
    run<double>(self,options.gpu_host_panels(),options.gpu_panel_max_bytes());
    run<std::complex<double>>(self,options.gpu_host_panels(),options.gpu_panel_max_bytes());
    if(options.gpu_host_panels()) {
      distributed<double>(world,options.gpu_panel_max_bytes());
      distributed<std::complex<double>>(world,options.gpu_panel_max_bytes());
    }
    if (world.is_root()) std::cout << "GPU front evaluator HODLR integration: PASS\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl; MPI_Abort(MPI_COMM_WORLD,1);
  }
  scalapack::Cblacs_exit(1); MPI_Finalize();
}
