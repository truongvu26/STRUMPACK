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
template<typename T> void run(const MPIComm& comm) {
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
  opts.set_knn_hodlrbf(0);
  structured::ClusterTree tree(n); tree.refine(8);
  HODLR::HODLRMatrix<T> source(comm,tree,opts), matrix;
  source.set_gpu_front_evaluator(HODLR::make_gpu_front_evaluator(std::move(data)));
  matrix=std::move(source); // callback user pointer must survive this move.
  typename HODLR::HODLRMatrix<T>::elem_blocks_t host =
    [&](auto& I,auto& J,auto& B,auto&) {
      for (std::size_t b=0;b<B.size();++b)
        for (std::size_t j=0;j<J[b].size();++j)
          for (std::size_t i=0;i<I[b].size();++i) B[b](i,j)=reference(I[b][i],J[b][j]);
    };
  matrix.compress(host);
  if (!matrix.backend_statistics().evaluator_calls)
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
int main(int argc,char** argv) {
  MPI_Init(&argc,&argv);
  try {
    MPIComm world(MPI_COMM_WORLD),self(MPI_COMM_SELF);
    HODLR::HODLRMatrix<double>::initialize_gpu(world);
    run<double>(self); run<std::complex<double>>(self);
    if (world.is_root()) std::cout << "GPU front evaluator HODLR integration: PASS\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl; MPI_Abort(MPI_COMM_WORLD,1);
  }
  scalapack::Cblacs_exit(1); MPI_Finalize();
}
