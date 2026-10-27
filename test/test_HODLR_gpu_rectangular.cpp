/* Rectangular GPU BACA/multiply and device-resident Schur pipeline against
 * independent asymmetric real/complex dense references. */
#include "HODLR/HODLRMatrix.hpp"
#include "HODLR/ButterflyMatrix.hpp"
#include "HODLR/GPUFrontEvaluator.hpp"
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
using namespace strumpack;
template<typename T> T value(double r,double i) { return T(r); }
template<> std::complex<double> value(double r,double i) { return {r,i}; }
template<typename T> T conjv(T v) { return v; }
template<> std::complex<double> conjv(std::complex<double> v) { return std::conj(v); }
template<typename T> T entry(int i,int j,bool reverse) {
  // Two low-rank terms, including a nonsymmetric child contribution.
  return value<T>((reverse?.003:.002)*(i+1)*(j+2),.0007*(i-2*j))+
    value<T>(.1*(1+i%3),.02*(i%2))*value<T>(.03*(1+j%5),-.01*(j%3));
}
template<typename T> T vector_value(int i,int j) { return value<T>(.01*(i+1)*(j+1),.02*(i-j)); }
template<typename T> void check(const MPIComm& comm) {
  constexpr int n=48,u=64;
  HODLR::HODLROptions<T> opts; opts.set_use_gpu(1); opts.set_gpu_rectangular(true);
  opts.set_leaf_size(8); opts.set_rel_tol(1.e-10); opts.set_knn_hodlrbf(0); opts.set_knn_lrbf(0);
  structured::ClusterTree tree(n),update_tree(u); tree.refine(8); update_tree.refine(8);
  HODLR::HODLRMatrix<T> diagonal(comm,tree,opts),update(comm,update_tree,opts,&diagonal);
  typename HODLR::HODLRMatrix<T>::delem_blocks_t diagonal_entries=[](auto& I,auto& J,auto& B,auto&) {
    for (std::size_t b=0;b<B.size();++b) for (std::size_t j=0;j<B[b].lcols();++j)
      for (std::size_t i=0;i<B[b].lrows();++i)
        B[b](i,j)=I[b][B[b].rowl2g(i)]==J[b][B[b].coll2g(j)]?value<T>(2,.25):T(0);
  };
  diagonal.compress(diagonal_entries); diagonal.factor();
  HODLR::ButterflyMatrix<T> f12(diagonal,update),f21(update,diagonal);
  // Reject invalid callback contracts before any registration/upload can
  // expose an undersized or differently typed output buffer to a GPU kernel.
  {
    HODLR::GPUFrontData<T> bad;
    bad.rows=n; bad.cols=u-1; bad.ptr.resize(n+1,0); bad.contribution.resize(n*(u-1));
    bool rejected=false;
    try { f12.set_gpu_front_evaluator(HODLR::make_gpu_front_evaluator(std::move(bad))); }
    catch(const std::invalid_argument&) { rejected=true; }
    if (!rejected) throw std::runtime_error("Wrong rectangular evaluator shape accepted");
  }
  if constexpr (std::is_same<T,double>::value) {
    HODLR::GPUFrontData<std::complex<double>> bad;
    bad.rows=n; bad.cols=u; bad.ptr.resize(n+1,0); bad.contribution.resize(n*u);
    bool rejected=false;
    try { f12.set_gpu_front_evaluator(HODLR::make_gpu_front_evaluator(std::move(bad))); }
    catch(const std::invalid_argument&) { rejected=true; }
    if (!rejected) throw std::runtime_error("Wrong rectangular evaluator scalar type accepted");
  }
  for (bool reverse:{false,true}) {
    auto& matrix=reverse?f21:f12;
    HODLR::GPUFrontData<T> data;
    data.rows=matrix.rows(); data.cols=matrix.cols(); data.ptr.resize(data.rows+1,0);
    for (int j=0;j<data.cols;++j) for (int i=0;i<data.rows;++i)
      data.contribution.push_back(entry<T>(i,j,reverse));
    matrix.set_gpu_front_evaluator(HODLR::make_gpu_front_evaluator(std::move(data)));
    typename HODLR::ButterflyMatrix<T>::delem_blocks_t host=[&](auto& I,auto& J,auto& B,auto&) {
      for (std::size_t b=0;b<B.size();++b) for (std::size_t j=0;j<B[b].lcols();++j)
        for (std::size_t i=0;i<B[b].lrows();++i)
          B[b](i,j)=entry<T>(I[b][B[b].rowl2g(i)],J[b][B[b].coll2g(j)],reverse);
    };
    matrix.compress(host);
    for (auto op:{Trans::N,Trans::T,Trans::C}) for (int nrhs:{1,3}) {
      const bool normal=op==Trans::N;
      const int ni=normal?matrix.lcols():matrix.lrows(),no=normal?matrix.lrows():matrix.lcols();
      const int bi=normal?matrix.begin_col():matrix.begin_row(),bo=normal?matrix.begin_row():matrix.begin_col();
      const int global_ni=normal?matrix.cols():matrix.rows();
      DenseMatrix<T> px(ni+5,nrhs),py(no+7,nrhs); px.fill(T(-999)); py.fill(T(-999));
      DenseMatrixWrapper<T> x(ni,nrhs,px,2,0),y(no,nrhs,py,3,0);
      for (int j=0;j<nrhs;++j) for (int i=0;i<ni;++i) x(i,j)=vector_value<T>(bi+i,j);
      matrix.mult(op,x,y);
      double diff=0,norm=0;
      for (int j=0;j<nrhs;++j) for (int i=0;i<no;++i) {
        T expected=0;
        for (int k=0;k<global_ni;++k) {
          auto a=normal?entry<T>(bo+i,k,reverse):entry<T>(k,bo+i,reverse);
          if (op==Trans::C) a=conjv(a);
          expected+=a*vector_value<T>(k,j);
        }
        diff+=std::norm(y(i,j)-expected); norm+=std::norm(expected);
      }
      for (int j=0;j<nrhs;++j) for (int i=0;i<int(py.rows());++i)
        if ((i<3 || i>=3+no) && py(i,j)!=T(-999)) throw std::runtime_error("Rectangular GPU overwrote padding");
      const auto err=std::sqrt(comm.all_reduce(diff,MPI_SUM)/comm.all_reduce(norm,MPI_SUM));
      if (!std::isfinite(err) || err>1.e-8) throw std::runtime_error("Rectangular GPU multiply failed, error="+std::to_string(err));
    }
  }
  // Ownership and native factor state survive moving the rectangular owner.
  HODLR::ButterflyMatrix<T> moved(std::move(f12));
  for (auto op:{Trans::N,Trans::T,Trans::C}) {
    DenseMatrix<T> x(update.lrows(),3),y(update.lrows(),3);
    for (int j=0;j<3;++j) for (std::size_t i=0;i<x.rows();++i) x(i,j)=vector_value<T>(update.begin_row()+i,j);
    if (!moved.schur_product(op,diagonal,f21,x,y)) throw std::runtime_error("Device Schur pipeline did not run");
    double diff=0,norm=0;
    for (int j=0;j<3;++j) for (std::size_t i=0;i<y.rows();++i) {
      const int global_i=update.begin_row()+i;
      T expected=0;
      for (int k=0;k<n;++k) {
        T inner=0;
        for (int l=0;l<u;++l) {
          auto a=op==Trans::N?entry<T>(k,l,false):entry<T>(l,k,true);
          if (op==Trans::C) a=conjv(a);
          inner+=a*vector_value<T>(l,j);
        }
        auto a=op==Trans::N?entry<T>(global_i,k,true):entry<T>(k,global_i,false);
        if (op==Trans::C) a=conjv(a);
        expected+=a*inner/(op==Trans::C?conjv(value<T>(2,.25)):value<T>(2,.25));
      }
      diff+=std::norm(y(i,j)-expected); norm+=std::norm(expected);
    }
    const auto err=std::sqrt(comm.all_reduce(diff,MPI_SUM)/comm.all_reduce(norm,MPI_SUM));
    if (!std::isfinite(err) || err>1.e-8) throw std::runtime_error("Device Schur pipeline failed");
  }
  const auto& stats=moved.backend_statistics();
  const auto boundary=std::size_t(3)*2*update.lrows()*3*sizeof(T);
  if (stats.schur_calls!=3 || stats.schur_boundary_bytes!=boundary || stats.schur_seconds<=0)
    throw std::runtime_error("Device Schur boundary/timing accounting failed");
  if (comm.is_root()) std::cout << "Device Schur calls=" << stats.schur_calls
    << " host_boundary_bytes_local=" << stats.schur_boundary_bytes
    << " intermediate_device_peak_bytes_local=" << stats.schur_intermediate_peak_bytes << std::endl;
}
int main(int argc,char** argv) {
  MPI_Init(&argc,&argv);
  try {
    MPIComm comm(MPI_COMM_WORLD);
    HODLR::HODLRMatrix<double>::initialize_gpu(comm);
    check<double>(comm); check<std::complex<double>>(comm);
    // A front may use a strict subset of the solver ranks. Other ranks stay
    // outside its collectives, exercising the original integration boundary.
    if (comm.size()>1) {
      if (comm.is_root()) {
        MPIComm self(MPI_COMM_SELF);
        check<double>(self); check<std::complex<double>>(self);
      }
      MPI_Barrier(comm.comm());
    }
    if (comm.is_root()) std::cout << "Rectangular GPU construction/multiply and device Schur: PASS\n";
  } catch(const std::exception& e) { std::cerr << e.what() << std::endl; MPI_Abort(MPI_COMM_WORLD,1); }
  scalapack::Cblacs_exit(1); MPI_Finalize();
}
