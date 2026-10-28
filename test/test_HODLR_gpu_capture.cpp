// Replay an assembled front independently of sparse traversal and children.
// Compare GPU matvec before/after extraction with the extracted representation.
#include "HODLR/HODLRMatrix.hpp"
#include "HODLR/GPUFrontEvaluator.hpp"
#include <fstream>
#include <iostream>
#include <numeric>
#include <cmath>
#include <algorithm>
using namespace strumpack;
double error(const DenseMatrix<double>& x,const DenseMatrix<double>& y) {
  double d=0,s=0;
  for(std::size_t j=0;j<x.cols();++j)for(std::size_t i=0;i<x.rows();++i) {
    d+=std::pow(x(i,j)-y(i,j),2);s+=std::pow(y(i,j),2);
  }
  return std::sqrt(d/(s?s:1));
}
DenseMatrix<double> apply(const std::vector<double>& a,int n,const DenseMatrix<double>& x,Trans op) {
  DenseMatrix<double> y(n,x.cols());y.zero();
  for(std::size_t j=0;j<x.cols();++j)for(int k=0;k<n;++k)for(int i=0;i<n;++i)
    y(i,j)+=(op==Trans::N?a[i+std::size_t(k)*n]:a[k+std::size_t(i)*n])*x(k,j);
  return y;
}
int main(int argc,char** argv) {
  MPI_Init(&argc,&argv);
  try {
    if(argc!=2)throw std::runtime_error("Usage: test_HODLR_gpu_capture captured-front.bin");
    std::ifstream file(argv[1],std::ios::binary);std::uint64_t h[5];
    file.read(reinterpret_cast<char*>(h),sizeof(h));
    if(!file || h[0]!=0x31504d5544525453ULL || h[1]!=h[2] || h[3]!=8 || (h[4]&1))
      throw std::runtime_error("Replay requires a square real-double construction capture");
    const int n=h[1];std::vector<double> a(std::size_t(n)*n),old(a.size());
    file.read(reinterpret_cast<char*>(a.data()),a.size()*sizeof(double));
    file.read(reinterpret_cast<char*>(old.data()),old.size()*sizeof(double));
    if(!file)throw std::runtime_error("Truncated capture input");
    std::vector<double> col_error(n);
    for(int j=0;j<n;++j)for(int i=0;i<n;++i)col_error[j]+=std::pow(a[i+std::size_t(j)*n]-old[i+std::size_t(j)*n],2);
    const int c0=std::max_element(col_error.begin(),col_error.end())-col_error.begin();col_error[c0]=-1;
    const int c1=std::max_element(col_error.begin(),col_error.end())-col_error.begin();
    MPIComm world(MPI_COMM_WORLD),self(MPI_COMM_SELF);HODLR::HODLRMatrix<double>::initialize_gpu(world);
    DenseMatrix<double> x(n,3);x.zero();x(c0,0)=1;x(c1,1)=1;
    for(int i=0;i<n;++i)x(i,2)=1./std::sqrt(double(n));
    auto expected=apply(a,n,x,Trans::N);
    for(int leaf:{5,4})for(int gpu:{0,1})for(int trial=0;trial<4;++trial) {
      HODLR::HODLROptions<double> opts;opts.set_verbose(false);opts.set_use_gpu(gpu);
      opts.set_leaf_size(128);opts.set_rel_tol(1.e-5);opts.set_lr_leaf(leaf);opts.set_knn_hodlrbf(0);
      structured::ClusterTree tree(n);tree.refine(128);
      HODLR::HODLRMatrix<double> matrix(self,tree,opts);
      if(gpu) {
        HODLR::GPUFrontData<double> data;data.rows=n;data.ptr.resize(n+1,0);data.contribution=a;
        matrix.set_gpu_front_evaluator(HODLR::make_gpu_front_evaluator(std::move(data)));
      }
      HODLR::HODLRMatrix<double>::elem_blocks_t entries=[&](auto& I,auto& J,auto& B,auto&) {
        for(std::size_t b=0;b<B.size();++b)for(std::size_t j=0;j<J[b].size();++j)
          for(std::size_t i=0;i<I[b].size();++i)B[b](i,j)=a[I[b][i]+J[b][j]*n];
      };
      matrix.compress(entries);
      DenseMatrix<double> before(n,3),after(n,3);matrix.mult(Trans::N,x,before);
      std::vector<std::vector<std::size_t>> ids(1,std::vector<std::size_t>(n));std::iota(ids[0].begin(),ids[0].end(),0);
      std::vector<DenseMatrix<double>> extracted;extracted.emplace_back(n,n);matrix.extract_elements(ids,ids,extracted);
      std::vector<double> values(a.size());
      double diff=0,norm=0;for(int j=0;j<n;++j)for(int i=0;i<n;++i) {
        values[i+std::size_t(j)*n]=extracted[0](i,j);
        diff+=std::pow(values[i+std::size_t(j)*n]-a[i+std::size_t(j)*n],2);norm+=std::pow(a[i+std::size_t(j)*n],2);
      }
      auto extracted_product=apply(values,n,x,Trans::N);matrix.mult(Trans::N,x,after);
      std::cout<<"capture_replay rank="<<world.rank()<<" leaf="<<leaf<<" gpu="<<gpu<<" trial="<<trial
        <<" extraction_error="<<std::sqrt(diff/norm)<<" before_error="<<error(before,expected)
        <<" multiply_vs_extraction="<<error(before,extracted_product)<<" extraction_changed_matrix="<<error(after,before)<<std::endl;
    }
  }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;MPI_Abort(MPI_COMM_WORLD,1);}
  scalapack::Cblacs_exit(1);MPI_Finalize();
}
