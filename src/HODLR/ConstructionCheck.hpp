#ifndef STRUMPACK_HODLR_CONSTRUCTION_CHECK_HPP
#define STRUMPACK_HODLR_CONSTRUCTION_CHECK_HPP
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <numeric>
#include <string>
#include <type_traits>
#include <stdexcept>
#include <vector>
#include "GPUFrontEvaluator.hpp"

namespace strumpack { namespace HODLR {
// Diagnostic-only, matching collective query on the FRONT communicator.
// Check 64 stratified rows, all columns, in the application's index order.
// The original callback still includes child and current Schur contributions.
struct ConstructionProbe {
  int ni=0,nr=0,nc=0,np=1;
  std::int64_t local_count=0;
  std::vector<int> rows,cols,ms,ns,owners;
  int maps[3]={1,1,0};
  template<typename T,typename F> void evaluate(F callback,T* data,void* user) {
    callback(&ni,&nr,&nc,&local_count,rows.data(),cols.data(),data,
             ms.data(),ns.data(),owners.data(),&np,maps,user);
  }
};
template<typename T,typename Comm,typename Reference,typename Extract>
void verify_construction(const Comm& comm,int m,int n,const char* kind,
                         Reference reference,Extract extract,const GPUFrontEvaluator* image=nullptr) {
  const char* setting=std::getenv("STRUMPACK_HODLR_VERIFY_CONSTRUCTION");
  if(!setting || !*setting || !m || !n)return;
  char* end=nullptr; const double bound=std::strtod(setting,&end);
  if(end==setting || *end || !std::isfinite(bound) || bound<=0)
    throw std::invalid_argument("STRUMPACK_HODLR_VERIFY_CONSTRUCTION requires a positive error bound");
  ConstructionProbe p; const int count=std::min(m,64);
  for(int first=0;first<count;first+=8) {
    int size=std::min(8,count-first);p.ms.push_back(size);p.ns.push_back(n);p.owners.push_back(0);
    for(int i=first;i<first+size;++i)p.rows.push_back(1+(std::int64_t(2*i+1)*m)/(2*count));
    for(int j=0;j<n;++j)p.cols.push_back(j+1);
  }
  p.ni=p.ms.size();p.nr=p.rows.size();p.nc=p.cols.size();
  p.local_count=comm.is_root()?std::int64_t(count)*n:0;
  // Sequential callbacks can fill the complete query on every rank; distributed
  // callbacks fill only the owner. Allocate reference storage for either ABI.
  std::vector<T> expected(std::size_t(count)*n,T(0.));
  std::vector<T> actual(std::max<std::int64_t>(p.local_count,1),T(0.));
  auto ref_query=p,actual_query=p;
  reference(ref_query,expected.data());extract(actual_query,actual.data());
  double diff=0,norm=0;
  if(comm.is_root())for(std::size_t i=0;i<expected.size();++i) {
    const double a=std::abs(expected[i]),d=std::abs(expected[i]-actual[i]);
    norm+=a*a;diff+=d*d;
  }
  norm=comm.all_reduce(norm,MPI_SUM);diff=comm.all_reduce(diff,MPI_SUM);
  const double error=norm?std::sqrt(diff/norm):std::sqrt(diff);
  if(comm.is_root())std::cout<<"# construction_check kind="<<kind<<" rows="<<m<<" cols="<<n
    <<" procs="<<comm.size()<<" relative_sample_error="<<error<<" bound="<<bound<<std::endl;
  if(!std::isfinite(error)||error>bound) {
    const char* dump=std::getenv("STRUMPACK_HODLR_CHECK_DUMP");
    if(dump && *dump && std::uint64_t(m)*n*sizeof(T)<=64u*1024u*1024u) {
      ConstructionProbe full;full.ni=1;full.nr=m;full.nc=n;
      full.rows.resize(m);full.cols.resize(n);full.ms={m};full.ns={n};full.owners={0};
      std::iota(full.rows.begin(),full.rows.end(),1);std::iota(full.cols.begin(),full.cols.end(),1);
      full.local_count=comm.is_root()?std::int64_t(m)*n:0;
      std::vector<T> full_reference(std::size_t(m)*n,T(0.));
      std::vector<T> full_actual(std::max<std::int64_t>(full.local_count,1),T(0.));
      auto rq=full,aq=full;reference(rq,full_reference.data());extract(aq,full_actual.data());
      if(comm.is_root()) {
        std::vector<T> prepared(image?std::size_t(m)*n:0);
        if(image)image->copy_host_block(m,full.rows.data(),n,full.cols.data(),prepared.data());
        const char* rank=std::getenv("SLURM_PROCID");
        const std::string name=std::string(dump)+"."+kind+".rank"+(rank?rank:"0")+".bin";
        std::ofstream file(name,std::ios::binary);
        const std::uint64_t header[5]={0x31504d5544525453ULL,std::uint64_t(m),std::uint64_t(n),sizeof(T),
          std::uint64_t(!std::is_arithmetic<T>::value)|(image?2u:0u)};
        file.write(reinterpret_cast<const char*>(header),sizeof(header));
        for(const auto* values:{&full_reference,&full_actual,&prepared})
          file.write(reinterpret_cast<const char*>(values->data()),values->size()*sizeof(T));
        file.close();if(!file)throw std::runtime_error("Cannot write failed-front diagnostic capture");
        std::cout<<"# construction_capture file="<<name<<" rows="<<m<<" cols="<<n<<" has_image="<<bool(image)<<std::endl;
      }
    }
    throw std::runtime_error("Constructed HODLR/BF entries exceed the diagnostic error bound");
  }
}
}}
#endif
