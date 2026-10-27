#ifndef STRUMPACK_GPU_HOST_PANELS_HPP
#define STRUMPACK_GPU_HOST_PANELS_HPP
#include "HODLRMatrix.hpp"
#include "StageProfile.hpp"
#if defined(STRUMPACK_USE_HODLR_GPU_HOST_PANELS)
#include <bpack_gpu.h>
#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <type_traits>
namespace strumpack { namespace HODLR {
// Construction-only lease of existing child-inclusive host callbacks. The
// native backend prepares bounded panels collectively, then compresses on GPU.
template<typename T> class GPUHostPanelSession {
public:
  GPUHostPanelSession(bool enabled,void** matrix,MPI_Fint comm,std::size_t budget,
                      bool rectangular,bool sequential,int rows,void* user)
    : matrix_(enabled?matrix:nullptr),rectangular_(rectangular),rows_(rows) {
    if (!matrix_) return;
    auto callback=sequential?&entries<true>:&entries<false>;
    if constexpr (std::is_same<T,double>::value) {
      if (rectangular) d_c_bf_set_gpu_host_panels(matrix,callback,user,&comm,&budget,&rows);
      else d_c_bpack_set_gpu_host_panels(matrix,callback,user,&comm,&budget);
    } else if constexpr (std::is_same<T,std::complex<double>>::value) {
      if (rectangular) z_c_bf_set_gpu_host_panels(matrix,callback,user,&comm,&budget,&rows);
      else z_c_bpack_set_gpu_host_panels(matrix,callback,user,&comm,&budget);
    }
  }
  GPUHostPanelSession(const GPUHostPanelSession&)=delete;
  GPUHostPanelSession& operator=(const GPUHostPanelSession&)=delete;
  ~GPUHostPanelSession() { if (matrix_) release(nullptr); }
  template<typename Stats> void finish(Stats& stats,const MPIComm& comm,int cols) {
    if (!matrix_) return;
    bpack_gpu_host_panel_stats s{}; release(&s);
    stats.host_panel_calls+=s.calls; stats.host_panel_windows+=s.windows;
    stats.host_panel_upload_bytes+=s.upload_bytes;
    stats.host_panel_peak_bytes=std::max<std::size_t>(stats.host_panel_peak_bytes,s.peak_staging_bytes);
    stats.host_panel_prepare_seconds+=s.preparation_seconds;
    stats.host_panel_metadata_seconds+=s.metadata_seconds;
    stats.host_panel_upload_seconds+=s.upload_seconds;
    if (comm.is_root() || std::getenv("STRUMPACK_HODLR_STAGE_FILE")) std::cout << "# GPU host panels kind=" << (rectangular_?"BF":"HODLR")
      << " rows=" << rows_ << " cols=" << cols << " procs=" << comm.size() << " front_rank=" << comm.rank() << " calls_local=" << s.calls
      << " windows_local=" << s.windows << " upload_bytes_local=" << s.upload_bytes
      << " staging_peak_bytes_local=" << s.peak_staging_bytes
      << " host_prepare_seconds_local=" << s.preparation_seconds
      << " metadata_seconds_local=" << s.metadata_seconds
      << " upload_seconds_local=" << s.upload_seconds << std::endl;
  }
private:
  template<bool Sequential> static void entries
  (int* ni,int* nr,int* nc,std::int64_t* count,int* rows,int* cols,void* values,
   int* ms,int* ns,int* owners,int* np,int* maps,void* user) {
    HODLR_STAGE_SCOPE(prepare,"prepare.host_panels",-1,0,0,0,0,*count*sizeof(T));
    if constexpr (Sequential)
      HODLR_block_evaluation_seq<T>(ni,nr,nc,count,rows,cols,static_cast<T*>(values),ms,ns,owners,np,maps,user);
    else HODLR_block_evaluation<T>(ni,nr,nc,count,rows,cols,static_cast<T*>(values),ms,ns,owners,np,maps,user);
  }
  void release(bpack_gpu_host_panel_stats* stats) {
    if constexpr (std::is_same<T,double>::value) {
      if (rectangular_) d_c_bf_finish_gpu_host_panels(matrix_,stats);
      else d_c_bpack_finish_gpu_host_panels(matrix_,stats);
    } else if constexpr (std::is_same<T,std::complex<double>>::value) {
      if (rectangular_) z_c_bf_finish_gpu_host_panels(matrix_,stats);
      else z_c_bpack_finish_gpu_host_panels(matrix_,stats);
    }
    matrix_=nullptr;
  }
  void** matrix_; bool rectangular_; int rows_;
};
}}
#endif
#endif
