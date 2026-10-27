// Diagnostic-only local phase scopes. No MPI, CUDA synchronization or changes
// to the numerical algorithm. All phases are inclusive unless marked exclusive.
#ifndef STRUMPACK_HODLR_STAGE_PROFILE_HPP
#define STRUMPACK_HODLR_STAGE_PROFILE_HPP
#if defined(STRUMPACK_HODLR_STAGE_PROFILE)
#include <nvtx3/nvToolsExt.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <string>
#include <vector>
namespace strumpack { namespace HODLR { namespace profiling {
inline double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
struct Recorder {
  std::ofstream file;
  std::mutex mutex;
  long long images=0,peak_images=0;
  Recorder() {
    const char* prefix=std::getenv("STRUMPACK_HODLR_STAGE_FILE");
    if (!prefix || !*prefix) return;
    const char* rank=std::getenv("SLURM_PROCID");
    file.open(std::string(prefix)+".rank"+(rank?rank:"0")+".csv");
    if (!file) std::abort();
    file << "phase,section,front,rows,cols,procs,front_rank,start,end,inclusive,exclusive,bytes,live_image_bytes,peak_image_bytes\n";
    file << std::setprecision(17);
  }
};
inline Recorder& recorder() { static Recorder r; return r; }
class Scope;
inline std::vector<Scope*>& stack() { static thread_local std::vector<Scope*> s; return s; }
class Scope {
public:
  Scope(const char* label,long long front,int rows,int cols,int procs,int front_rank,std::size_t bytes=0)
    : label_(label),front_(front),rows_(rows),cols_(cols),procs_(procs),rank_(front_rank),bytes_(bytes) {
    auto& r=recorder(); enabled_=r.file.is_open();
    if (!enabled_) return;
    auto& s=stack();
    if (front_<0 && !s.empty()) front_=s.back()->front_;
    if (!procs_ && !s.empty()) { procs_=s.back()->procs_; rank_=s.back()->rank_; }
    section_=label_.rfind("solver.",0)==0?label_:(s.empty()?"outside":s.back()->section_);
    start_=now(); s.push_back(this); nvtxRangePushA(label_.c_str());
  }
  ~Scope() {
    if (!enabled_) return;
    const auto end=now(),duration=end-start_;
    nvtxRangePop();
    auto& s=stack(); s.pop_back();
    if (!s.empty()) s.back()->children_+=duration;
    auto& r=recorder(); std::lock_guard<std::mutex> lock(r.mutex);
    r.file << label_ << ',' << section_ << ',' << front_ << ',' << rows_ << ',' << cols_ << ',' << procs_ << ',' << rank_
      << ',' << start_ << ',' << end << ',' << duration << ',' << std::max(0.,duration-children_)
      << ',' << bytes_ << ',' << r.images << ',' << r.peak_images << '\n';
  }
private:
  std::string label_,section_; long long front_;
  int rows_,cols_,procs_,rank_; std::size_t bytes_;
  bool enabled_=false; double start_=0.,children_=0.;
};
inline void image_delta(long long bytes) {
  auto& r=recorder(); if (!r.file.is_open()) return;
  std::lock_guard<std::mutex> lock(r.mutex);
  r.images+=bytes; r.peak_images=std::max(r.peak_images,r.images);
}
}}}
#define HODLR_STAGE_SCOPE(var,label,front,rows,cols,procs,rank,bytes) \
  strumpack::HODLR::profiling::Scope var(label,front,rows,cols,procs,rank,bytes)
#define HODLR_STAGE_IMAGE_DELTA(bytes) strumpack::HODLR::profiling::image_delta(bytes)
#else
#define HODLR_STAGE_SCOPE(...) (void)0
#define HODLR_STAGE_IMAGE_DELTA(...) (void)0
#endif
#endif
