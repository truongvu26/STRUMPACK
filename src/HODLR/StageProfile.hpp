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
  std::ofstream file,native;
  std::string native_path;
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
    native_path=std::string(prefix)+".native.rank"+(rank?rank:"0")+".csv";
  }
};
inline Recorder& recorder() { static Recorder r; return r; }
class Scope;
inline std::vector<Scope*>& stack() { static thread_local std::vector<Scope*> s; return s; }
class Scope {
public:
  Scope(const char* label,long long front,int rows,int cols,int procs,int front_rank,std::size_t bytes=0,bool active=true)
    : front_(front),rows_(rows),cols_(cols),procs_(procs),rank_(front_rank),bytes_(bytes) {
    if (!active) return;
    auto& r=recorder(); enabled_=r.file.is_open();
    if (!enabled_) return;
    label_=label;
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
  long long front() const { return front_; }
  const std::string& section() const { return section_; }
private:
  std::string label_,section_; long long front_;
  int rows_,cols_,procs_,rank_; std::size_t bytes_;
  bool enabled_=false; double start_=0.,children_=0.;
};
// Native counters are a separate sidecar, never synthetic wall-time scopes:
// Entry_* accumulate, while Time_C_Extract and Flop_C_Extract reset per call.
// Reading these getters is local and introduces no MPI or CUDA synchronization.
template<typename Matrix> class NativeExtraction {
public:
  NativeExtraction(Matrix& matrix,const char* kind,long long front,int rows,int cols,
                   int procs,int rank,std::size_t queries,bool active=true)
    : matrix_(matrix),kind_(kind),front_(front),rows_(rows),cols_(cols),procs_(procs),rank_(rank),queries_(queries) {
    if (!active || !recorder().file.is_open()) return;
    enabled_=true;
    auto& s=stack();
    caller_front_=s.empty()?-1:s.back()->front();
    section_=s.empty()?"outside":s.back()->section();
    for (int i=0;i<3;++i) before_[i]=matrix_.get_stat(keys_[i]);
    start_=now();
  }
  ~NativeExtraction() {
    if (!enabled_) return;
    const auto end=now();
    double delta[3]; bool reset=false;
    for (int i=0;i<3;++i) {
      delta[i]=matrix_.get_stat(keys_[i])-before_[i];
      reset=reset || delta[i]<0.;
    }
    const auto native=matrix_.get_stat("Time_C_Extract");
    const auto flops=matrix_.get_stat("Flop_C_Extract");
    auto& r=recorder(); std::lock_guard<std::mutex> lock(r.mutex);
    if (!r.native.is_open()) {
      r.native.open(r.native_path);
      if (!r.native) std::abort();
      r.native << "kind,section,caller_front,front,rows,cols,procs,front_rank,queries,start,end,scope_wall_seconds,native_total_seconds,traverse_seconds,reconstruct_seconds,redistribute_seconds,flops,counter_reset\n";
      r.native << std::setprecision(17);
    }
    r.native << kind_ << ',' << section_ << ',' << caller_front_ << ',' << front_
      << ',' << rows_ << ',' << cols_ << ',' << procs_ << ',' << rank_ << ',' << queries_
      << ',' << start_ << ',' << end << ',' << end-start_ << ',' << native
      << ',' << delta[0] << ',' << delta[1] << ',' << delta[2] << ',' << flops << ',' << reset << '\n';
  }
private:
  Matrix& matrix_; const char* kind_; long long front_,caller_front_=-1;
  int rows_,cols_,procs_,rank_; std::size_t queries_;
  std::string section_; bool enabled_=false; double start_=0.,before_[3]{};
  static constexpr const char* keys_[3]={"Time_Entry_Traverse","Time_Entry_BF","Time_Entry_Comm"};
};
inline void image_delta(long long bytes) {
  auto& r=recorder(); if (!r.file.is_open()) return;
  std::lock_guard<std::mutex> lock(r.mutex);
  r.images+=bytes; r.peak_images=std::max(r.peak_images,r.images);
}
}}}
#define HODLR_STAGE_SCOPE(var,label,front,rows,cols,procs,rank,bytes) \
  strumpack::HODLR::profiling::Scope var(label,front,rows,cols,procs,rank,bytes)
#define HODLR_STAGE_SCOPE_IF(var,active,label,front,rows,cols,procs,rank,bytes) \
  strumpack::HODLR::profiling::Scope var(label,front,rows,cols,procs,rank,bytes,active)
#define HODLR_STAGE_NATIVE_EXTRACTION_IF(var,active,kind,matrix,front,rows,cols,procs,rank,queries) \
  strumpack::HODLR::profiling::NativeExtraction var(matrix,kind,front,rows,cols,procs,rank,queries,active)
#define HODLR_STAGE_IMAGE_DELTA(bytes) strumpack::HODLR::profiling::image_delta(bytes)
#else
#define HODLR_STAGE_SCOPE(...) (void)0
#define HODLR_STAGE_SCOPE_IF(...) (void)0
#define HODLR_STAGE_NATIVE_EXTRACTION_IF(...) (void)0
#define HODLR_STAGE_IMAGE_DELTA(...) (void)0
#endif
#endif
