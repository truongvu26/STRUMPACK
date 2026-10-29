/*
 * STRUMPACK -- STRUctured Matrices PACKage, Copyright (c) 2014, The
 * Regents of the University of California, through Lawrence Berkeley
 * National Laboratory (subject to receipt of any required approvals
 * from the U.S. Dept. of Energy).  All rights reserved.
 *
 * If you have questions about your rights to use or distribute this
 * software, please contact Berkeley Lab's Technology Transfer
 * Department at TTD@lbl.gov.
 *
 * NOTICE. This software is owned by the U.S. Department of Energy. As
 * such, the U.S. Government has been granted for itself and others
 * acting on its behalf a paid-up, nonexclusive, irrevocable,
 * worldwide license in the Software to reproduce, prepare derivative
 * works, and perform publicly and display publicly.  Beginning five
 * (5) years after the date permission to assert copyright is obtained
 * from the U.S. Department of Energy, and subject to any subsequent
 * five (5) year renewals, the U.S. Government is granted for itself
 * and others acting on its behalf a paid-up, nonexclusive,
 * irrevocable, worldwide license in the Software to reproduce,
 * prepare derivative works, distribute copies to the public, perform
 * publicly and display publicly, and to permit others to do so.
 *
 * Developers: Pieter Ghysels, Francois-Henry Rouet, Xiaoye S. Li.
 *             (Lawrence Berkeley National Lab, Computational Research
 *             Division).
 *
 */
#include <array>

#include "FrontGPU.hpp"
#include "FrontGPUKernels.hpp"
#include "HODLR/StageProfile.hpp"
#if defined(STRUMPACK_HAVE_HODLR_GPU_DENSE_SCHUR)
#include "HODLR/GPUDenseSchur.hpp"
#endif

#include <chrono>
#include <limits>
#include <stdexcept>
#include <type_traits>
#if defined(STRUMPACK_USE_CUDA)
#include <cuda_runtime_api.h>
#endif

#if defined(STRUMPACK_USE_MPI)
#include "ExtendAdd.hpp"
#include "FrontMPI.hpp"
#include "FrontBLRMPI.hpp"
#endif


namespace strumpack {

  namespace {
    using ExactFrontClock = std::chrono::steady_clock;
    double exact_front_seconds(ExactFrontClock::time_point started) {
      return std::chrono::duration<double>(ExactFrontClock::now()-started).count();
    }
    std::size_t exact_front_product(std::size_t a, std::size_t b) {
      if (a && b>std::numeric_limits<std::size_t>::max()/a)
        throw std::overflow_error("GPU exact front size exceeds addressable storage");
      return a*b;
    }
    std::size_t exact_front_sum(std::size_t a, std::size_t b) {
      if (b>std::numeric_limits<std::size_t>::max()-a)
        throw std::overflow_error("GPU exact front size exceeds addressable storage");
      return a+b;
    }
    // Keep space for CUDA/cuBLAS/cuSOLVER and native HODLR allocations. This
    // is a local admission bound, not a reservation against another process.
    std::size_t exact_front_device_budget(std::size_t free) {
      const auto reserve=std::max(std::size_t(256)*1024*1024,free/8);
      return free>reserve ? free-reserve : 0;
    }
    void exact_front_allocation_error() {
#if defined(STRUMPACK_USE_CUDA)
      // The generic allocator maps CUDA errors to bad_alloc. Only expected
      // memory pressure permits CPU fallback; an execution error must escape.
      const auto status=cudaGetLastError();
      if (status!=cudaSuccess && status!=cudaErrorMemoryAllocation)
        throw std::runtime_error(std::string("GPU exact front allocation: ")+
                                 cudaGetErrorString(status));
#endif
    }
    struct ExactFrontStats {
      std::size_t batch_subtrees=0, batch_fronts=0, split_gpu_nodes=0;
      std::size_t cpu_fallback_nodes=0, memory_splits=0, mixed_splits=0;
      std::size_t allocation_fallbacks=0, upload_bytes=0, download_bytes=0;
      std::size_t fallback_schur_gpu_nodes=0,fallback_schur_cpu_nodes=0;
      std::size_t fallback_schur_upload_bytes=0,fallback_schur_download_bytes=0;
      std::size_t fallback_schur_device_peak_bytes=0;
      std::size_t device_peak_bytes=0;
      double batch_seconds=0.,split_gpu_seconds=0.,cpu_fallback_seconds=0.;
      double fallback_schur_seconds=0.,fallback_schur_device_seconds=0.;
    };
    thread_local ExactFrontStats* exact_front_stats=nullptr;
    // Recursive splits accumulate into one summary for the outer GPU subtree.
    class ExactFrontScope {
    public:
      ExactFrontScope(bool enabled,long long front,std::size_t rows,std::size_t upd)
        : front_(front),rows_(rows),upd_(upd),started_(ExactFrontClock::now()) {
        if (enabled && !exact_front_stats) { owner_=true;exact_front_stats=&stats_; }
      }
      ~ExactFrontScope() {
        if (!owner_) return;
        exact_front_stats=nullptr;
        std::cout << "# HODLR exact dense subtree front=" << front_
                  << " separator_rows=" << rows_ << " update_rows=" << upd_
                  << " batch_subtrees=" << stats_.batch_subtrees
                  << " batch_fronts=" << stats_.batch_fronts
                  << " split_gpu_nodes=" << stats_.split_gpu_nodes
                  << " cpu_fallback_nodes=" << stats_.cpu_fallback_nodes
                  << " memory_splits=" << stats_.memory_splits
                  << " mixed_splits=" << stats_.mixed_splits
                  << " allocation_fallbacks=" << stats_.allocation_fallbacks
                  << " fallback_schur_gpu_nodes=" << stats_.fallback_schur_gpu_nodes
                  << " fallback_schur_cpu_nodes=" << stats_.fallback_schur_cpu_nodes
                  << " wall_seconds_local=" << exact_front_seconds(started_)
                  << " batch_seconds_local=" << stats_.batch_seconds
                  << " split_gpu_seconds_local=" << stats_.split_gpu_seconds
                  << " cpu_fallback_seconds_local=" << stats_.cpu_fallback_seconds
                  << " upload_bytes_local=" << stats_.upload_bytes
                  << " download_bytes_local=" << stats_.download_bytes
                  << " explicit_device_peak_bytes_local=" << stats_.device_peak_bytes
                  << " fallback_schur_seconds_local=" << stats_.fallback_schur_seconds
                  << " fallback_schur_device_seconds_local=" << stats_.fallback_schur_device_seconds
                  << " fallback_schur_upload_bytes_local=" << stats_.fallback_schur_upload_bytes
                  << " fallback_schur_download_bytes_local=" << stats_.fallback_schur_download_bytes
                  << " fallback_schur_device_peak_bytes_local=" << stats_.fallback_schur_device_peak_bytes
                  << std::endl;
      }
    private:
      bool owner_=false;
      long long front_;
      std::size_t rows_,upd_;
      ExactFrontClock::time_point started_;
      ExactFrontStats stats_;
    };
  }

  template<typename scalar_t, typename integer_t> class LevelInfo {
    using F_t = Front<scalar_t,integer_t>;
    using FG_t = FrontGPU<scalar_t,integer_t>;
    using DenseMW_t = DenseMatrixWrapper<scalar_t>;
    using SpMat_t = CompressedSparseMatrix<scalar_t,integer_t>;

  public:
    LevelInfo() {}

    LevelInfo(const std::vector<F_t*>& fronts, gpu::Handle& handle,
              int max_streams, const SpMat_t* A=nullptr) {
      f.reserve(fronts.size());
      for (auto& F : fronts)
        f.push_back(dynamic_cast<FG_t*>(F));
      // This pragma causes "internal error: null pointer" on the
      // intel compiler.  It seems to be because the variables are
      // class members.
      // #pragma omp parallel for reduction(+:L_size,U_size,Schur_size,piv_size,total_upd_size,N8,N16,N24,N32,factors_small) reduction(max:max_dsep)
      for (std::size_t i=0; i<f.size(); i++) {
        auto F = f[i];
        const std::size_t dsep = F->dim_sep();
        const std::size_t dupd = F->dim_upd();
        L_size += dsep*(dsep + dupd);
        U_size += dsep*dupd;
        Schur_size += dupd*dupd;
        piv_size += dsep;
        total_upd_size += dupd;
        if (dsep <= 32) {
          factors_small += dsep*(dsep + 2*dupd);
          if (dsep <= 8)       N8++;
          else if (dsep <= 16) N16++;
          else if (dsep <= 24) N24++;
          else N32++;
        } else
          getrf_work_size = std::max
            (getrf_work_size, std::max
             (gpu::getrf_buffersize<scalar_t>(handle, dsep),
              gpu::getrs_buffersize<scalar_t>
              (handle, Trans::N, dsep, dupd, dsep, dsep)));
      }
      small_fronts = N8 + N16 + N24 + N32;
      if (small_fronts && small_fronts != f.size())
        std::partition
          (f.begin(), f.end(), [](const FG_t* const& a) -> bool {
            return a->dim_sep() <= 32; });
      if (A) {
        auto N = f.size();
        elems11.resize(N+1);
        elems12.resize(N+1);
        elems21.resize(N+1);
        Isize.resize(N+1);
#pragma omp parallel for
        for (std::size_t i=0; i<N; i++) {
          auto& F = *(f[i]);
          A->count_front_elements
            (F.sep_begin(), F.sep_end(), F.upd(),
             elems11[i+1], elems12[i+1], elems21[i+1]);
          if (F.lchild_) Isize[i+1] += F.lchild_->dim_upd();
          if (F.rchild_) Isize[i+1] += F.rchild_->dim_upd();
        }
        for (std::size_t i=0; i<N; i++) {
          elems11[i+1] += elems11[i];
          elems12[i+1] += elems12[i];
          elems21[i+1] += elems21[i];
          Isize[i+1] += Isize[i];
        }
      }
      factor_size = L_size + U_size;

      factor_bytes = sizeof(scalar_t) * factor_size;
      factor_bytes = gpu::round_up(factor_bytes);

      work_bytes = sizeof(scalar_t) * (Schur_size + getrf_work_size * max_streams);
      work_bytes = gpu::round_up(work_bytes);
      work_bytes += sizeof(int) * (piv_size + f.size());
      work_bytes = gpu::round_up(work_bytes);
      work_bytes += sizeof(gpu::FrontData<scalar_t>) * (N8 + N16 + N24 + N32);
      work_bytes = gpu::round_up(work_bytes);

      ea_bytes = sizeof(gpu::AssembleData<scalar_t>) * f.size();
      ea_bytes = gpu::round_up(ea_bytes);
      ea_bytes += sizeof(std::size_t) * Isize.back();
      ea_bytes = gpu::round_up(ea_bytes);
      ea_bytes += sizeof(Triplet<scalar_t>) * (elems11.back() + elems12.back() + elems21.back());
      ea_bytes = gpu::round_up(ea_bytes);
    }

    void print_info(int l, int lvls) {
      std::cout << "#  level " << l << " of " << lvls
                << " has " << f.size() << " nodes and "
                << N8 << " <=8, " << N16 << " <=16, "
                << N24 << " <=24, " << N32 << " <=32, needs "
                << factor_bytes / 1.e6
                << " MB for factors, "
                << Schur_size * sizeof(scalar_t) / 1.e6
                << " MB for Schur complements" << std::endl;
    }

    void flops(long long& level_flops, long long& small_flops) {
      level_flops = small_flops = 0;
      auto N = f.size();
#pragma omp parallel for reduction(+: level_flops, small_flops)
      for (std::size_t i=0; i<N; i++) {
        auto F = f[i];
        auto flops = LU_flops(F->F11_) +
          gemm_flops(Trans::N, Trans::N, scalar_t(-1.),
                     F->F21_, F->F12_, scalar_t(1.)) +
          trsm_flops(Side::L, scalar_t(1.), F->F11_, F->F12_) +
          trsm_flops(Side::R, scalar_t(1.), F->F11_, F->F21_);
        level_flops += flops;
        if (F->dim_sep() <= 32)
          small_flops += flops;
      }
    }

    /*
     * first store L factors, then U factors,
     *  F11, F21, F11, F21, ..., F12, F12, ...
     */
    void set_factor_pointers(scalar_t* factors) {
      for (auto F : f) {
        const std::size_t dsep = F->dim_sep();
        const std::size_t dupd = F->dim_upd();
        F->F11_ = DenseMW_t(dsep, dsep, factors, dsep); factors += dsep*dsep;
        F->F12_ = DenseMW_t(dsep, dupd, factors, dsep); factors += dsep*dupd;
        F->F21_ = DenseMW_t(dupd, dsep, factors, dupd); factors += dupd*dsep;
      }
    }

    void set_pivot_pointers(int* pmem) {
      for (auto F : f) {
        F->piv_ = pmem;
        pmem += F->dim_sep();
      }
    }

    void set_work_pointers(void* wmem, int max_streams) {
      auto schur = gpu::aligned_ptr<scalar_t>(wmem);
      for (auto F : f) {
        const int dupd = F->dim_upd();
        if (dupd) {
          F->F22_ = DenseMW_t(dupd, dupd, schur, dupd);
          schur += dupd*dupd;
        }
      }
      dev_getrf_work = schur;
      schur += max_streams * getrf_work_size;
      auto imem = gpu::aligned_ptr<int>(schur);
      for (auto F : f) {
        F->piv_ = imem;
        imem += F->dim_sep();
      }
      dev_getrf_err = imem;   imem += f.size();
      auto fdat = gpu::aligned_ptr<gpu::FrontData<scalar_t>>(imem);
      f8  = fdat;  fdat += N8;
      f16 = fdat;  fdat += N16;
      f24 = fdat;  fdat += N24;
      f32 = fdat;  fdat += N32;
    }

    int align = 0;
    std::vector<FG_t*> f;
    std::size_t L_size = 0, U_size = 0, factor_size = 0,
      factors_small = 0, Schur_size = 0, piv_size = 0,
      total_upd_size = 0;
    std::size_t N8 = 0, N16 = 0, N24 = 0, N32 = 0, small_fronts = 0;
    std::size_t work_bytes = 0, ea_bytes = 0, factor_bytes = 0;
    std::vector<std::size_t> elems11, elems12, elems21, Isize;
    scalar_t* dev_getrf_work = nullptr;
    int* dev_getrf_err = nullptr;
    std::int64_t getrf_work_size = 0;
    gpu::FrontData<scalar_t> *f8 = nullptr, *f16 = nullptr,
      *f24 = nullptr, *f32 = nullptr;
  };


  template<typename scalar_t,typename integer_t>
  FrontGPU<scalar_t,integer_t>::FrontGPU
  (integer_t sep, integer_t sep_begin, integer_t sep_end,
   std::vector<integer_t>& upd)
    : F_t(nullptr, nullptr, sep, sep_begin, sep_end, upd) {}

  template<typename scalar_t,typename integer_t>
  FrontGPU<scalar_t,integer_t>::~FrontGPU() {
#if defined(STRUMPACK_COUNT_FLOPS)
    const std::size_t dupd = dim_upd();
    const std::size_t dsep = dim_sep();
    STRUMPACK_SUB_MEMORY(dsep*(dsep+2*dupd)*sizeof(scalar_t));
#endif
  }

  template<typename scalar_t,typename integer_t> scalar_t*
  FrontGPU<scalar_t,integer_t>::get_device_F22(scalar_t* dF22) {
    gpu::copy_host_to_device(dF22, F22_);
    return dF22;
  }

  template<typename scalar_t,typename integer_t> void
  FrontGPU<scalar_t,integer_t>::release_work_memory() {
    F22_.clear();
    host_Schur_.reset(nullptr);
  }

  template<typename scalar_t,typename integer_t> void
  FrontGPU<scalar_t,integer_t>::extract_CB_sub_matrix
  (const std::vector<std::size_t>& I, const std::vector<std::size_t>& J,
   DenseM_t& B, int task_depth) const {
    // Factorization retains this subtree root's Schur contribution on host.
    // Add only the contribution: the parent extracts original sparse entries.
    std::vector<std::size_t> lJ,oJ;
    this->find_upd_indices(J,lJ,oJ);
    if (lJ.empty()) return;
    std::vector<std::size_t> lI,oI;
    this->find_upd_indices(I,lI,oI);
    if (lI.empty()) return;
    assert(F22_.rows()==std::size_t(dim_upd()) &&
           F22_.cols()==std::size_t(dim_upd()));
    for (std::size_t j=0;j<lJ.size();++j)
      for (std::size_t i=0;i<lI.size();++i)
        B(oI[i],oJ[j])+=F22_(lI[i],lJ[j]);
    STRUMPACK_FLOPS((is_complex<scalar_t>()?2:1)*lI.size()*lJ.size());
  }

#if defined(STRUMPACK_USE_MPI)
  template<typename scalar_t,typename integer_t> void
  FrontGPU<scalar_t,integer_t>::extend_add_copy_to_buffers
  (std::vector<std::vector<scalar_t>>& sbuf,
   const FrontMPI<scalar_t,integer_t>* pa) const {
    ExtendAdd<scalar_t,integer_t>::extend_add_seq_copy_to_buffers
      (F22_, sbuf, pa, this);
  }

  template<typename scalar_t,typename integer_t> void
  FrontGPU<scalar_t,integer_t>::extadd_blr_copy_to_buffers
  (std::vector<std::vector<scalar_t>>& sbuf,
   const FrontBLRMPI<scalar_t,integer_t>* pa) const {
    BLR::BLRExtendAdd<scalar_t,integer_t>::
      seq_copy_to_buffers(F22_, sbuf, pa, this);
  }

  template<typename scalar_t,typename integer_t> void
  FrontGPU<scalar_t,integer_t>::extadd_blr_copy_to_buffers_col
  (std::vector<std::vector<scalar_t>>& sbuf,
   const FrontBLRMPI<scalar_t,integer_t>* pa,
   integer_t begin_col, integer_t end_col, const Opts_t& opts) const {
    BLR::BLRExtendAdd<scalar_t,integer_t>::
      seq_copy_to_buffers_col(F22_, sbuf, pa, this, begin_col, end_col);
  }
#endif

  template<typename scalar_t,typename integer_t> void
  FrontGPU<scalar_t,integer_t>::extend_add_to_dense
  (DenseM_t& paF11, DenseM_t& paF12, DenseM_t& paF21, DenseM_t& paF22,
   const F_t* p, int task_depth) {
    this->extend_add(paF11, paF12, paF21, paF22, F22_, p);
    release_work_memory();
  }

  template<typename scalar_t,typename integer_t>
  std::size_t peak_device_memory
  (const std::vector<LevelInfo<scalar_t,integer_t>>& ldata) {
    std::size_t peak_dmem = 0;
    for (std::size_t l=0; l<ldata.size(); l++) {
      auto& L = ldata[l];
      // memory needed on this level: factors,
      // schur updates, pivot vectors, cuSOLVER work space,
      // assembly data (indices, sparse elements)
      std::size_t level_mem = L.factor_bytes + L.work_bytes + L.ea_bytes;
      // the contribution blocks of the previous level are still
      // needed for the extend-add
      if (l+1 < ldata.size())
        level_mem += ldata[l+1].work_bytes;
      peak_dmem = std::max(peak_dmem, level_mem);
    }
    return peak_dmem;
  }

  template<typename scalar_t, typename integer_t> void
  FrontGPU<scalar_t,integer_t>::front_assembly
  (const SpMat_t& A, LInfo_t& L, char* hea_mem, char* dea_mem) {
    using Trip_t = Triplet<scalar_t>;
    auto N = L.f.size();
    auto hasmbl = gpu::aligned_ptr<gpu::AssembleData<scalar_t>>(hea_mem);
    auto Iptr   = gpu::aligned_ptr<std::size_t>(hasmbl + N);
    auto e11    = gpu::aligned_ptr<Trip_t>(Iptr + L.Isize.back());
    auto e12    = e11 + L.elems11.back();
    auto e21    = e12 + L.elems12.back();
    auto dasmbl = gpu::aligned_ptr<gpu::AssembleData<scalar_t>>(dea_mem);
    auto dIptr  = gpu::aligned_ptr<std::size_t>(dasmbl + N);
    auto de11   = gpu::aligned_ptr<Trip_t>(dIptr + L.Isize.back());
    auto de12   = de11 + L.elems11.back();
    auto de21   = de12 + L.elems12.back();

#pragma omp parallel for
    for (std::size_t n=0; n<N; n++) {
      auto& f = *(L.f[n]);
      A.set_front_elements
        (f.sep_begin_, f.sep_end_, f.upd_,
         e11+L.elems11[n], e12+L.elems12[n], e21+L.elems21[n]);
      hasmbl[n] = gpu::AssembleData<scalar_t>
        (f.dim_sep(), f.dim_upd(), f.F11_.data(), f.F12_.data(),
         f.F21_.data(), f.F22_.data(),
         L.elems11[n+1]-L.elems11[n], L.elems12[n+1]-L.elems12[n],
         L.elems21[n+1]-L.elems21[n],
         de11+L.elems11[n], de12+L.elems12[n], de21+L.elems21[n]);
      auto fIptr = Iptr + L.Isize[n];
      auto fdIptr = dIptr + L.Isize[n];
      if (f.lchild_) {
        auto c = dynamic_cast<FG_t*>(f.lchild_.get());
        hasmbl[n].set_ext_add_left(c->dim_upd(), c->F22_.data(), fdIptr);
        c->upd_to_parent(&f, fIptr);
        fIptr += c->dim_upd();
        fdIptr += c->dim_upd();
      }
      if (f.rchild_) {
        auto c = dynamic_cast<FG_t*>(f.rchild_.get());
        hasmbl[n].set_ext_add_right(c->dim_upd(), c->F22_.data(), fdIptr);
        c->upd_to_parent(&f, fIptr);
      }
    }
    gpu::copy_host_to_device<char>(dea_mem, hea_mem, L.ea_bytes);
    gpu::assemble(N, hasmbl, dasmbl);
  }

  template<typename scalar_t, typename integer_t> void
  FrontGPU<scalar_t,integer_t>::factor_small_fronts
  (LInfo_t& L, gpu::FrontData<scalar_t>* fdata, int* dinfo,
   const SPOptions<scalar_t>& opts) {
    if (!L.small_fronts) return;
    for (std::size_t n=0, n8=0, n16=L.N8, n24=n16+L.N16, n32=n24+L.N24;
         n<L.small_fronts; n++) {
      auto& f = *(L.f[n]);
      const auto dsep = f.dim_sep();
      gpu::FrontData<scalar_t>
        t(dsep, f.dim_upd(), f.F11_.data(), f.F12_.data(),
          f.F21_.data(), f.F22_.data(), f.piv_);
      if (dsep <= 8)       fdata[n8++] = t;
      else if (dsep <= 16) fdata[n16++] = t;
      else if (dsep <= 24) fdata[n24++] = t;
      else                 fdata[n32++] = t;
    }
    gpu::copy_host_to_device(L.f8, fdata, L.small_fronts);
    auto replace = opts.replace_tiny_pivots();
    auto thresh = opts.pivot_threshold();
    gpu::factor_block_batch<scalar_t,8>(L.N8, L.f8, replace, thresh, dinfo);
    gpu::factor_block_batch<scalar_t,16>(L.N16, L.f16, replace, thresh, dinfo+L.N8);
    gpu::factor_block_batch<scalar_t,24>(L.N24, L.f24, replace, thresh, dinfo+L.N8+L.N16);
    gpu::factor_block_batch<scalar_t,32>(L.N32, L.f32, replace, thresh, dinfo+L.N8+L.N16+L.N24);
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  FrontGPU<scalar_t,integer_t>::factor_hodlr_node
  (const Opts_t& opts,int task_depth) {
    const std::size_t n=dim_sep(),u=dim_upd();
    if (!n) return ReturnCode::SUCCESS;
    const auto nn=exact_front_product(n,n),nu=exact_front_product(n,u);
    const auto uu=exact_front_product(u,u);
    const auto base=exact_front_sum(nn,exact_front_sum(exact_front_product(2,nu),uu));
    const auto pivbytes=exact_front_product(n+1,sizeof(int));
    auto required=exact_front_sum(exact_front_product(base,sizeof(scalar_t)),pivbytes);
    std::unique_ptr<gpu::Handle> handle;
    gpu::DeviceMemory<scalar_t> memory;
    gpu::DeviceMemory<int> pivots;
    std::int64_t worksize=0;
    bool admitted=false;
    // Do not even initialize a new solver handle under severe pressure. All
    // allocations in the try block precede GPU work and host factor mutation.
    if (required<=exact_front_device_budget(gpu::available_memory())) {
      try {
        handle=std::make_unique<gpu::Handle>();
        worksize=std::max(gpu::getrf_buffersize<scalar_t>(*handle,n),
                         gpu::getrs_buffersize<scalar_t>(*handle,Trans::N,n,u,n,n));
        if (worksize<0)
          throw std::runtime_error("Negative GPU exact front workspace size");
        const auto elements=exact_front_sum(base,std::size_t(worksize));
        required=exact_front_sum(exact_front_product(elements,sizeof(scalar_t)),pivbytes);
        if (required<=exact_front_device_budget(gpu::available_memory())) {
          memory=gpu::DeviceMemory<scalar_t>(elements);
          pivots=gpu::DeviceMemory<int>(n+1);
          admitted=true;
        }
      } catch (const std::bad_alloc&) {
        // No factor/Schur copies or kernels have run. CPU LU is still safe.
        memory.release();pivots.release();handle.reset();
        exact_front_allocation_error();
        if (exact_front_stats) ++exact_front_stats->allocation_fallbacks;
      }
    }
    const auto started=ExactFrontClock::now();
    if (!admitted) {
      handle.reset();
      HODLR_STAGE_SCOPE(p_cpu,"front.gpu_exact.cpu_fallback",this->sep_begin_,n,u,1,0,0);
      auto code=F11_.LU(pivot_mem_,task_depth) ?
        ReturnCode::ZERO_PIVOT : ReturnCode::SUCCESS;
      piv_=pivot_mem_.data();
      if (opts.replace_tiny_pivots()) {
        const auto threshold=opts.pivot_threshold();
        for (std::size_t i=0;i<n;++i)
          if (std::abs(F11_(i,i))<threshold)
            F11_(i,i)=std::real(F11_(i,i))<0 ? -threshold : threshold;
      }
      if (u) {
        // FrontGPU solves expect F12=A11^{-1}*A12 and an unmodified F21.
        F11_.solve_LU_in_place(F12_,piv_,task_depth);
        bool completed=false;
#if defined(STRUMPACK_HAVE_HODLR_GPU_DENSE_SCHUR)
        if constexpr (std::is_same<scalar_t,double>::value) {
          if (opts.HODLR_options().gpu_dense_schur() && n>=1000 &&
              2.L*u*u*n>=1.e11L) {
            HODLR::DenseSchurGPUStats stats;
            HODLR_STAGE_SCOPE(p_schur,"front.gpu_exact.fallback_schur_gpu_attempt",this->sep_begin_,n,u,1,0,0);
            completed=HODLR::gpu_dense_schur_update
              (u,u,n,F21_.data(),F21_.ld(),F12_.data(),F12_.ld(),F22_.data(),F22_.ld(),stats);
            if (exact_front_stats) {
              exact_front_stats->fallback_schur_seconds+=stats.wall_seconds;
              exact_front_stats->fallback_schur_device_seconds+=stats.device_gemm_seconds;
              if (stats.used_gpu) {
                ++exact_front_stats->fallback_schur_gpu_nodes;
                exact_front_stats->fallback_schur_upload_bytes+=stats.upload_bytes;
                exact_front_stats->fallback_schur_download_bytes+=stats.download_bytes;
                exact_front_stats->fallback_schur_device_peak_bytes=std::max
                  (exact_front_stats->fallback_schur_device_peak_bytes,stats.device_peak_bytes);
                exact_front_stats->upload_bytes+=stats.upload_bytes;
                exact_front_stats->download_bytes+=stats.download_bytes;
                exact_front_stats->device_peak_bytes=std::max(exact_front_stats->device_peak_bytes,stats.device_peak_bytes);
              }
            }
            if (completed) {
              STRUMPACK_FLOPS(gemm_flops(Trans::N,Trans::N,scalar_t(-1.),F21_,F12_,scalar_t(1.)));
              STRUMPACK_BYTES(8*blas::gemm_moves(u,u,n));
            }
          }
        }
#endif
        if (!completed) {
          gemm(Trans::N,Trans::N,scalar_t(-1.),F21_,F12_,scalar_t(1.),F22_,task_depth);
          if (exact_front_stats) ++exact_front_stats->fallback_schur_cpu_nodes;
        }
      }
      if (exact_front_stats) {
        ++exact_front_stats->cpu_fallback_nodes;
        exact_front_stats->cpu_fallback_seconds+=exact_front_seconds(started);
      }
      return code;
    }

    HODLR_STAGE_SCOPE(p_gpu,"front.gpu_exact.split_gpu",this->sep_begin_,n,u,1,0,required);
    // Every explicit device buffer is now allocated. Execution errors must
    // propagate; retrying CPU LU after these operations would corrupt factors.
    auto work=memory+nn;
    auto data12=work+worksize;
    auto data21=data12+nu;
    auto data22=data21+nu;
    DenseMW_t dF11(n,n,memory,n),dF12(n,u,data12,n);
    DenseMW_t dF21(u,n,data21,u),dF22(u,u,data22,u);
    gpu::copy_host_to_device(dF11,F11_);
    gpu::getrf(*handle,dF11,work,worksize,pivots,pivots+n);
    if (opts.replace_tiny_pivots())
      gpu::replace_pivots(n,dF11.data(),opts.pivot_threshold());
    int info=0;
    gpu::copy_device_to_host(&info,pivots+n,1);
    auto code=info ? ReturnCode::ZERO_PIVOT : ReturnCode::SUCCESS;
    if (u) {
      gpu::copy_host_to_device(dF12,F12_);
      gpu::getrs(*handle,Trans::N,dF11,pivots,dF12,pivots+n,work,worksize);
      gpu::copy_device_to_host(&info,pivots+n,1);
      if (info) code=ReturnCode::ZERO_PIVOT;
      gpu::copy_host_to_device(dF21,F21_);
      gpu::copy_host_to_device(dF22,F22_);
      gpu::gemm(*handle,Trans::N,Trans::N,scalar_t(-1.),dF21,dF12,scalar_t(1.),dF22);
      gpu::copy_device_to_host(F12_,dF12);
      gpu::copy_device_to_host(F22_,dF22);
    }
    pivot_mem_.resize(n);piv_=pivot_mem_.data();
    gpu::copy_device_to_host(piv_,pivots.template as<int>(),n);
    gpu::copy_device_to_host(F11_,dF11);
    if (exact_front_stats) {
      ++exact_front_stats->split_gpu_nodes;
      exact_front_stats->split_gpu_seconds+=exact_front_seconds(started);
      exact_front_stats->device_peak_bytes=std::max(exact_front_stats->device_peak_bytes,required);
      exact_front_stats->upload_bytes+=exact_front_product(base,sizeof(scalar_t));
      exact_front_stats->download_bytes+=exact_front_product(nn+nu+uu,sizeof(scalar_t))+
        (n+1+(u?1:0))*sizeof(int);
    }
    return code;
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  FrontGPU<scalar_t,integer_t>::split_smaller
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   int etree_level, int task_depth) {
    const std::size_t dupd = dim_upd(), dsep = dim_sep();
    ReturnCode err_code = ReturnCode::SUCCESS;
    if (lchild_) {
      auto el = lchild_->multifrontal_factorization
        (A, opts, etree_level+1, task_depth);
      if (el != ReturnCode::SUCCESS) err_code = el;
    }
    if (rchild_) {
      auto er = rchild_->multifrontal_factorization
        (A, opts, etree_level+1, task_depth);
      if (er != ReturnCode::SUCCESS) err_code = er;
    }
    if (opts.HODLR_options().gpu_dense_fronts()) {
      const auto limit=std::size_t(std::numeric_limits<int>::max());
      if (dsep>limit || dupd>limit)
        throw std::overflow_error("GPU exact front dimensions exceed the BLAS integer range");
      exact_front_product(exact_front_product(dsep,exact_front_sum(dsep,2*dupd)),sizeof(scalar_t));
      exact_front_product(exact_front_product(dupd,dupd),sizeof(scalar_t));
    }
    STRUMPACK_ADD_MEMORY(dsep*(dsep+2*dupd)*sizeof(scalar_t));
    STRUMPACK_ADD_MEMORY(dupd*dupd*sizeof(scalar_t));
    host_factors_.reset(new scalar_t[dsep*(dsep+2*dupd)]);
    host_Schur_.reset(new scalar_t[dupd*dupd]);
    {
      auto fmem = host_factors_.get();
      F11_ = DenseMW_t(dsep, dsep, fmem, dsep); fmem += dsep*dsep;
      F12_ = DenseMW_t(dsep, dupd, fmem, dsep); fmem += dsep*dupd;
      F21_ = DenseMW_t(dupd, dsep, fmem, dupd);
    }
    F22_ = DenseMW_t(dupd, dupd, host_Schur_.get(), dupd);
    F11_.zero(); F12_.zero();
    F21_.zero(); F22_.zero();
    A.extract_front
      (F11_, F12_, F21_, this->sep_begin_, this->sep_end_,
       this->upd_, task_depth);
    if (lchild_) {
#pragma omp parallel
#pragma omp single
      lchild_->extend_add_to_dense(F11_, F12_, F21_, F22_, this, 0);
    }
    if (rchild_) {
#pragma omp parallel
#pragma omp single
      rchild_->extend_add_to_dense(F11_, F12_, F21_, F22_, this, 0);
    }
    // TaskTimer tl("");
    // tl.start();
    if (opts.HODLR_options().gpu_dense_fronts()) {
      const auto local=factor_hodlr_node(opts,task_depth);
      if (local!=ReturnCode::SUCCESS) err_code=local;
    } else if (dsep) {
      gpu::Handle h;
      auto lwork = gpu::getrf_buffersize<scalar_t>(h, dsep);
      gpu::DeviceMemory<scalar_t> dm11(dsep*dsep + lwork);
      gpu::DeviceMemory<int> dpiv(dsep+1); // and ierr
      DenseMW_t dF11(dsep, dsep, dm11, dsep);
      gpu::copy_host_to_device(dF11, F11_);
      gpu::getrf(h, dF11, dm11 + dsep*dsep, lwork, dpiv, dpiv+dsep);
      if (opts.replace_tiny_pivots())
        gpu::replace_pivots
          (F11_.rows(), dF11.data(), opts.pivot_threshold());
      int info;
      gpu::copy_device_to_host(&info, dpiv+dsep, 1);
      if (info) err_code = ReturnCode::ZERO_PIVOT;
      pivot_mem_.resize(dsep);
      piv_ = pivot_mem_.data();
      gpu::copy_device_to_host(piv_, dpiv.as<int>(), dsep);
      gpu::copy_device_to_host(F11_, dF11);
      if (dupd) {
        gpu::DeviceMemory<scalar_t> dm12(dsep*dupd);
        DenseMW_t dF12(dsep, dupd, dm12, dsep);
        gpu::copy_host_to_device(dF12, F12_);
        gpu::getrs(h, Trans::N, dF11, dpiv, dF12, dpiv+dsep);
        gpu::copy_device_to_host(F12_, dF12);
        dm11.release();
        gpu::DeviceMemory<scalar_t> dm2122((dsep+dupd)*dupd);
        DenseMW_t dF21(dupd, dsep, dm2122, dupd);
        DenseMW_t dF22(dupd, dupd, dm2122+(dsep*dupd), dupd);
        gpu::copy_host_to_device(dF21, F21_);
        gpu::copy_host_to_device(dF22, host_Schur_.get());
        gpu::gemm(h, Trans::N, Trans::N, scalar_t(-1.),
                  dF21, dF12, scalar_t(1.), dF22);
        gpu::copy_device_to_host(host_Schur_.get(), dF22);
      }
    }
    // count flops
    auto level_flops = LU_flops(F11_) +
      gemm_flops(Trans::N, Trans::N, scalar_t(-1.), F21_, F12_, scalar_t(1.)) +
      trsm_flops(Side::L, scalar_t(1.), F11_, F12_) +
      trsm_flops(Side::R, scalar_t(1.), F11_, F21_);
    (void)level_flops;
    STRUMPACK_FULL_RANK_FLOPS(level_flops);
    // if (opts.verbose()) {
    //   auto level_time = tl.elapsed();
    //   std::cout << "#   GPU Factorization complete, took: "
    //             << level_time << " seconds, "
    //             << level_flops / 1.e9 << " GFLOPS, "
    //             << (float(level_flops) / level_time) / 1.e9
    //             << " GFLOP/s" << std::endl;
    // }
    return err_code;
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  FrontGPU<scalar_t,integer_t>::multifrontal_factorization
  (const SpMat_t& A, const SPOptions<scalar_t>& opts,
   int etree_level, int task_depth) {
    const bool hodlr_exact=opts.HODLR_options().gpu_dense_fronts();
    ExactFrontScope exact_scope(hodlr_exact,this->sep_begin_,dim_sep(),dim_upd());
    HODLR_STAGE_SCOPE_IF(p_exact,hodlr_exact,"front.gpu_exact.subtree",this->sep_begin_,dim_sep(),dim_upd(),1,0,0);
    const auto batch_started=ExactFrontClock::now();
    // A low-memory leaf must reach CPU admission without initializing another
    // cuSOLVER handle. Split traversal will make the same check per node.
    if (hodlr_exact && !exact_front_device_budget(gpu::available_memory())) {
      if (exact_front_stats) ++exact_front_stats->memory_splits;
      return split_smaller(A,opts,etree_level,task_depth);
    }
    ReturnCode err_code = ReturnCode::SUCCESS;
    const int max_streams = opts.gpu_streams();
    std::vector<gpu::Handle> handles(max_streams);
    const int lvls = this->levels();
    std::vector<LInfo_t> ldata(lvls);
    {
      std::vector<std::vector<F_t*>> fp(lvls);
      try {
        this->get_level_fronts_gpu(fp);
      } catch (...) {
        if (exact_front_stats) ++exact_front_stats->mixed_splits;
        return split_smaller(A, opts, etree_level, task_depth);
      }
      for (int l=lvls-1; l>=0; l--)
        ldata[l] = LInfo_t(fp[l], handles[0], max_streams, &A);
    }

    auto peak_dmem = peak_device_memory(ldata);
    const auto free_device=gpu::available_memory();
    if (hodlr_exact ? peak_dmem>exact_front_device_budget(free_device) :
        peak_dmem>=0.9*free_device) {
      if (exact_front_stats) ++exact_front_stats->memory_splits;
      if (opts.verbose())
        std::cout << "# Factorization does not fit in GPU memory, "
          "splitting in smaller traversals." << std::endl;
      return split_smaller(A, opts, etree_level, task_depth);
    }

    std::vector<gpu::Stream> streams(max_streams);
    gpu::Stream copy_stream;
    for (int i=0; i<max_streams; i++)
      handles[i].set_stream(streams[i]);
    std::size_t max_small_fronts = 0, max_pinned = 0;
    for (int l=lvls-1; l>=0; l--) {
      auto& L = ldata[l];
      max_small_fronts = std::max(max_small_fronts, L.N8+L.N16+L.N24+L.N32);
      for (auto& f : L.f) {
        const std::size_t dsep = f->dim_sep();
        const std::size_t dupd = f->dim_upd();
        std::size_t fs = dsep*(dsep + 2*dupd);
        max_pinned = std::max(max_pinned, fs);
      }
      max_pinned = std::max(max_pinned, L.factors_small);
    }
    gpu::HostMemory<scalar_t> pinned(2*max_pinned);
    gpu::HostMemory<gpu::FrontData<scalar_t>> fdata(max_small_fronts);
    std::size_t peak_hea_mem = 0;
    for (int l=lvls-1; l>=0; l--)
      peak_hea_mem = std::max(peak_hea_mem, ldata[l].ea_bytes);
    gpu::HostMemory<char> hea_mem(peak_hea_mem);
    gpu::DeviceMemory<char> all_dmem;
    if (hodlr_exact) {
      try {
        // No front pointers or host factors have been changed yet.
        all_dmem=gpu::DeviceMemory<char>(peak_dmem);
      } catch (const std::bad_alloc&) {
        pinned.release();fdata.release();hea_mem.release();
        handles.clear();streams.clear();
        exact_front_allocation_error();
        if (exact_front_stats) ++exact_front_stats->allocation_fallbacks;
        return split_smaller(A,opts,etree_level,task_depth);
      }
      if (exact_front_stats)
        exact_front_stats->device_peak_bytes=std::max(exact_front_stats->device_peak_bytes,peak_dmem);
    } else all_dmem=gpu::DeviceMemory<char>(peak_dmem);
    char* old_work = nullptr;
    for (int l=lvls-1; l>=0; l--) {
      // TaskTimer tl("");
      // tl.start();
      auto& L = ldata[l];
      // if (opts.verbose()) L.print_info(l, lvls);
      try {
        char *work_mem = nullptr, *dea_mem = nullptr;
        scalar_t* dev_factors = nullptr;
        if (l % 2) {
          work_mem = all_dmem;
          dea_mem = work_mem + L.work_bytes;
          dev_factors = gpu::aligned_ptr<scalar_t>(dea_mem + L.ea_bytes);
        } else {
          work_mem = all_dmem + peak_dmem - L.work_bytes;
          dea_mem = work_mem - L.ea_bytes;
          dev_factors = gpu::aligned_ptr<scalar_t>(dea_mem - L.factor_bytes);
        }
        gpu::memset<scalar_t>(work_mem, 0, L.Schur_size);
        gpu::memset<scalar_t>(dev_factors, 0, L.factor_size);
        L.set_factor_pointers(dev_factors);
        L.set_work_pointers(work_mem, max_streams);
        old_work = work_mem;

        // default stream
        front_assembly(A, L, hea_mem, dea_mem);
        gpu::Event e_assemble;
        e_assemble.record();

        // default stream
        factor_small_fronts(L, fdata, L.dev_getrf_err, opts);
        gpu::Event e_small;
        e_small.record();

        for (auto& s : streams)
          e_assemble.wait(s);

        // larger fronts in multiple streams.  Copy back in nchunks
        // chunks, but a single chunk cannot be larger than the pinned
        // buffer
        const int nchunks = 16;
        std::size_t Bf = (L.f.size()-L.small_fronts + nchunks - 1) / nchunks;
        std::vector<std::size_t> chunks, factors_chunk;
        for (std::size_t n=L.small_fronts, fc=0, c=0; n<L.f.size(); n++) {
          auto& f = *(L.f[n]);
          const std::size_t dsep = f.dim_sep();
          const std::size_t dupd = f.dim_upd();
          std::size_t size_front = dsep * (dsep + 2*dupd);
          if (c == Bf || fc + size_front > max_pinned) {
            chunks.push_back(c);
            factors_chunk.push_back(fc);
            c = fc = 0;
          }
          c++;
          fc += size_front;
          if (n == L.f.size()-1) { // final chunk
            chunks.push_back(c);
            factors_chunk.push_back(fc);
          }
        }

        e_small.wait(copy_stream);
        gpu::copy_device_to_host_async<scalar_t>
          (pinned, dev_factors, L.factors_small, copy_stream);

        STRUMPACK_ADD_MEMORY(L.factor_bytes);
        L.f[0]->host_factors_.reset(new scalar_t[L.factor_size]);
        scalar_t* host_factors = L.f[0]->host_factors_.get();
        copy_stream.synchronize();
#pragma omp parallel for
        for (std::size_t i=0; i<L.factors_small; i++)
          host_factors[i] = pinned[i];
        host_factors += L.factors_small;

        if (!chunks.empty()) {
          scalar_t* pin[2] = {pinned.template as<scalar_t>(),
            pinned.template as<scalar_t>() + max_pinned};
          std::vector<gpu::Event> events(chunks.size());

          for (std::size_t c=0, n=L.small_fronts; c<chunks.size(); c++) {
            int s = c % streams.size(), n0 = n;
#pragma omp parallel
#pragma omp single
            {
              if (c) {
#pragma omp task
                {
                  copy_stream.synchronize();
                  auto fc = factors_chunk[c-1];
#if defined(STRUMPACK_USE_OPENMP_TASKLOOP)
#pragma omp taskloop //num_tasks(omp_get_num_threads()-1)
#endif
                  for (std::size_t i=0; i<fc; i++)
                    host_factors[i] = pin[(c-1) % 2][i];
                  host_factors += fc;
                }
              }
#pragma omp task
              {
                for (std::size_t ci=0; ci<chunks[c]; ci++, n++) {
                  auto& f = *(L.f[n]);
                  gpu::getrf
                    (handles[s], f.F11_, L.dev_getrf_work + s * L.getrf_work_size,
                     L.getrf_work_size, f.piv_, L.dev_getrf_err + n);
                  if (opts.replace_tiny_pivots())
                    gpu::replace_pivots
                      (f.dim_sep(), f.F11_.data(),
                       opts.pivot_threshold(), &streams[s]);
                  if (f.dim_upd()) {
                    gpu::getrs
                      (handles[s], Trans::N, f.F11_, f.piv_,
                       f.F12_, L.dev_getrf_err + n,
                       L.dev_getrf_work + s * L.getrf_work_size,
                       L.getrf_work_size);
                    gpu::gemm
                      (handles[s], Trans::N, Trans::N,
                       scalar_t(-1.), f.F21_, f.F12_, scalar_t(1.), f.F22_);
                  }
                }
                events[c].record(streams[s]);
                events[c].wait(copy_stream);
                auto& f = *(L.f[n0]);
                gpu::copy_device_to_host_async<scalar_t>
                  (pin[c % 2], f.F11_.data(), factors_chunk[c], copy_stream);
              }
            }
          }
          copy_stream.synchronize();
          auto fc = factors_chunk.back();
#pragma omp parallel for
          for (std::size_t i=0; i<fc; i++)
            host_factors[i] = pin[(chunks.size()-1) % 2][i];
        }

        L.f[0]->pivot_mem_.resize(L.piv_size);
        copy_stream.synchronize();
        gpu::copy_device_to_host
          (L.f[0]->pivot_mem_.data(), L.f[0]->piv_, L.piv_size);
        L.set_factor_pointers(L.f[0]->host_factors_.get());
        L.set_pivot_pointers(L.f[0]->pivot_mem_.data());

        std::vector<int> getrf_infos(L.f.size());
        gpu::copy_device_to_host
          (getrf_infos.data(), L.dev_getrf_err, L.f.size());
        for (auto ierr : getrf_infos)
          if (ierr) {
            err_code = ReturnCode::ZERO_PIVOT;
            break;
          }
      } catch (const std::bad_alloc& e) {
        std::cerr << "Out of memory" << std::endl;
        abort();
      }
      long long level_flops, small_flops;
      L.flops(level_flops, small_flops);
      STRUMPACK_FULL_RANK_FLOPS(level_flops);
      STRUMPACK_FLOPS(small_flops);
      if (exact_front_stats) {
        exact_front_stats->batch_fronts+=L.f.size();
        exact_front_stats->upload_bytes+=L.ea_bytes+
          L.small_fronts*sizeof(gpu::FrontData<scalar_t>);
        exact_front_stats->download_bytes+=L.factor_size*sizeof(scalar_t)+
          (L.piv_size+L.f.size())*sizeof(int);
      }
      // if (opts.verbose()) {
      //   auto level_time = tl.elapsed();
      //   std::cout << "#   GPU Factorization complete, took: "
      //             << level_time << " seconds, "
      //             << level_flops / 1.e9 << " GFLOPS, "
      //             << (float(level_flops) / level_time) / 1.e9
      //             << " GFLOP/s" << std::endl;
      // }
    }
    const std::size_t dupd = dim_upd();
    if (dupd) { // get the contribution block from the device
      host_Schur_.reset(new scalar_t[dupd*dupd]);
      gpu::copy_device_to_host
        (host_Schur_.get(), reinterpret_cast<scalar_t*>(old_work), dupd*dupd);
      F22_ = DenseMW_t(dupd, dupd, host_Schur_.get(), dupd);
      if (exact_front_stats)
        exact_front_stats->download_bytes+=dupd*dupd*sizeof(scalar_t);
    }
    if (exact_front_stats) {
      ++exact_front_stats->batch_subtrees;
      exact_front_stats->batch_seconds+=exact_front_seconds(batch_started);
    }
    return err_code;
  }

  template<typename scalar_t,typename integer_t> void
  FrontGPU<scalar_t,integer_t>::fwd_solve_phase2
  (DenseM_t& b, DenseM_t& bupd, int etree_level, int task_depth) const {
    if (dim_sep()) {
      DenseMW_t bloc(dim_sep(), b.cols(), b, this->sep_begin_, 0);
      F11_.solve_LU_in_place(bloc, piv_, task_depth);
      if (dim_upd()) {
        if (b.cols() == 1)
          gemv(Trans::N, scalar_t(-1.), F21_, bloc,
               scalar_t(1.), bupd, task_depth);
        else
          gemm(Trans::N, Trans::N, scalar_t(-1.), F21_, bloc,
               scalar_t(1.), bupd, task_depth);
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontGPU<scalar_t,integer_t>::bwd_solve_phase1
  (DenseM_t& y, DenseM_t& yupd, int etree_level, int task_depth) const {
    if (dim_sep()) {
      DenseMW_t yloc(dim_sep(), y.cols(), y, this->sep_begin_, 0);
      if (y.cols() == 1) {
        if (dim_upd())
          gemv(Trans::N, scalar_t(-1.), F12_, yupd,
               scalar_t(1.), yloc, task_depth);
      } else {
        if (dim_upd())
          gemm(Trans::N, Trans::N, scalar_t(-1.), F12_, yupd,
               scalar_t(1.), yloc, task_depth);
      }
    }
  }

  template<typename scalar_t,typename integer_t> ReturnCode
  FrontGPU<scalar_t,integer_t>::node_inertia
  (integer_t& neg, integer_t& zero, integer_t& pos) const {
    using real_t = typename RealType<scalar_t>::value_type;
    for (std::size_t i=0; i<F11_.rows(); i++) {
      if (piv_[i] != int(i+1)) return ReturnCode::INACCURATE_INERTIA;
      auto absFii = std::abs(F11_(i, i));
      if (absFii > real_t(0.)) pos++;
      else if (absFii < real_t(0.)) neg++;
      else if (absFii == real_t(0.)) zero++;
      else std::cerr << "F(" << i << "," << i << ")=" << F11_(i,i) << std::endl;
    }
    return ReturnCode::SUCCESS;
  }

  // explicit template instantiations
  template class FrontGPU<float,int>;
  template class FrontGPU<double,int>;
  template class FrontGPU<std::complex<float>,int>;
  template class FrontGPU<std::complex<double>,int>;

  template class FrontGPU<float,long int>;
  template class FrontGPU<double,long int>;
  template class FrontGPU<std::complex<float>,long int>;
  template class FrontGPU<std::complex<double>,long int>;

  template class FrontGPU<float,long long int>;
  template class FrontGPU<double,long long int>;
  template class FrontGPU<std::complex<float>,long long int>;
  template class FrontGPU<std::complex<double>,long long int>;

} // end namespace strumpack
