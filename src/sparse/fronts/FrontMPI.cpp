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
 * five (5) year renewals, the U.S. Government igs granted for itself
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

#include "FrontMPI.hpp"
#include "FrontBLRMPI.hpp"
#include "ExtendAdd.hpp"
#include "HODLR/StageProfile.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace strumpack {

  template<typename scalar_t,typename integer_t>
  FrontMPI<scalar_t,integer_t>::FrontMPI
  (integer_t sep, integer_t sep_begin, integer_t sep_end,
   std::vector<integer_t>& upd, const MPIComm& comm, int P)
    : F_t(nullptr, nullptr, sep, sep_begin, sep_end, upd),
      blacs_grid_(comm, P) {
  }

  template<typename scalar_t,typename integer_t> integer_t
  FrontMPI<scalar_t,integer_t>::maximum_rank(int task_depth) const {
    auto mr = this->front_rank();
    if (visit(lchild_))
      mr = std::max(mr, lchild_->maximum_rank(task_depth));
    if (visit(rchild_))
      mr = std::max(mr, rchild_->maximum_rank(task_depth));
    return mr;
  }

  template<typename scalar_t,typename integer_t> void
  FrontMPI<scalar_t,integer_t>::extract_2d
  (const SpMat_t& A, const VecVec_t& I, const VecVec_t& J,
   std::vector<DistMW_t>& B, bool skip_sparse) const {
    HODLR_STAGE_SCOPE_IF(p_extract,this->dim_sep()>=1000,"front.extract.parent",this->sep_begin_,this->dim_sep(),this->dim_upd(),Comm().size(),Comm().rank(),0);
    TIMER_TIME(TaskType::EXTRACT_2D, 1, t_ex);
    int rank = Comm().rank(), nprocs = P();
    std::vector<DistM_t> Bl, Br;
    std::vector<DenseM_t> Blseq, Brseq;
    if (visit(lchild_)) {
      HODLR_STAGE_SCOPE_IF(p_left,this->dim_sep()>=1000,"front.extract.child_left",this->sep_begin_,this->dim_sep(),this->dim_upd(),Comm().size(),Comm().rank(),0);
      lchild_->get_submatrix_2d(I, J, Bl, Blseq);
    }
    if (visit(rchild_)) {
      HODLR_STAGE_SCOPE_IF(p_right,this->dim_sep()>=1000,"front.extract.child_right",this->sep_begin_,this->dim_sep(),this->dim_upd(),Comm().size(),Comm().rank(),0);
      rchild_->get_submatrix_2d(I, J, Br, Brseq);
    }
    std::vector<int> bgrid(3*I.size(), -1);
    {
      HODLR_STAGE_SCOPE_IF(p_metadata,this->dim_sep()>=1000,"front.extract.metadata",this->sep_begin_,this->dim_sep(),this->dim_upd(),Comm().size(),Comm().rank(),0);
      for (std::size_t i=0; i<I.size(); i++)
        if (B[i].grid() && B[i].prow() == 0 && B[i].pcol() == 0) {
          bgrid[3*i] = rank;
          bgrid[3*i+1] = B[i].nprows();
          bgrid[3*i+2] = B[i].npcols();
        }
      Comm().all_reduce(bgrid.data(), bgrid.size(), MPI_MAX);
    }
    std::vector<std::vector<scalar_t>> sbuf(nprocs);
    std::size_t ssize = 0;
    for (std::size_t i=0; i<I.size(); i++)
      ssize += I[i].size() * J[i].size();
    for (int p=0; p<nprocs; p++)
      sbuf[p].reserve(std::round(1.2*float(ssize)/nprocs));
    bool vl = visit(lchild_), vr = visit(rchild_);
    for (std::size_t i=0; i<I.size(); i++) {
      if (!skip_sparse) {
        HODLR_STAGE_SCOPE_IF(p_sparse,this->dim_sep()>=1000,"front.extract.sparse",this->sep_begin_,this->dim_sep(),this->dim_upd(),Comm().size(),Comm().rank(),0);
        TIMER_TIME(TaskType::EXTRACT_SEP_2D, 2, t_ex_sep);
        DistM_t tmp(grid(), I[i].size(), J[i].size());
        A.extract_separator_2d(this->sep_end_, I[i], J[i], tmp);
        {
          HODLR_STAGE_SCOPE_IF(p_pack_sparse,this->dim_sep()>=1000,"front.extract.pack_sparse",this->sep_begin_,this->dim_sep(),this->dim_upd(),Comm().size(),Comm().rank(),0);
          subgrid_copy_to_buffers
            (tmp, B[i], bgrid[3*i], bgrid[3*i+1], bgrid[3*i+2], sbuf);
        }
      }
      if (vl) {
        HODLR_STAGE_SCOPE_IF(p_pack_left,this->dim_sep()>=1000,"front.extract.pack_child_left",this->sep_begin_,this->dim_sep(),this->dim_upd(),Comm().size(),Comm().rank(),0);
        if (lchild_->isMPI())
          subgrid_copy_to_buffers
            (Bl[i], B[i], bgrid[3*i], bgrid[3*i+1], bgrid[3*i+2], sbuf);
        else
          subproc_copy_to_buffers
            (Blseq[i], B[i], bgrid[3*i], bgrid[3*i+1], bgrid[3*i+2], sbuf);
      }
      if (vr) {
        HODLR_STAGE_SCOPE_IF(p_pack_right,this->dim_sep()>=1000,"front.extract.pack_child_right",this->sep_begin_,this->dim_sep(),this->dim_upd(),Comm().size(),Comm().rank(),0);
        if (rchild_->isMPI())
          subgrid_copy_to_buffers
            (Br[i], B[i], bgrid[3*i], bgrid[3*i+1], bgrid[3*i+2], sbuf);
        else
          subproc_copy_to_buffers
            (Brseq[i], B[i], bgrid[3*i], bgrid[3*i+1], bgrid[3*i+2], sbuf);
      }
    }
    std::vector<scalar_t,NoInit<scalar_t>> rbuf;
    std::vector<scalar_t*> pbuf;
    {
      HODLR_STAGE_SCOPE_IF(p_exchange,this->dim_sep()>=1000,"front.extract.exchange",this->sep_begin_,this->dim_sep(),this->dim_upd(),Comm().size(),Comm().rank(),0);
      TIMER_TIME(TaskType::EXTRACT_2D_A2A, 1, t_a2a);
      Comm().all_to_all_v(sbuf, rbuf, pbuf);
    }
    BLACSGrid *gl = nullptr, *gr = nullptr;
    if (lchild_) gl = lchild_->grid();
    if (rchild_) gr = rchild_->grid();
    int ml = master(lchild_), mr = master(rchild_);
    {
      HODLR_STAGE_SCOPE_IF(p_unpack,this->dim_sep()>=1000,"front.extract.unpack",this->sep_begin_,this->dim_sep(),this->dim_upd(),Comm().size(),Comm().rank(),0);
      for (std::size_t i=0; i<I.size(); i++) {
        if (!B[i].active()) continue;
        B[i].zero();
        if (!skip_sparse) subgrid_add_from_buffers(grid(), 0, B[i], pbuf);
        if (lchild_) subgrid_add_from_buffers(gl, ml, B[i], pbuf);
        if (rchild_) subgrid_add_from_buffers(gr, mr, B[i], pbuf);
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontMPI<scalar_t,integer_t>::extract_2d_compact
  (const SpMat_t& A, const VecVec_t& I, const VecVec_t& J,
   std::vector<DistMW_t>& B, const std::vector<int>& parent_owners,
   bool skip_sparse) const {
    HODLR_STAGE_SCOPE_IF(p_compact,this->dim_sep()>=1000,"front.extract.compact",this->sep_begin_,this->dim_sep(),this->dim_upd(),Comm().size(),Comm().rank(),0);
    const auto queries=I.size(); const int rank=Comm().rank(),procs=P();
    if (J.size()!=queries || B.size()!=queries || parent_owners.size()!=queries)
      throw std::invalid_argument("Compact extraction query metadata size mismatch");
    const auto limit=std::min(std::size_t(std::numeric_limits<int>::max()),
                             std::numeric_limits<std::size_t>::max()/sizeof(scalar_t));
    auto add_count=[limit](std::size_t& sum,std::size_t count) {
      if (count>limit-sum) throw std::overflow_error("Compact extraction batch exceeds native/MPI count limit");
      sum+=count;
    };
    auto product=[limit](std::size_t m,std::size_t n) {
      if (m>limit || n>limit || (m && n>limit/m))
        throw std::overflow_error("Compact extraction block exceeds native/MPI count limit");
      return m*n;
    };
    // Every count checked here is replicated query metadata. Fail before any
    // child/parent collective rather than choosing different rank-local paths.
    std::size_t metadata=0,full_entries=0;
    if (queries>limit/3) throw std::overflow_error("Too many compact extraction queries");
    add_count(metadata,3*queries);
    std::vector<std::size_t> destination_counts(procs,0);
    for (std::size_t q=0;q<queries;++q) {
      if (parent_owners[q]<0 || parent_owners[q]>=procs)
        throw std::invalid_argument("Compact extraction owner outside parent communicator");
      add_count(metadata,I[q].size()); add_count(metadata,J[q].size());
      const auto count=product(I[q].size(),J[q].size());
      add_count(full_entries,count);
      if (!skip_sparse) add_count(destination_counts[parent_owners[q]],count);
      if (lchild_) add_count(destination_counts[parent_owners[q]],count);
      if (rchild_) add_count(destination_counts[parent_owners[q]],count);
    }
    struct Child {
      const F_t* front=nullptr; int master=0,procs=1; bool active=false,owned=false;
      VecVec_t rows,cols,out_rows,out_cols;
      std::vector<int> owners;
      std::vector<DistM_t> distributed;
      std::vector<DenseM_t> sequential;
    };
    auto prepare=[&](const F_t* front) {
      Child child; child.front=front;
      if (!front) return child;
      HODLR_STAGE_SCOPE_IF(p_map,this->dim_sep()>=1000,"front.extract.compact.map",this->sep_begin_,this->dim_sep(),this->dim_upd(),Comm().size(),Comm().rank(),0);
      child.master=master(front); child.procs=front->P(); child.active=visit(front);
      if (child.procs<=0) throw std::invalid_argument("Invalid compact extraction child process count");
      child.rows.resize(queries); child.cols.resize(queries);
      child.out_rows.resize(queries); child.out_cols.resize(queries); child.owners.resize(queries);
      std::size_t child_entries=0;
      for (std::size_t q=0;q<queries;++q) {
        // A parent owner may be far outside a small right child's subgroup.
        // Use floor-mod, not one addition followed by a signed remainder.
        const auto diff=static_cast<long long>(parent_owners[q])-child.master;
        child.owners[q]=int((diff%child.procs+child.procs)%child.procs);
        // Only child participants need native queries, and only the final
        // output owner needs the scatter map. Other parent ranks must not
        // repeat these potentially long index searches for a remote child.
        if (!child.active && rank!=parent_owners[q]) continue;
        std::vector<std::size_t> local_rows,local_cols;
        front->find_upd_indices(I[q],local_rows,child.out_rows[q]);
        front->find_upd_indices(J[q],local_cols,child.out_cols[q]);
        for (auto p:child.out_rows[q]) child.rows[q].push_back(I[q][p]);
        for (auto p:child.out_cols[q]) child.cols[q].push_back(J[q][p]);
        const auto count=product(child.rows[q].size(),child.cols[q].size());
        add_count(child_entries,count);
      }
      return child;
    };
    auto left=prepare(lchild_.get()),right=prepare(rchild_.get());
    std::size_t transfer_bound=0;
    // Individual messages have int counts (checked above). The MPI wrapper
    // uses Isend/Irecv with size_t offsets, so the total only needs to fit
    // addressable scalar storage rather than a single int displacement.
    const auto max_elements=std::numeric_limits<std::size_t>::max()/sizeof(scalar_t);
    for (auto count:destination_counts) {
      if (count>max_elements-transfer_bound)
        throw std::overflow_error("Compact extraction payload exceeds addressable storage");
      transfer_bound+=count;
    }
    auto extract=[&](Child& child,const char* label) {
      if (!child.front) return;
      HODLR_STAGE_SCOPE_IF(p_child,this->dim_sep()>=1000,label,this->sep_begin_,this->dim_sep(),this->dim_upd(),Comm().size(),Comm().rank(),0);
      child.owned=child.front->get_submatrix_2d_owned
        (child.rows,child.cols,child.owners,child.sequential,child.active);
      if (!child.owned && child.active)
        child.front->get_submatrix_2d(child.rows,child.cols,child.distributed,child.sequential);
    };
    extract(left,"front.extract.compact.child_left");
    extract(right,"front.extract.compact.child_right");
    std::vector<std::vector<scalar_t>> sbuf(procs);
    auto append=[&](int owner,const scalar_t* data,int rows,int cols,int ld) {
      if (!rows || !cols) return;
      auto& out=sbuf[owner];
      for (int c=0;c<cols;++c)
        out.insert(out.end(),data+std::size_t(c)*ld,data+std::size_t(c)*ld+rows);
    };
    auto pack=[&](const Child& child,std::size_t q) {
      if (!child.front || !child.active) return;
      HODLR_STAGE_SCOPE_IF(p_pack,this->dim_sep()>=1000,child.owned?"front.extract.compact.pack_owned":"front.extract.compact.pack_grid",this->sep_begin_,this->dim_sep(),this->dim_upd(),Comm().size(),Comm().rank(),0);
      if (child.owned) {
        if (rank!=child.master+child.owners[q]) return;
        const auto& data=child.sequential[q];
        assert(data.rows()==child.rows[q].size() && data.cols()==child.cols[q].size());
        append(parent_owners[q],data.data(),data.rows(),data.cols(),data.ld());
      } else {
        const auto& data=child.distributed[q];
        if (data.active())
          append(parent_owners[q],data.data(),data.lrows(),data.lcols(),data.ld());
      }
    };
    // Match the legacy per-query sparse/left/right stream order. Each source
    // rank appends only compact child values, never a zero-filled full query.
    for (std::size_t q=0;q<queries;++q) {
      if (!skip_sparse) {
        HODLR_STAGE_SCOPE_IF(p_sparse,this->dim_sep()>=1000,"front.extract.compact.sparse",this->sep_begin_,this->dim_sep(),this->dim_upd(),Comm().size(),Comm().rank(),0);
        DistM_t tmp(grid(),I[q].size(),J[q].size());
        A.extract_separator_2d(this->sep_end_,I[q],J[q],tmp);
        if (tmp.active())
          append(parent_owners[q],tmp.data(),tmp.lrows(),tmp.lcols(),tmp.ld());
      }
      pack(left,q); pack(right,q);
    }
    std::vector<scalar_t,NoInit<scalar_t>> rbuf;
    std::vector<scalar_t*> pbuf;
    {
      HODLR_STAGE_SCOPE_IF(p_exchange,this->dim_sep()>=1000,"front.extract.compact.exchange",this->sep_begin_,this->dim_sep(),this->dim_upd(),Comm().size(),Comm().rank(),([&](){std::size_t n=0;for(const auto& v:sbuf)n+=v.size()*sizeof(scalar_t);return n;})());
      Comm().all_to_all_v(sbuf,rbuf,pbuf);
    }
    auto add=[&](const Child& child,std::size_t q) {
      if (!child.front) return;
      const auto& rows=child.out_rows[q]; const auto& cols=child.out_cols[q];
      const auto* g=child.front->grid();
      const int pr=g?g->nprows():1,pc=g?g->npcols():1;
      for (std::size_t c=0;c<cols.size();++c)
        for (std::size_t r=0;r<rows.size();++r) {
          const int source=child.owned?child.master+child.owners[q]:
            child.master+int((r/DistM_t::default_MB)%pr+((c/DistM_t::default_NB)%pc)*pr);
          B[q](rows[r],cols[c])+=*pbuf[source]++;
        }
    };
    {
      HODLR_STAGE_SCOPE_IF(p_unpack,this->dim_sep()>=1000,"front.extract.compact.unpack",this->sep_begin_,this->dim_sep(),this->dim_upd(),Comm().size(),Comm().rank(),0);
      for (std::size_t q=0;q<queries;++q) {
        if (rank!=parent_owners[q]) continue;
        assert(B[q].active() && B[q].nprows()==1 && B[q].npcols()==1);
        B[q].zero();
        if (!skip_sparse) subgrid_add_from_buffers(grid(),0,B[q],pbuf);
        add(left,q); add(right,q);
      }
    }
  }

  template<typename scalar_t,typename integer_t> void
  FrontMPI<scalar_t,integer_t>::get_submatrix_2d
  (const VecVec_t& I, const VecVec_t& J,
   std::vector<DistM_t>& Bdist, std::vector<DenseM_t>&) const {
    TIMER_TIME(TaskType::GET_SUBMATRIX_2D, 3, t_ex_schur);
    for (std::size_t i=0; i<I.size(); i++) {
      Bdist.emplace_back(grid(), I[i].size(), J[i].size());
      Bdist.back().zero();
    }
    extract_CB_sub_matrix_2d(I, J, Bdist);
  }

  template<typename scalar_t,typename integer_t> void
  FrontMPI<scalar_t,integer_t>::extract_CB_sub_matrix_2d
  (const VecVec_t& I, const VecVec_t& J, std::vector<DistM_t>& B) const {
    for (std::size_t i=0; i<I.size(); i++)
      extract_CB_sub_matrix_2d(I[i], J[i], B[i]);
  }

  /**
   * Check if the child needs to be visited. Not necessary when this
   * rank is not part of the processes assigned to the child.
   */
  template<typename scalar_t,typename integer_t> bool
  FrontMPI<scalar_t,integer_t>::visit
  (const F_t* ch) const {
    if (!ch) return false;
    // TODO do this differently!!
    if (auto mpi_child = dynamic_cast<const FMPI_t*>(ch)) {
      if (mpi_child->Comm().is_null())
        return false; // child is MPI
    } else if (Comm().rank() != master(ch))
      return false; // child is sequential
    return true;
  }

  template<typename scalar_t,typename integer_t> bool
  FrontMPI<scalar_t,integer_t>::visit
  (const std::unique_ptr<F_t>& ch) const {
    return visit(ch.get());
  }

  template<typename scalar_t,typename integer_t> int
  FrontMPI<scalar_t,integer_t>::master
  (const std::unique_ptr<F_t>& ch) const {
    return master(ch.get());
  }

  template<typename scalar_t,typename integer_t> int
  FrontMPI<scalar_t,integer_t>::master(const F_t* ch) const {
    if (!ch) return 0;
    return (ch == lchild_.get()) ? 0 : P() - ch->P();
  }

  template<typename scalar_t,typename integer_t> void
  FrontMPI<scalar_t,integer_t>::extend_add_b
  (DistM_t& b, DistM_t& bupd, const DistM_t& CBl, const DistM_t& CBr,
   const DenseM_t& seqCBl, const DenseM_t& seqCBr) const {
    if (Comm().is_root()) {
      STRUMPACK_FLOPS(static_cast<long long int>(CBl.rows()*b.cols()));
      STRUMPACK_FLOPS(static_cast<long long int>(CBr.rows()*b.cols()));
    }
    std::vector<std::vector<scalar_t>> sbuf(P());
    if (visit(lchild_))
      lchild_->extend_add_column_copy_to_buffers(CBl, seqCBl, sbuf, this);
    if (visit(rchild_))
      rchild_->extend_add_column_copy_to_buffers(CBr, seqCBr, sbuf, this);
    std::vector<scalar_t,NoInit<scalar_t>> rbuf;
    std::vector<scalar_t*> pbuf;
    Comm().all_to_all_v(sbuf, rbuf, pbuf);
    for (auto& ch : {lchild_.get(), rchild_.get()})
      if (ch) ch->extend_add_column_copy_from_buffers
                (b, bupd, pbuf.data()+master(ch), this);
  }

  template<typename scalar_t,typename integer_t> void
  FrontMPI<scalar_t,integer_t>::extract_from_R2D
  (const DistM_t& R, DistM_t& cR, DenseM_t& seqcR,
   const FMPI_t* pa, bool visit) const {
    auto I = this->upd_to_parent(pa);
    cR = DistM_t(grid(), I.size(), R.cols(),
                 R.extract_rows(I), pa->grid()->ctxt_all());
  }

  template<typename scalar_t,typename integer_t> void
  FrontMPI<scalar_t,integer_t>::extract_b
  (const DistM_t& b, const DistM_t& bupd, DistM_t& CBl, DistM_t& CBr,
   DenseM_t& seqCBl, DenseM_t& seqCBr) const {
    std::vector<std::vector<scalar_t>> sbuf(P());
    for (auto ch : {lchild_.get(), rchild_.get()})
      if (ch) ch->extract_column_copy_to_buffers
                (b, bupd, master(ch), sbuf, this);
    std::vector<scalar_t,NoInit<scalar_t>> rbuf;
    std::vector<scalar_t*> pbuf;
    Comm().all_to_all_v(sbuf, rbuf, pbuf);
    if (visit(lchild_))
      lchild_->extract_column_copy_from_buffers(b, CBl, seqCBl, pbuf, this);
    if (visit(rchild_))
      rchild_->extract_column_copy_from_buffers(b, CBr, seqCBr, pbuf, this);
  }

  template<typename scalar_t,typename integer_t> long long
  FrontMPI<scalar_t,integer_t>::dense_factor_nonzeros
  (int task_depth) const {
    long long nnz = 0;
    if (!Comm().is_null() && Comm().is_root()) {
      long long dsep = this->dim_sep();
      long long dupd = this->dim_upd();
      nnz = dsep * (dsep + 2 * dupd);
    }
    if (visit(lchild_))
      nnz += lchild_->dense_factor_nonzeros(task_depth);
    if (visit(rchild_))
      nnz += rchild_->dense_factor_nonzeros(task_depth);
    return nnz;
  }

  template<typename scalar_t,typename integer_t> long long
  FrontMPI<scalar_t,integer_t>::factor_nonzeros
  (int task_depth) const {
    long long nnz = node_factor_nonzeros();
    if (visit(lchild_))
      nnz += lchild_->factor_nonzeros(task_depth);
    if (visit(rchild_))
      nnz += rchild_->factor_nonzeros(task_depth);
    return nnz;
  }

  template<typename scalar_t,typename integer_t> long long
  FrontMPI<scalar_t,integer_t>::node_factor_nonzeros() const {
    long long dsep = this->dim_sep();
    long long dupd = this->dim_upd();
    return Comm().is_root() ? dsep * (dsep + 2 * dupd) : 0;
  }

  template<typename scalar_t,typename integer_t> void
  FrontMPI<scalar_t,integer_t>::partition_fronts
  (const Opts_t& opts, const SpMat_t& A, integer_t* sorder,
   bool is_root, int task_depth) {
    if (visit(lchild_))
      lchild_->partition_fronts(opts, A, sorder, false, task_depth+1);
    if (visit(rchild_))
      rchild_->partition_fronts(opts, A, sorder, false, task_depth+1);
    this->partition(opts, A, sorder, is_root, task_depth);
  }

  template<typename scalar_t,typename integer_t> void
  FrontMPI<scalar_t,integer_t>::extend_add_copy_from_buffers
  (DistM_t& F11, DistM_t& F12, DistM_t& F21, DistM_t& F22, scalar_t** pbuf,
   const FrontMPI<scalar_t,integer_t>* pa) const {
    ExtendAdd<scalar_t,integer_t>::extend_add_copy_from_buffers
      (F11, F12, F21, F22, pbuf, pa, this);
  }

  template<typename scalar_t,typename integer_t> void
  FrontMPI<scalar_t,integer_t>::extend_add_column_copy_to_buffers
  (const DistM_t& CB, const DenseM_t& seqCB,
   std::vector<std::vector<scalar_t>>& sbuf, const FMPI_t* pa) const {
    ExtendAdd<scalar_t,integer_t>::extend_add_column_copy_to_buffers
      (CB, sbuf, pa, this->upd_to_parent(pa));
  }

  template<typename scalar_t,typename integer_t> void
  FrontMPI<scalar_t,integer_t>::extend_add_column_copy_from_buffers
  (DistM_t& B, DistM_t& Bupd, scalar_t** pbuf,
   const FrontMPI<scalar_t,integer_t>* pa) const {
    ExtendAdd<scalar_t,integer_t>::extend_add_column_copy_from_buffers
      (B, Bupd, pbuf, pa, this);
  }

  template<typename scalar_t,typename integer_t> void
  FrontMPI<scalar_t,integer_t>::extract_column_copy_to_buffers
  (const DistM_t& b, const DistM_t& bupd, int ch_master,
   std::vector<std::vector<scalar_t>>& sbuf, const FMPI_t* pa) const {
    ExtendAdd<scalar_t,integer_t>::extract_column_copy_to_buffers
      (b, bupd, sbuf, pa, this);
  }

  template<typename scalar_t,typename integer_t> void
  FrontMPI<scalar_t,integer_t>::extract_column_copy_from_buffers
  (const DistM_t& b, DistM_t& CB, DenseM_t& seqCB,
   std::vector<scalar_t*>& pbuf, const FMPI_t* pa) const {
    CB = DistM_t(grid(), this->dim_upd(), b.cols());
    ExtendAdd<scalar_t,integer_t>::extract_column_copy_from_buffers
      (CB, pbuf, pa, this);
  }

  template<typename scalar_t,typename integer_t> void
  FrontMPI<scalar_t,integer_t>::skinny_ea_to_buffers
  (const DistM_t& S, const DenseM_t& seqS,
   std::vector<std::vector<scalar_t>>& sbuf, const FMPI_t* pa) const {
    ExtendAdd<scalar_t,integer_t>::skinny_extend_add_copy_to_buffers
      (S, sbuf, pa, this->upd_to_parent(pa));
  }

  template<typename scalar_t,typename integer_t> void
  FrontMPI<scalar_t,integer_t>::skinny_ea_from_buffers
  (DistM_t& S, scalar_t** pbuf, const FMPI_t* pa) const {
    ExtendAdd<scalar_t,integer_t>::skinny_extend_add_copy_from_buffers
      (S, pbuf, pa, this);
  }

  // explicit template instantiations
  template class FrontMPI<float,int>;
  template class FrontMPI<double,int>;
  template class FrontMPI<std::complex<float>,int>;
  template class FrontMPI<std::complex<double>,int>;

  template class FrontMPI<float,long int>;
  template class FrontMPI<double,long int>;
  template class FrontMPI<std::complex<float>,long int>;
  template class FrontMPI<std::complex<double>,long int>;

  template class FrontMPI<float,long long int>;
  template class FrontMPI<double,long long int>;
  template class FrontMPI<std::complex<float>,long long int>;
  template class FrontMPI<std::complex<double>,long long int>;

} // end namespace strumpack
