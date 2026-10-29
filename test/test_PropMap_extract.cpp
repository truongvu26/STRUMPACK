// CPU fixture for the actual PropMapSparseMatrix virtual extractor.
// No private-header rewriting and no copy of the extraction implementation.
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "sparse/PropMapSparseMatrix.hpp"
#include "dense/DistributedMatrix.hpp"

using namespace strumpack;

template<typename T> struct Entry {
  static T get(int r, int c) { return T(257*r+c+1); }
};
template<typename R> struct Entry<std::complex<R>> {
  static std::complex<R> get(int r, int c) {
    return {R(257*r+c+1), R(r-2*c)};
  }
};

template<typename T,typename Index> class Fixture final
  : public PropMapSparseMatrix<T,Index> {
public:
  // mode: 0 complete local CSC; 1 missing local columns; 2 no local columns.
  Fixture(int size, int mode) : size_(size), mode_(mode) {
    this->n_ = size;
    this->symm_sparse_ = false;
    this->ptr_.push_back(0);
    for (int c=0; c<size; c++) {
      if (!contains(c)) continue;
      this->global_col_.push_back(c);
      std::vector<int> stored_rows;
      if (c%11!=0) stored_rows={c,(3*c+1)%size,(7*c+5)%size};
      std::sort(stored_rows.begin(),stored_rows.end());
      stored_rows.erase(std::unique(stored_rows.begin(),stored_rows.end()),stored_rows.end());
      for (int r:stored_rows) {
        this->ind_.push_back(r);
        this->val_.push_back(Entry<T>::get(r,c));
        // The production extractor historically selects the FIRST stored
        // occurrence of a repeated row, rather than summing or taking last.
        if (r==c && c%5==0) {
          this->ind_.push_back(r);
          this->val_.push_back(T(-777777));
        }
      }
      this->ptr_.push_back(this->ind_.size());
    }
    this->local_cols_ = this->global_col_.size();
    this->nnz_ = this->ind_.size();
  }

  bool contains(int c) const {
    return mode_!=2 && (mode_==0 || c%3!=1);
  }
  bool nonzero(int r, int c) const {
    // Include stored empty columns as well as absent columns.
    return c%11!=0 && (r==c || r==(3*c+1)%size_ || r==(7*c+5)%size_);
  }
  T reference(int r, int c, int shi) const {
    return contains(c) && nonzero(r,c) && (r<shi || c<shi)
      ? Entry<T>::get(r,c) : T(0);
  }
private:
  int size_, mode_;
};

template<typename T,typename Index> std::uint64_t check_block
  (const Fixture<T,Index>& matrix, const BLACSGrid& grid,
   const std::vector<std::size_t>& rows,
   const std::vector<std::size_t>& cols, int shi, const char* name) {
  DistributedMatrix<T> block(&grid,rows.size(),cols.size());
  block.fill(T(-999999));
  matrix.extract_separator_2d(shi,rows,cols,block);
  std::uint64_t checked=0;
  for (int j=0; j<block.lcols(); j++)
    for (int i=0; i<block.lrows(); i++) {
      const auto r=rows[block.rowl2g(i)], c=cols[block.coll2g(j)];
      const auto expected=matrix.reference(r,c,shi);
      if (block(i,j)!=expected)
        throw std::runtime_error(std::string(name)+
          " failed: source=("+std::to_string(r)+","+std::to_string(c)+")");
      checked++;
    }
  return checked;
}

template<typename T,typename Index> std::uint64_t check_type
  (const BLACSGrid& grid, bool mixed_indices) {
  const int size=192, shi=64;
  std::vector<std::size_t> rows(size), cols(size);
  // Arbitrary order, repeated requested rows AND requested columns, multiple
  // block-cyclic tiles in both dimensions, and rectangular sampled blocks.
  for (int i=0; i<size; i++) {
    rows[i]=(i*71)%size;
    cols[i]=(i*53)%size;
  }
  rows.insert(rows.end(),{17,17,0,191,64});
  cols.insert(cols.end(),{7,7,17,17,0});
  std::vector<std::size_t> sep_rows,upd_rows;
  for (auto r:rows) (r<shi ? sep_rows : upd_rows).push_back(r);
  std::uint64_t checked=0;
  for (int mode:{0,1,2}) {
    Fixture<T,Index> matrix(size,mode);
    checked+=check_block(matrix,grid,rows,cols,size,"arbitrary/full mask");
    checked+=check_block(matrix,grid,rows,cols,0,"arbitrary/zero mask");
    checked+=check_block(matrix,grid,sep_rows,cols,shi,"separator rows/mixed columns");
    checked+=check_block(matrix,grid,upd_rows,cols,shi,"update rows/mixed columns");
    checked+=check_block(matrix,grid,{},cols,shi,"empty rows");
    checked+=check_block(matrix,grid,rows,{},shi,"empty columns");
    if (mixed_indices)
      checked+=check_block(matrix,grid,rows,cols,shi,"mixed unsorted rows/shi mask");
  }
  return checked;
}

std::uint64_t check_grid(const BLACSGrid& grid, bool mixed_indices) {
  std::uint64_t checked=0;
  checked+=check_type<float,int>(grid,mixed_indices);
  checked+=check_type<double,int>(grid,mixed_indices);
  checked+=check_type<std::complex<float>,int>(grid,mixed_indices);
  checked+=check_type<std::complex<double>,int>(grid,mixed_indices);
  checked+=check_type<double,long>(grid,mixed_indices);
  checked+=check_type<std::complex<double>,long long>(grid,mixed_indices);
  return checked;
}

void benchmark(const MPIComm& comm, const BLACSGrid& grid) {
  constexpr int size=40000, columns=512, repeats=3;
  Fixture<double,int> matrix(size,0);
  std::vector<std::size_t> rows(size),cols(columns);
  for (int i=0; i<size; i++) rows[i]=(std::size_t(i)*7919)%size;
  for (int j=0; j<columns; j++) cols[j]=(std::size_t(j)*1543)%size;
  DistributedMatrix<double> block(&grid,size,columns);
  matrix.extract_separator_2d(size,rows,cols,block);
  comm.barrier();
  const auto started=MPI_Wtime();
  for (int k=0; k<repeats; k++)
    matrix.extract_separator_2d(size,rows,cols,block);
  const auto seconds=comm.all_reduce((MPI_Wtime()-started)/repeats,MPI_MAX);
  // Independent dense oracle checks the timed implementation, too.
  for (int j=0; j<block.lcols(); j++)
    for (int i=0; i<block.lrows(); i++)
      if (block(i,j)!=matrix.reference(rows[block.rowl2g(i)],cols[block.coll2g(j)],size))
        throw std::runtime_error("timed sparse extraction differs from dense oracle");
  if (comm.is_root())
    std::cout << "PropMap extraction benchmark: rows=" << size << " cols=" << columns
      << " ranks=" << comm.size() << " max_rank_seconds=" << seconds << '\n';
}

int main(int argc,char** argv) {
  MPI_Init(&argc,&argv);
  try {
    {
      MPIComm world(MPI_COMM_WORLD);
      bool mixed=true,bench=false;
      for (int i=1; i<argc; i++) {
        if (std::string(argv[i])=="--legacy-safe") mixed=false;
        else if (std::string(argv[i])=="--benchmark") bench=true;
        else throw std::runtime_error("unknown argument");
      }
      std::uint64_t checked=0;
      {
        BLACSGrid full(world);
        checked+=check_grid(full,mixed);
        if (bench) benchmark(world,full);
      }
      {
        // Every rank also exercises a local output grid (the callback's
        // usual 1x1 destination representation).
        BLACSGrid local(MPIComm(MPI_COMM_SELF),1);
        checked+=check_grid(local,mixed);
      }
      {
        // Null grids must return without touching data or calling collectives.
        BLACSGrid inactive(MPIComm(MPI_COMM_NULL),world.size());
        checked+=check_grid(inactive,mixed);
      }
      const auto all_checked=world.all_reduce(checked,MPI_SUM);
      if (world.is_root())
        std::cout << "PropMap 2D extraction: PASS checked=" << all_checked
          << " ranks=" << world.size() << " mixed_unsorted_rows=" << mixed << '\n';
    }
    scalapack::Cblacs_exit(1);
    MPI_Finalize();
  } catch (const std::exception& error) {
    std::cerr << "PropMap 2D extraction: FAIL " << error.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD,1);
  }
}
