/* Experimental bounded GPU frontal assembly; host-only public interface. */
#ifndef STRUMPACK_GPU_FRONT_EVALUATOR_HPP
#define STRUMPACK_GPU_FRONT_EVALUATOR_HPP

#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

struct bpack_gpu_block_batch;

namespace strumpack { namespace HODLR {

// All indices use the matrix's APPLICATION ordering, before BPACK clustering.
// Contribution is column-major: embedded children plus, for F22, the current
// negative Schur product. F22 has an empty original sparse matrix.
template<typename T> struct GPUFrontData {
  int rows = 0;
  std::vector<std::int64_t> ptr, col;
  std::vector<T> sparse, contribution;
};

class GPUFrontEvaluator {
public:
  virtual ~GPUFrontEvaluator() = default;
  virtual bool complex_values() const = 0;
  virtual int rows() const = 0;
  virtual void register_matrix(void** matrix) = 0;
  // Fence queued reads before prepared inputs or child work can be released.
  virtual void finish() = 0;
  virtual std::size_t calls() const = 0;
  virtual std::size_t upload_bytes() const = 0;
  virtual std::size_t scratch_peak_bytes() const = 0;
  // The ABI entry point is exposed for stream/stride contract regression tests.
  virtual void evaluate_blocks(const bpack_gpu_block_batch& batch) noexcept = 0;
};

template<typename T> std::shared_ptr<GPUFrontEvaluator>
make_gpu_front_evaluator(GPUFrontData<T>&& data);

}} // namespace strumpack::HODLR
#endif
