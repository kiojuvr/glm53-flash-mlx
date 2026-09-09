#pragma once

#include <cstdint>
#include <vector>

#include <Metal/Metal.hpp>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/stream.h"

namespace glm53::native_execution {

namespace mx = mlx::core;

// Actual GLM-5.3 sparse-attention output geometry. The head-major
// [64,256,256] attention result is made row-major once in a fixed arena and
// immediately consumed by the exact block-128 E4M3 o_proj [4096,16384].
class NativePrefillDSAOutputPlan {
public:
  NativePrefillDSAOutputPlan();

  mx::array execute(const mx::array &head_major,
                    const mx::array &weight,
                    const mx::array &scale_inv);
  mx::array execute_hc(const mx::array &head_major,
                       const mx::array &weight,
                       const mx::array &scale_inv,
                       const mx::array &residual,
                       const mx::array &post,
                       const mx::array &comb);

  int query_rows() const { return kQueryRows; }
  int heads() const { return kHeads; }
  int value_dim() const { return kValueDim; }
  int hidden_size() const { return kHiddenSize; }
  uint64_t execution_count() const { return execution_count_; }
  uint64_t dynamic_allocation_count() const { return 0; }
  uint64_t graph_node_count() const { return 0; }
  uint64_t shape_discovery_count() const { return 0; }
  uint64_t host_synchronization_count() const { return 0; }
  uint64_t returned_intermediate_tensor_bytes() const { return 0; }
  uint64_t scratch_bytes() const { return row_major_.nbytes() + output_.nbytes(); }
  uint64_t hc_scratch_bytes() const {
    return scratch_bytes() + hc_output_.nbytes();
  }
  std::vector<uint64_t> buffer_identities() const;
  mx::array debug_row_major() const { return row_major_; }

private:
  static constexpr int kQueryRows = 256;
  static constexpr int kHeads = 64;
  static constexpr int kValueDim = 256;
  static constexpr int kInputSize = kHeads * kValueDim;
  static constexpr int kHiddenSize = 4096;

  mx::Stream stream_;
  mx::array row_major_;
  mx::array output_;
  mx::array hc_output_;
  MTL::ComputePipelineState *transpose_pipeline_{nullptr};
  MTL::ComputePipelineState *projection_pipeline_{nullptr};
  MTL::ComputePipelineState *hc_expand_pipeline_{nullptr};
  std::vector<uint64_t> initial_buffer_identities_;
  uint64_t execution_count_{0};

  void validate(const mx::array &value, const char *name,
                mx::Dtype dtype, size_t elements) const;
  void encode_projection(const mx::array &head_major,
                         const mx::array &weight,
                         const mx::array &scale_inv);
};

} // namespace glm53::native_execution
