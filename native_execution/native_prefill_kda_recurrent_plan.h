#pragma once

#include <cstdint>
#include <vector>

#include <Metal/Metal.hpp>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/stream.h"

namespace glm53::native_execution {

namespace mx = mlx::core;

// Fixed Q256 GLM-5.3 KDA recurrence. The R=4 kernel preserves the Direct
// token, feature, SIMD-reduction, FP32-state update, and BF16-output order.
// Output and next state are plan-owned so the primitive can later be encoded
// directly inside the model-wide prefill plan without an MLX graph boundary.
class NativePrefillKDARecurrentPlan {
public:
  NativePrefillKDARecurrentPlan();

  std::vector<mx::array> execute(
      const mx::array &q, const mx::array &k, const mx::array &v,
      const mx::array &g, const mx::array &beta, const mx::array &state,
      const mx::array &mask, bool has_mask);

  int query_rows() const { return kQueryRows; }
  int heads() const { return kHeads; }
  int key_dim() const { return kKeyDim; }
  int value_dim() const { return kValueDim; }
  int row_block() const { return kRowBlock; }
  uint64_t execution_count() const { return execution_count_; }
  uint64_t dynamic_allocation_count() const { return 0; }
  uint64_t graph_node_count() const { return 0; }
  uint64_t shape_discovery_count() const { return 0; }
  uint64_t host_synchronization_count() const { return 0; }
  uint64_t returned_intermediate_tensor_bytes() const { return 0; }
  uint64_t arena_bytes() const;
  bool buffer_identities_stable() const;
  std::vector<uint64_t> buffer_identities() const;

private:
  static constexpr int kQueryRows = 256;
  static constexpr int kHeads = 64;
  static constexpr int kKeyDim = 128;
  static constexpr int kValueDim = 128;
  static constexpr int kRowBlock = 4;

  mx::Stream stream_;
  MTL::ComputePipelineState *pipeline_{nullptr};
  mx::array output_;
  mx::array next_state_;
  std::vector<uint64_t> initial_buffer_identities_;
  uint64_t execution_count_{0};

  void validate(const mx::array &value, const char *name,
                mx::Dtype dtype, size_t elements) const;
};

} // namespace glm53::native_execution
