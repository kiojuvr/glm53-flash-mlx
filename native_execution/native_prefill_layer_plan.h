#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <Metal/Metal.hpp>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/stream.h"

namespace glm53::native_execution {

namespace mx = mlx::core;

// Arithmetic-free feasibility substrate for the composed prefill layer plan.
// It owns the complete 512K/256-row arena now, so later DSA and MoE kernels
// replace the pass-through stages without changing storage or submission
// ownership. Only the final hidden buffer escapes to MLX.
class NativePrefillLayerSubstrate {
public:
  NativePrefillLayerSubstrate();

  std::vector<mx::array> execute(const mx::array &hidden);

  int query_rows() const { return kQueryRows; }
  int query_block_rows() const { return kQueryBlockRows; }
  int query_block_count() const { return kQueryBlockCount; }
  int hidden_size() const { return kHiddenSize; }
  int physical_pool_rows() const { return kPhysicalPoolRows; }
  int selected_width() const { return kSelectedWidth; }
  int route_rows() const { return kRouteRows; }
  uint64_t execution_count() const { return execution_count_; }
  uint64_t dynamic_allocation_count() const { return 0; }
  uint64_t graph_node_count() const { return 0; }
  uint64_t shape_discovery_count() const { return 0; }
  uint64_t host_synchronization_count() const { return 0; }
  uint64_t native_encoder_scopes_per_execute() const { return 1; }
  uint64_t startup_pipeline_lookup_count() const { return 1; }
  uint64_t pipeline_lookup_count_per_execute() const { return 0; }
  uint64_t returned_intermediate_tensor_bytes() const { return 0; }
  uint64_t dsa_score_scratch_bytes() const;
  uint64_t scratch_bytes() const;
  uint64_t arena_bytes() const;
  mx::array output() const { return hidden_ping_; }
  bool buffer_identities_stable() const;
  std::vector<uint64_t> buffer_identities() const;
  std::vector<std::string> fixed_topology() const;

private:
  static constexpr int kQueryRows = 256;
  static constexpr int kQueryBlockRows = 128;
  static constexpr int kQueryBlockCount = 2;
  static constexpr int kHiddenSize = 4096;
  static constexpr int kPhysicalPoolRows = 131072;
  static constexpr int kSelectedWidth = 2051;
  static constexpr int kTopK = 8;
  static constexpr int kIntermediateSize = 2048;
  static constexpr int kRouteRows = kQueryRows * kTopK;

  mx::Stream stream_;
  MTL::ComputePipelineState *copy_pipeline_{nullptr};
  mx::array hidden_ping_;
  mx::array hidden_pong_;
  mx::array dsa_score_scratch_;
  mx::array selected_indices_;
  mx::array selected_valid_;
  mx::array route_experts_;
  mx::array route_scores_;
  mx::array route_order_;
  mx::array moe_hidden_scratch_;
  mx::array moe_down_scratch_;
  std::vector<uint64_t> initial_buffer_identities_;
  uint64_t execution_count_{0};

  void validate_hidden(const mx::array &hidden) const;
};

} // namespace glm53::native_execution
