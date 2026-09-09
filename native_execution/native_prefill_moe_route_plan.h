#pragma once

#include <cstdint>
#include <vector>

#include <Metal/Metal.hpp>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/stream.h"

namespace glm53::native_execution {

namespace mx = mlx::core;

// Fixed-geometry route grouping for one 256-row GLM-5.3 sparse-MoE chunk.
// The plan deliberately does not materialize sorted hidden rows: later expert
// kernels consume sorted_route_order / top-k to address the original hidden
// tile directly. Equal expert ids retain original route order, matching the
// authoritative MLX argsort/inverse-argsort contract.
class NativePrefillMoERoutePlan {
public:
  explicit NativePrefillMoERoutePlan(int expert_count = 288);

  std::vector<mx::array> execute(const mx::array &expert_ids,
                                 const mx::array &scores);

  int query_rows() const { return kQueryRows; }
  int top_k() const { return kTopK; }
  int route_rows() const { return kRouteRows; }
  int expert_count() const { return expert_count_; }
  int tile_rows() const { return kTileRows; }
  int descriptor_capacity() const { return descriptor_capacity_; }
  uint64_t execution_count() const { return execution_count_; }
  uint64_t dynamic_allocation_count() const { return 0; }
  uint64_t graph_node_count() const { return 0; }
  uint64_t shape_discovery_count() const { return 0; }
  uint64_t host_synchronization_count() const { return 0; }
  uint64_t materialized_sorted_hidden_bytes() const { return 0; }
  uint64_t scratch_bytes() const;
  std::vector<uint64_t> buffer_identities() const;

  mx::array sorted_route_order() const { return sorted_route_order_; }
  mx::array inverse_route_order() const { return inverse_route_order_; }
  mx::array sorted_experts() const { return sorted_experts_; }
  mx::array sorted_scores() const { return sorted_scores_; }
  mx::array expert_offsets() const { return expert_offsets_; }
  mx::array tile_experts() const { return tile_experts_; }
  mx::array tile_starts() const { return tile_starts_; }
  mx::array tile_lengths() const { return tile_lengths_; }
  mx::array descriptor_count() const { return descriptor_count_; }
  mx::array invalid_route_count() const { return invalid_route_count_; }

private:
  static constexpr int kQueryRows = 256;
  static constexpr int kTopK = 8;
  static constexpr int kRouteRows = kQueryRows * kTopK;
  static constexpr int kTileRows = 8;

  int expert_count_;
  int descriptor_capacity_;
  mx::Stream stream_;
  mx::array sorted_route_order_;
  mx::array inverse_route_order_;
  mx::array sorted_experts_;
  mx::array sorted_scores_;
  mx::array expert_offsets_;
  mx::array tile_experts_;
  mx::array tile_starts_;
  mx::array tile_lengths_;
  mx::array descriptor_count_;
  mx::array invalid_route_count_;
  MTL::ComputePipelineState *group_pipeline_{nullptr};
  MTL::ComputePipelineState *offset_pipeline_{nullptr};
  MTL::ComputePipelineState *scatter_pipeline_{nullptr};
  MTL::ComputePipelineState *descriptor_pipeline_{nullptr};
  std::vector<uint64_t> initial_buffer_identities_;
  uint64_t execution_count_{0};

  void validate_input(const mx::array &array, const char *name,
                      mx::Dtype dtype) const;
};

} // namespace glm53::native_execution
