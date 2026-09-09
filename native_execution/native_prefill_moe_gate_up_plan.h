#pragma once

#include <cstdint>
#include <vector>

#include <Metal/Metal.hpp>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/stream.h"
#include "native_prefill_moe_route_plan.h"

namespace glm53::native_execution {

namespace mx = mlx::core;

// Composed Q256 sparse-MoE ingress: stable route grouping immediately feeds
// an exact Direct-order BM8 gate/up/SwiGLU kernel. Route metadata never crosses
// back to MLX and sorted hidden rows are addressed indirectly from the source.
class NativePrefillMoEGateUpPlan {
public:
  explicit NativePrefillMoEGateUpPlan(int expert_count = 288);

  mx::array execute(const mx::array &hidden, const mx::array &expert_ids,
                    const mx::array &scores,
                    const mx::array &gate_up_weight,
                    const mx::array &gate_up_scale_inv);
  mx::array execute_routed(
      const mx::array &hidden, const mx::array &expert_ids,
      const mx::array &scores, const mx::array &gate_up_weight,
      const mx::array &gate_up_scale_inv, const mx::array &down_weight,
      const mx::array &down_scale_inv);
  mx::array execute_routed_fused(
      const mx::array &hidden, const mx::array &expert_ids,
      const mx::array &scores, const mx::array &gate_up_weight,
      const mx::array &gate_up_scale_inv, const mx::array &down_weight,
      const mx::array &down_scale_inv);

  int query_rows() const { return kQueryRows; }
  int route_rows() const { return kRouteRows; }
  int hidden_size() const { return kHiddenSize; }
  int intermediate_size() const { return kIntermediateSize; }
  int expert_count() const { return expert_count_; }
  uint64_t execution_count() const { return execution_count_; }
  uint64_t dynamic_allocation_count() const { return 0; }
  uint64_t graph_node_count() const { return 0; }
  uint64_t shape_discovery_count() const { return 0; }
  uint64_t host_synchronization_count() const { return 0; }
  uint64_t returned_route_metadata_bytes() const { return 0; }
  uint64_t materialized_sorted_hidden_bytes() const { return 0; }
  uint64_t fused_materialized_routed_down_bytes() const { return 0; }
  uint64_t scratch_bytes() const;
  std::vector<uint64_t> buffer_identities() const;
  mx::array debug_route_order() const { return route_plan_.sorted_route_order(); }
  mx::array debug_sorted_experts() const { return route_plan_.sorted_experts(); }

private:
  static constexpr int kQueryRows = 256;
  static constexpr int kTopK = 8;
  static constexpr int kRouteRows = kQueryRows * kTopK;
  static constexpr int kHiddenSize = 4096;
  static constexpr int kIntermediateSize = 2048;
  static constexpr int kScaleRows = 32;
  static constexpr int kScaleCols = 32;

  int expert_count_;
  mx::Stream stream_;
  NativePrefillMoERoutePlan route_plan_;
  mx::array activated_;
  mx::array routed_down_;
  mx::array routed_output_;
  MTL::ComputePipelineState *gate_up_pipeline_{nullptr};
  MTL::ComputePipelineState *down_pipeline_{nullptr};
  MTL::ComputePipelineState *reduce_pipeline_{nullptr};
  MTL::ComputePipelineState *fused_down_reduce_pipeline_{nullptr};
  std::vector<uint64_t> initial_buffer_identities_;
  uint64_t execution_count_{0};

  void validate_input(const mx::array &array, const char *name,
                      mx::Dtype dtype, size_t elements) const;
  void encode_ingress(const mx::array &hidden, const mx::array &expert_ids,
                      const mx::array &scores,
                      const mx::array &gate_up_weight,
                      const mx::array &gate_up_scale_inv);
};

} // namespace glm53::native_execution
