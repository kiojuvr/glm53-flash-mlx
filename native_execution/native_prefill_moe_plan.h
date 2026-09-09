#pragma once

#include <cstdint>
#include <vector>

#include <Metal/Metal.hpp>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/stream.h"
#include "native_prefill_moe_gate_up_plan.h"
#include "native_prefill_shared_expert_plan.h"

namespace glm53::native_execution {

namespace mx = mlx::core;

class NativePrefillMoEPlan {
public:
  explicit NativePrefillMoEPlan(int expert_count = 288);

  mx::array execute(
      const mx::array &hidden, const mx::array &expert_ids,
      const mx::array &scores, const mx::array &gate_up_weight,
      const mx::array &gate_up_scale_inv, const mx::array &down_weight,
      const mx::array &down_scale_inv, const mx::array &shared_gate_weight,
      const mx::array &shared_gate_scale_inv,
      const mx::array &shared_up_weight,
      const mx::array &shared_up_scale_inv,
      const mx::array &shared_down_weight,
      const mx::array &shared_down_scale_inv);
  mx::array execute_fused_down_reduce(
      const mx::array &hidden, const mx::array &expert_ids,
      const mx::array &scores, const mx::array &gate_up_weight,
      const mx::array &gate_up_scale_inv, const mx::array &down_weight,
      const mx::array &down_scale_inv, const mx::array &shared_gate_weight,
      const mx::array &shared_gate_scale_inv,
      const mx::array &shared_up_weight,
      const mx::array &shared_up_scale_inv,
      const mx::array &shared_down_weight,
      const mx::array &shared_down_scale_inv);
  mx::array execute_indirect(
      const mx::array &hidden, const mx::array &expert_ids,
      const mx::array &scores, const mx::array &gate_up_weight,
      const mx::array &gate_up_scale_inv, const mx::array &down_weight,
      const mx::array &down_scale_inv, const mx::array &shared_gate_weight,
      const mx::array &shared_gate_scale_inv,
      const mx::array &shared_up_weight,
      const mx::array &shared_up_scale_inv,
      const mx::array &shared_down_weight,
      const mx::array &shared_down_scale_inv);
  mx::array execute_indirect_hc(
      const mx::array &hidden, const mx::array &expert_ids,
      const mx::array &scores, const mx::array &gate_up_weight,
      const mx::array &gate_up_scale_inv, const mx::array &down_weight,
      const mx::array &down_scale_inv, const mx::array &shared_gate_weight,
      const mx::array &shared_gate_scale_inv,
      const mx::array &shared_up_weight,
      const mx::array &shared_up_scale_inv,
      const mx::array &shared_down_weight,
      const mx::array &shared_down_scale_inv, const mx::array &residual,
      const mx::array &post, const mx::array &comb);

  int query_rows() const { return 256; }
  int hidden_size() const { return 4096; }
  uint64_t execution_count() const { return execution_count_; }
  uint64_t dynamic_allocation_count() const { return 0; }
  uint64_t graph_node_count() const { return 0; }
  uint64_t shape_discovery_count() const { return 0; }
  uint64_t host_synchronization_count() const { return 0; }
  uint64_t returned_intermediate_tensor_bytes() const { return 0; }
  uint64_t fused_materialized_routed_down_bytes() const {
    return routed_plan_.fused_materialized_routed_down_bytes();
  }
  uint64_t scratch_bytes() const;
  std::vector<uint64_t> buffer_identities() const;

private:
  static constexpr int kQueryRows = 256;
  static constexpr int kHiddenSize = 4096;
  mx::Stream stream_;
  NativePrefillMoEGateUpPlan routed_plan_;
  NativePrefillSharedExpertPlan shared_plan_;
  mx::array output_;
  mx::array hc_output_;
  MTL::ComputePipelineState *add_pipeline_{nullptr};
  MTL::ComputePipelineState *hc_expand_pipeline_{nullptr};
  std::vector<uint64_t> initial_buffer_identities_;
  uint64_t execution_count_{0};

  mx::array finish(const mx::array &routed, const mx::array &shared);
  mx::array finish_hc(const mx::array &residual, const mx::array &post,
                      const mx::array &comb);
  void validate_hc_input(const mx::array &value, const char *name,
                         mx::Dtype dtype, size_t elements) const;
};

} // namespace glm53::native_execution
