#pragma once

#include <cstdint>
#include <vector>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/stream.h"

#include "native_prefill_dsa_output_plan.h"
#include "native_prefill_moe_plan.h"
#include "native_q256_dsa_prefill_plan.h"

namespace glm53::native_execution {

namespace mx = mlx::core;

// Feasibility plan for the two regions which account for more than 95% of
// the synchronized 320K Q256 prefill profile.  The layerwise entry points and
// execute_all encode identical kernels and barriers.  Only the latter keeps
// the 11 DSA and 42 MoE stage submissions behind one Python/native boundary.
//
// This is deliberately not a model implementation: its two returned arrays
// are diagnostic terminal anchors.  It answers whether call aggregation by
// itself is valuable before the input-dependent seams are moved into C++.
class NativePrefillDominantRegionPlan {
public:
  NativePrefillDominantRegionPlan(int physical_k, int tile_rows = 65536,
                                  int expert_count = 288);

  mx::array execute_dsa(
      const mx::array &selected_indices, const mx::array &selected_valid,
      const mx::array &latent, const mx::array &key_weight,
      const mx::array &value_weight, const mx::array &attention_query,
      float attention_scale, const mx::array &output_weight,
      const mx::array &output_scale_inv);

  mx::array execute_moe(
      const mx::array &hidden, const mx::array &expert_ids,
      const mx::array &scores, const mx::array &gate_up_weight,
      const mx::array &gate_up_scale_inv, const mx::array &down_weight,
      const mx::array &down_scale_inv, const mx::array &shared_gate_weight,
      const mx::array &shared_gate_scale_inv,
      const mx::array &shared_up_weight,
      const mx::array &shared_up_scale_inv,
      const mx::array &shared_down_weight,
      const mx::array &shared_down_scale_inv);

  std::vector<mx::array> execute_all(
      const mx::array &selected_indices, const mx::array &selected_valid,
      const mx::array &latent, const mx::array &key_weight,
      const mx::array &value_weight, const mx::array &attention_query,
      float attention_scale, const mx::array &output_weight,
      const mx::array &output_scale_inv, const mx::array &hidden,
      const mx::array &expert_ids, const mx::array &scores,
      const mx::array &gate_up_weight,
      const mx::array &gate_up_scale_inv, const mx::array &down_weight,
      const mx::array &down_scale_inv, const mx::array &shared_gate_weight,
      const mx::array &shared_gate_scale_inv,
      const mx::array &shared_up_weight,
      const mx::array &shared_up_scale_inv,
      const mx::array &shared_down_weight,
      const mx::array &shared_down_scale_inv);

  int dsa_layer_count() const { return kDSALayerCount; }
  int moe_layer_count() const { return kMoELayerCount; }
  int layerwise_native_calls() const {
    return kDSALayerCount + kMoELayerCount;
  }
  int composed_native_calls() const { return 1; }
  uint64_t layerwise_execution_count() const {
    return layerwise_execution_count_;
  }
  uint64_t composed_execution_count() const {
    return composed_execution_count_;
  }
  uint64_t dynamic_allocation_count() const { return 0; }
  uint64_t graph_node_count() const { return 0; }
  uint64_t shape_discovery_count() const { return 0; }
  uint64_t host_synchronization_count() const { return 0; }
  uint64_t returned_diagnostic_anchor_bytes() const;
  uint64_t scratch_bytes() const;
  bool buffer_identities_stable() const;
  std::vector<uint64_t> buffer_identities() const;

private:
  static constexpr int kDSALayerCount = 11;
  static constexpr int kMoELayerCount = 42;

  mx::Stream stream_;
  NativeQ256DSAPrefillPlan dsa_body_;
  NativePrefillDSAOutputPlan dsa_output_;
  NativePrefillMoEPlan moe_;
  std::vector<uint64_t> initial_buffer_identities_;
  uint64_t layerwise_execution_count_{0};
  uint64_t composed_execution_count_{0};

  mx::array encode_dsa(
      const mx::array &selected_indices, const mx::array &selected_valid,
      const mx::array &latent, const mx::array &key_weight,
      const mx::array &value_weight, const mx::array &attention_query,
      float attention_scale, const mx::array &output_weight,
      const mx::array &output_scale_inv);
  mx::array encode_moe(
      const mx::array &hidden, const mx::array &expert_ids,
      const mx::array &scores, const mx::array &gate_up_weight,
      const mx::array &gate_up_scale_inv, const mx::array &down_weight,
      const mx::array &down_scale_inv, const mx::array &shared_gate_weight,
      const mx::array &shared_gate_scale_inv,
      const mx::array &shared_up_weight,
      const mx::array &shared_up_scale_inv,
      const mx::array &shared_down_weight,
      const mx::array &shared_down_scale_inv);
  void barrier();
};

} // namespace glm53::native_execution
