#pragma once

#include <cstdint>
#include <vector>

#include "mlx/array.h"

#include "native_projected_qk_union_tile_loop_plan.h"
#include "native_shared_physical_value_tile_plan.h"

namespace glm53::native_execution {

namespace mx = mlx::core;

// Probe-only complete Q256 DSA prefill island. QK projection, precise softmax,
// physical V projection, and four exact BM64 AV blocks share one persistent
// native command topology. Only the final attention output is returned.
class NativeQ256DSAPrefillPlan {
public:
  explicit NativeQ256DSAPrefillPlan(int physical_k, int tile_rows = 65536,
                                    int value_dim = 128);

  mx::array execute(const mx::array &selected_indices,
                    const mx::array &selected_valid,
                    const mx::array &latent,
                    const mx::array &key_weight,
                    const mx::array &value_weight,
                    const mx::array &attention_query,
                    float attention_scale);

  int physical_k() const { return qk_plan_.physical_k(); }
  int tile_rows() const { return qk_plan_.tile_rows(); }
  int tile_count() const { return qk_plan_.tile_count(); }
  int query_rows() const { return 256; }
  int value_dim() const { return value_plan_.value_dim(); }
  uint64_t execution_count() const { return execution_count_; }
  uint64_t dynamic_allocation_count() const { return 0; }
  uint64_t graph_node_count() const { return 0; }
  uint64_t shape_discovery_count() const { return 0; }
  uint64_t host_synchronization_count() const { return 0; }
  uint64_t returned_intermediate_tensor_bytes() const { return 0; }
  uint64_t materialized_selected_key_bytes() const { return 0; }
  uint64_t materialized_query_local_selected_value_bytes() const { return 0; }
  uint64_t scratch_bytes() const {
    return qk_plan_.scratch_bytes() + value_plan_.scratch_bytes();
  }
  std::vector<uint64_t> buffer_identities() const;
  mx::array debug_probabilities() const {
    return qk_plan_.debug_attention_probabilities();
  }
  mx::array debug_scores() const { return qk_plan_.debug_attention_scores(); }
  mx::array debug_union_count() const { return qk_plan_.union_count(); }

private:
  NativeProjectedQKUnionTileLoopPlan qk_plan_;
  NativeSharedPhysicalValueTilePlan value_plan_;
  std::vector<uint64_t> initial_buffer_identities_;
  uint64_t execution_count_{0};
};

} // namespace glm53::native_execution
