#pragma once

#include <cstdint>
#include <vector>

#include <Metal/Metal.hpp>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/stream.h"

#include "native_dsa_score_plan.h"
#include "native_selected_v_av_plan.h"

namespace glm53::native_execution {

namespace mx = mlx::core;

// Q4 prefill execution island from pooled Indexer score through exact sparse
// attention. Selection, selected-latent gather, K/V projection, QK, precise
// softmax, and virtual-BK16 AV share the default Metal command encoder. Only
// the final attention output is returned to MLX.
class NativeSelectedKVAttentionSelectionPlan {
public:
  NativeSelectedKVAttentionSelectionPlan(
      int physical_pool_rows, int physical_kv_rows,
      float indexer_softmax_scale, float attention_scale);

  mx::array execute(
      const mx::array &index_query, const mx::array &mixture_weights,
      const mx::array &pool_keys, const mx::array &pool_indices,
      const mx::array &pool_valid, const mx::array &raw_positions,
      const mx::array &raw_valid, const mx::array &current_valid,
      const mx::array &latent, const mx::array &key_weight,
      const mx::array &value_weight, const mx::array &attention_query,
      int logical_pool_rows, int kv_len, int active_tail_count);

  int physical_pool_rows() const { return score_plan_.physical_pool_rows(); }
  int physical_kv_rows() const { return physical_kv_rows_; }
  int query_rows() const { return kQueryRows; }
  int selected_width() const { return kSelectedWidth; }
  uint64_t execution_count() const { return execution_count_; }
  uint64_t dynamic_allocation_count() const { return 0; }
  uint64_t graph_node_count() const { return 0; }
  uint64_t shape_discovery_count() const { return 0; }
  uint64_t host_synchronization_count() const { return 0; }
  uint64_t returned_intermediate_tensor_bytes() const { return 0; }
  uint64_t scratch_bytes() const;
  std::vector<uint64_t> buffer_identities() const;
  mx::array debug_selected_indices() const {
    return attention_indices_;
  }
  mx::array debug_selected_valid() const {
    return attention_valid_;
  }
  mx::array debug_score_order_indices() const {
    return score_plan_.debug_selected_indices();
  }
  mx::array debug_selected_latent() const { return selected_latent_; }

private:
  static constexpr int kQueryRows = 4;
  static constexpr int kSelectedWidth = 2051;
  static constexpr int kLatentDim = 512;

  int physical_kv_rows_;
  float attention_scale_;
  mx::Stream stream_;
  NativeDSAScoreSelectionPlan score_plan_;
  NativeSelectedVProjectionAVPlan attention_plan_;
  mx::array attention_indices_;
  mx::array attention_valid_;
  mx::array selected_latent_;
  MTL::ComputePipelineState *order_pipeline_{nullptr};
  MTL::ComputePipelineState *gather_pipeline_{nullptr};
  std::vector<uint64_t> initial_buffer_identities_;
  uint64_t execution_count_{0};

  void validate_latent(const mx::array &latent) const;
};

} // namespace glm53::native_execution
