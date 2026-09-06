#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/stream.h"

#include "native_dsa_score_plan.h"

namespace glm53::native_execution {

namespace mx = mlx::core;

// Tier-2 decode-only plan.  Tier-1 pooled score/selection/expansion and the
// exact MLX 0.32.2 D512 sparse-attention fallback share one command encoder
// and fixed-address arena.  Only the D512 attention output escapes.
class NativeDSASparseAttentionPlan {
public:
  NativeDSASparseAttentionPlan(int physical_pool_rows, int physical_kv_rows,
                               float indexer_softmax_scale,
                               float attention_scale);

  mx::array execute(const mx::array &index_query,
                    const mx::array &mixture_weights,
                    const mx::array &pool_keys, const mx::array &pool_indices,
                    const mx::array &pool_valid, const mx::array &raw_positions,
                    const mx::array &raw_valid, const mx::array &current_valid,
                    const mx::array &attention_query, const mx::array &latent,
                    int logical_pool_rows, int kv_len, int active_tail_count);

  // Diagnostic-only entry points used to attribute the Tier-2 operator
  // regression.  They reuse the plan-owned arena and exact production-probe
  // pipelines, but deliberately expose the durable boundary between input
  // preparation and the D512 attention math.  The main execute() path never
  // calls these methods and still returns only attention_output_.
  std::vector<mx::array>
  debug_prepare_inputs(const mx::array &selected_indices,
                       const mx::array &selected_valid,
                       const mx::array &attention_query,
                       const mx::array &latent, int kv_len);
  mx::array debug_attention_math(const mx::array &scaled_query,
                                 const mx::array &gathered_latent,
                                 const mx::array &selected_valid);

  int physical_pool_rows() const { return score_plan_.physical_pool_rows(); }
  int physical_kv_rows() const { return physical_kv_rows_; }
  int selected_width() const { return kSelectedWidth; }
  uint64_t execution_count() const { return execution_count_; }
  uint64_t dynamic_allocation_count() const { return 0; }
  uint64_t graph_node_count() const { return 0; }
  uint64_t shape_discovery_count() const { return 0; }
  uint64_t host_synchronization_count() const { return 0; }
  uint64_t returned_intermediate_tensor_bytes() const { return 0; }
  uint64_t scratch_bytes() const;
  mx::array debug_selected_indices() const;
  mx::array debug_selected_valid() const;
  mx::array debug_gathered_latent() const { return gathered_latent_; }
  mx::array debug_attention_scores() const { return attention_scores_; }
  std::vector<uint64_t> buffer_identities() const;

private:
  static constexpr int kAttentionHeads = 64;
  static constexpr int kAttentionDim = 512;
  static constexpr int kSelectedWidth = 2051;
  static constexpr int kSplitKPartitions = 4;

  int physical_kv_rows_;
  float attention_scale_;
  mx::Stream stream_;
  NativeDSAScoreSelectionPlan score_plan_;
  mx::array scaled_query_;
  mx::array gathered_latent_;
  mx::array attention_scores_;
  mx::array splitk_accum_;
  mx::array attention_output_;
  std::vector<uint64_t> initial_buffer_identities_;
  uint64_t execution_count_{0};

  void validate_input(const mx::array &array, const char *name, mx::Dtype dtype,
                      size_t elements) const;
};

} // namespace glm53::native_execution
