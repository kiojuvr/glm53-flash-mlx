#pragma once

#include <cstdint>
#include <vector>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/stream.h"

#include "native_dsa_score_plan.h"

namespace glm53::native_execution {

namespace mx = mlx::core;

// Probe-only decode plan which advances the compact raw19 rollback window,
// publishes the one affected kpool=4 row, and immediately executes the
// accepted Tier-1 pooled-score/selection plan on the updated state.
class NativeIndexPoolUpdateSelectionPlan {
public:
  NativeIndexPoolUpdateSelectionPlan(int physical_pool_rows,
                                     float softmax_scale);

  std::vector<mx::array> execute(
      const mx::array &key, const mx::array &gate,
      const mx::array &current_valid, const mx::array &query,
      const mx::array &mixture_weights, mx::array &pool_keys,
      mx::array &pool_indices, mx::array &pool_valid,
      const mx::array &raw_keys, const mx::array &raw_gates,
      const mx::array &raw_valid, const mx::array &raw_positions,
      const mx::array &compress_ape, int previous_total_tokens);

  int physical_pool_rows() const { return physical_pool_rows_; }
  int selected_width() const { return kSelectedWidth; }
  uint64_t execution_count() const { return execution_count_; }
  uint64_t dynamic_allocation_count() const { return 0; }
  uint64_t graph_node_count() const { return 0; }
  uint64_t shape_discovery_count() const { return 0; }
  uint64_t host_synchronization_count() const { return 0; }
  uint64_t returned_intermediate_tensor_bytes() const { return 0; }
  uint64_t scratch_bytes() const;
  std::vector<uint64_t> buffer_identities() const;

  mx::array debug_pool_logits() const { return pool_logits_; }
  mx::array debug_pool_probabilities() const { return pool_probabilities_; }
  mx::array current_raw_keys() const { return *current_raw_keys_; }
  mx::array current_raw_gates() const { return *current_raw_gates_; }
  mx::array current_raw_valid() const { return *current_raw_valid_; }
  mx::array current_raw_positions() const { return *current_raw_positions_; }

private:
  static constexpr int kHeadDim = 128;
  static constexpr int kIndexKPool = 4;
  static constexpr int kRawWindow = 19;
  static constexpr int kSelectedWidth = 2051;

  int physical_pool_rows_;
  mx::Stream stream_;
  NativeDSAScoreSelectionPlan score_plan_;
  mx::array raw_keys_a_;
  mx::array raw_keys_b_;
  mx::array raw_gates_a_;
  mx::array raw_gates_b_;
  mx::array raw_valid_a_;
  mx::array raw_valid_b_;
  mx::array raw_positions_a_;
  mx::array raw_positions_b_;
  mx::array pool_logits_;
  mx::array pool_probabilities_;
  mx::array *current_raw_keys_;
  mx::array *current_raw_gates_;
  mx::array *current_raw_valid_;
  mx::array *current_raw_positions_;
  std::vector<uint64_t> initial_buffer_identities_;
  uint64_t execution_count_{0};

  void validate_input(const mx::array &array, const char *name,
                      mx::Dtype dtype, size_t elements) const;
};

} // namespace glm53::native_execution
