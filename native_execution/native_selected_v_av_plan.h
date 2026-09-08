#pragma once

#include <cstdint>
#include <vector>

#include <Metal/Metal.hpp>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/stream.h"

namespace glm53::native_execution {

namespace mx = mlx::core;

// Probe-only Q4 native island. It projects selected latent rows to BF16 V and
// immediately consumes that plan-owned buffer through the exact virtual-BK16
// AV reduction. Projected V never crosses back into the MLX graph.
class NativeSelectedVProjectionAVPlan {
public:
  explicit NativeSelectedVProjectionAVPlan(int physical_k,
                                           bool attention_enabled = false);

  mx::array execute(const mx::array &selected_probabilities,
                    const mx::array &selected_latent,
                    const mx::array &value_weight,
                    const mx::array &selected_indices,
                    const mx::array &selected_valid);
  mx::array execute_attention(const mx::array &selected_latent,
                              const mx::array &key_weight,
                              const mx::array &value_weight,
                              const mx::array &attention_query,
                              const mx::array &selected_indices,
                              const mx::array &selected_valid,
                              float attention_scale);

  int physical_k() const { return physical_k_; }
  bool attention_enabled() const { return attention_enabled_; }
  int packed_k() const { return packed_k_; }
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
  mx::array debug_projected_key() const { return projected_key_; }
  mx::array debug_projected_value() const { return projected_value_; }
  mx::array debug_scaled_query() const { return scaled_query_; }
  mx::array debug_attention_scores() const { return attention_scores_; }
  mx::array debug_attention_probabilities() const {
    return attention_probabilities_;
  }
  mx::array debug_lane_to_selected() const { return lane_to_selected_; }

private:
  static constexpr int kQueryRows = 4;
  static constexpr int kHeads = 64;
  static constexpr int kLatentDim = 512;
  static constexpr int kValueDim = 128;
  static constexpr int kSelectedWidth = 2051;
  static constexpr int kBK = 16;
  static constexpr int kMaximumPackedK = kSelectedWidth * kBK;

  int physical_k_;
  int packed_k_;
  bool attention_enabled_;
  mx::Stream stream_;
  mx::array projected_key_;
  mx::array projected_value_;
  mx::array scaled_query_;
  mx::array attention_scores_;
  mx::array attention_probabilities_;
  mx::array lane_to_selected_;
  mx::array output_;
  MTL::ComputePipelineState *key_projection_pipeline_{nullptr};
  MTL::ComputePipelineState *projection_pipeline_{nullptr};
  MTL::ComputePipelineState *query_scale_pipeline_{nullptr};
  MTL::ComputePipelineState *qk_pipeline_{nullptr};
  MTL::ComputePipelineState *mask_pipeline_{nullptr};
  MTL::ComputePipelineState *softmax_pipeline_{nullptr};
  MTL::ComputePipelineState *map_pipeline_{nullptr};
  MTL::ComputePipelineState *av_pipeline_{nullptr};
  std::vector<uint64_t> initial_buffer_identities_;
  uint64_t execution_count_{0};

  void validate_input(const mx::array &array, const char *name,
                      mx::Dtype dtype, size_t elements) const;
};

} // namespace glm53::native_execution
