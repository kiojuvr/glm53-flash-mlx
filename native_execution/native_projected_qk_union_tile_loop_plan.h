#pragma once

#include <cstdint>
#include <vector>

#include <Metal/Metal.hpp>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/stream.h"

#include "native_selected_union_plan.h"

namespace glm53::native_execution {

namespace mx = mlx::core;

// Probe-only fixed-topology multi-tile Q256 union plan.  The host encodes the
// maximum number of tiles from physical capacity once per execution.  The
// device-resident union count decides which rows in those tiles are active;
// no count readback, shape discovery, or execute-time allocation is allowed.
class NativeProjectedQKUnionTileLoopPlan {
public:
  explicit NativeProjectedQKUnionTileLoopPlan(int physical_k);

  mx::array execute(
      const mx::array &selected_indices, const mx::array &selected_valid,
      const mx::array &latent, const mx::array &key_weight,
      const mx::array &attention_query, float attention_scale);

  int physical_k() const { return physical_k_; }
  int tile_rows() const { return kTileRows; }
  int tile_count() const { return tile_count_; }
  int query_rows() const { return kAttentionQueryRows; }
  int selected_width() const { return kSelectedWidth; }
  uint64_t execution_count() const { return execution_count_; }
  uint64_t dynamic_allocation_count() const { return 0; }
  uint64_t graph_node_count() const { return 0; }
  uint64_t shape_discovery_count() const { return 0; }
  uint64_t host_synchronization_count() const { return 0; }
  uint64_t returned_intermediate_tensor_bytes() const { return 0; }
  uint64_t materialized_selected_key_bytes() const { return 0; }
  uint64_t scratch_bytes() const;
  std::vector<uint64_t> buffer_identities() const;
  mx::array union_indices() const { return union_plan_.union_indices(); }
  mx::array union_count() const { return union_plan_.union_count(); }
  mx::array query_union_slots() const {
    return union_plan_.query_union_slots();
  }
  mx::array debug_union_latent_tile() const { return union_latent_tile_; }
  mx::array debug_projected_union_key_tile() const {
    return projected_union_key_tile_;
  }
  mx::array debug_scaled_queries() const { return scaled_queries_; }

private:
  static constexpr int kTileRows = 4096;
  static constexpr int kSelectionQueryRows = 256;
  static constexpr int kAttentionQueryRows = 4;
  static constexpr int kHeads = 64;
  static constexpr int kLatentDim = 512;
  static constexpr int kSelectedWidth = 2051;

  int physical_k_;
  int tile_count_;
  mx::Stream stream_;
  NativeSelectedUnionPlan union_plan_;
  mx::array union_latent_tile_;
  mx::array projected_union_key_tile_;
  mx::array scaled_queries_;
  mx::array attention_scores_;
  MTL::ComputePipelineState *clear_scores_pipeline_{nullptr};
  MTL::ComputePipelineState *gather_tile_pipeline_{nullptr};
  MTL::ComputePipelineState *projection_pipeline_{nullptr};
  MTL::ComputePipelineState *query_scale_pipeline_{nullptr};
  MTL::ComputePipelineState *projected_qk_pipeline_{nullptr};
  std::vector<uint64_t> initial_buffer_identities_;
  uint64_t execution_count_{0};

  void validate_input(const mx::array &array, const char *name,
                      mx::Dtype dtype, size_t elements) const;
};

} // namespace glm53::native_execution
