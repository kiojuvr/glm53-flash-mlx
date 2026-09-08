#pragma once

#include <cstdint>
#include <vector>

#include <Metal/Metal.hpp>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/stream.h"

namespace glm53::native_execution {

namespace mx = mlx::core;

// Probe-only exact Q4 operator for one union tile. K is projected once per
// union row/head and consumed through query-local union slots by a row-gather
// form of the captured exact GEMV QK reduction in the same encoder scope.
class NativeProjectedQKUnionTilePlan {
public:
  NativeProjectedQKUnionTilePlan();

  mx::array execute(
      const mx::array &union_latent, const mx::array &key_weight,
      const mx::array &attention_query, const mx::array &query_union_slots,
      const mx::array &selected_valid, float attention_scale);

  int union_tile_rows() const { return kUnionTileRows; }
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
  mx::array debug_projected_union_key() const { return projected_union_key_; }
  mx::array debug_scaled_query() const { return scaled_query_; }

private:
  static constexpr int kUnionTileRows = 4096;
  static constexpr int kQueryRows = 4;
  static constexpr int kHeads = 64;
  static constexpr int kLatentDim = 512;
  static constexpr int kSelectedWidth = 2051;

  mx::Stream stream_;
  mx::array projected_union_key_;
  mx::array scaled_query_;
  mx::array attention_scores_;
  MTL::ComputePipelineState *projection_pipeline_{nullptr};
  MTL::ComputePipelineState *query_scale_pipeline_{nullptr};
  MTL::ComputePipelineState *projected_qk_pipeline_{nullptr};
  std::vector<uint64_t> initial_buffer_identities_;
  uint64_t execution_count_{0};

  void validate_input(const mx::array &array, const char *name,
                      mx::Dtype dtype, size_t elements) const;
};

} // namespace glm53::native_execution
