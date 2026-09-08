#pragma once

#include <cstdint>
#include <vector>

#include <Metal/Metal.hpp>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/stream.h"

namespace glm53::native_execution {

namespace mx = mlx::core;

// Probe-only first executable slice of the Q256 shared physical value pass.
// One BM64 query block consumes one physical BK16 tile.  Probabilities are
// scattered into physical order, V is projected once per head, and the exact
// Steel BM64 reduction consumes both plan-owned arenas without materializing
// query-local selected V.
class NativeSharedPhysicalValueTilePlan {
public:
  explicit NativeSharedPhysicalValueTilePlan(int physical_k,
                                             int tile_rows = 0,
                                             int query_rows = 64);

  mx::array execute(const mx::array &selected_probabilities,
                    const mx::array &selected_indices,
                    const mx::array &selected_valid,
                    const mx::array &latent,
                    const mx::array &value_weight);

  int physical_k() const { return physical_k_; }
  int tile_rows() const { return tile_rows_; }
  int tile_count() const { return tile_count_; }
  int query_rows() const { return query_rows_; }
  int query_blocks() const { return query_rows_ / kQueryBlockRows; }
  int selected_width() const { return kSelectedWidth; }
  uint64_t execution_count() const { return execution_count_; }
  uint64_t dynamic_allocation_count() const { return 0; }
  uint64_t graph_node_count() const { return 0; }
  uint64_t shape_discovery_count() const { return 0; }
  uint64_t host_synchronization_count() const { return 0; }
  uint64_t returned_intermediate_tensor_bytes() const { return 0; }
  uint64_t materialized_query_local_selected_value_bytes() const { return 0; }
  uint64_t scratch_bytes() const;
  std::vector<uint64_t> buffer_identities() const;
  mx::array debug_physical_probabilities() const {
    return physical_probabilities_;
  }
  mx::array debug_projected_values() const { return projected_values_; }

private:
  static constexpr int kQueryBlockRows = 64;
  static constexpr int kHeads = 64;
  static constexpr int kLatentDim = 512;
  static constexpr int kValueDim = 128;
  static constexpr int kSelectedWidth = 2051;
  static constexpr int kBK = 16;

  int physical_k_;
  int tile_rows_;
  int tile_count_;
  int query_rows_;
  mx::Stream stream_;
  mx::array physical_probabilities_;
  mx::array projected_values_;
  mx::array output_;
  mx::array fp32_accumulator_;
  MTL::ComputePipelineState *clear_pipeline_{nullptr};
  MTL::ComputePipelineState *scatter_pipeline_{nullptr};
  MTL::ComputePipelineState *projection_pipeline_{nullptr};
  MTL::ComputePipelineState *av_pipeline_{nullptr};
  std::vector<uint64_t> initial_buffer_identities_;
  uint64_t execution_count_{0};

  void validate_input(const mx::array &array, const char *name,
                      mx::Dtype dtype, size_t elements) const;
};

} // namespace glm53::native_execution
