#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <Metal/Metal.hpp>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/stream.h"

namespace glm53::native_execution {

namespace mx = mlx::core;

// Arithmetic-neutral proof of the model-wide submission topology. Both entry
// points encode the same 45 copies into the same ping-pong arena. execute_all
// crosses Python/native once; execute_layer exposes the old per-layer boundary
// as a controlled oracle. Exact layer arithmetic replaces each copy only after
// its KDA/DSA and dense/MoE coverage gate passes.
class NativePrefill45LayerSubmissionPlan {
public:
  NativePrefill45LayerSubmissionPlan();

  mx::array execute_layer(const mx::array &hidden, int layer_index);
  mx::array execute_all(const mx::array &hidden);

  int layer_count() const { return kLayerCount; }
  int query_rows() const { return kQueryRows; }
  int hidden_size() const { return kHiddenSize; }
  int kda_layer_count() const { return 34; }
  int dsa_layer_count() const { return 11; }
  int dense_ffn_layer_count() const { return 3; }
  int moe_layer_count() const { return 42; }
  uint64_t all_execution_count() const { return all_execution_count_; }
  uint64_t layer_execution_count() const { return layer_execution_count_; }
  uint64_t dynamic_allocation_count() const { return 0; }
  uint64_t graph_node_count() const { return 0; }
  uint64_t shape_discovery_count() const { return 0; }
  uint64_t host_synchronization_count() const { return 0; }
  uint64_t native_calls_per_all_execute() const { return 1; }
  uint64_t native_calls_per_layerwise_execute() const { return kLayerCount; }
  uint64_t returned_intermediate_tensor_bytes() const { return 0; }
  uint64_t arena_bytes() const;
  bool buffer_identities_stable() const;
  std::vector<uint64_t> buffer_identities() const;
  std::vector<std::string> attention_types() const;
  std::vector<std::string> ffn_types() const;

private:
  static constexpr int kLayerCount = 45;
  static constexpr int kQueryRows = 256;
  static constexpr int kHiddenSize = 4096;

  mx::Stream stream_;
  MTL::ComputePipelineState *copy_pipeline_{nullptr};
  mx::array hidden_ping_;
  mx::array hidden_pong_;
  std::vector<uint64_t> initial_buffer_identities_;
  uint64_t all_execution_count_{0};
  uint64_t layer_execution_count_{0};

  void validate_hidden(const mx::array &hidden) const;
  mx::array encode_layer(const mx::array &hidden, int layer_index);
};

} // namespace glm53::native_execution
