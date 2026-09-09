#pragma once

#include <cstdint>
#include <vector>

#include <Metal/Metal.hpp>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/stream.h"

namespace glm53::native_execution {

namespace mx = mlx::core;

class NativePrefillSharedExpertPlan {
public:
  NativePrefillSharedExpertPlan();

  mx::array execute(
      const mx::array &hidden, const mx::array &gate_weight,
      const mx::array &gate_scale_inv, const mx::array &up_weight,
      const mx::array &up_scale_inv, const mx::array &down_weight,
      const mx::array &down_scale_inv);

  int query_rows() const { return kQueryRows; }
  int hidden_size() const { return kHiddenSize; }
  int intermediate_size() const { return kIntermediateSize; }
  int tile_rows() const { return kTileRows; }
  uint64_t execution_count() const { return execution_count_; }
  uint64_t dynamic_allocation_count() const { return 0; }
  uint64_t graph_node_count() const { return 0; }
  uint64_t shape_discovery_count() const { return 0; }
  uint64_t host_synchronization_count() const { return 0; }
  uint64_t returned_intermediate_tensor_bytes() const { return 0; }
  uint64_t scratch_bytes() const;
  std::vector<uint64_t> buffer_identities() const;

private:
  static constexpr int kQueryRows = 256;
  static constexpr int kHiddenSize = 4096;
  static constexpr int kIntermediateSize = 2048;
  static constexpr int kTileRows = 8;

  mx::Stream stream_;
  mx::array activated_;
  mx::array output_;
  MTL::ComputePipelineState *gate_up_pipeline_{nullptr};
  MTL::ComputePipelineState *down_pipeline_{nullptr};
  std::vector<uint64_t> initial_buffer_identities_;
  uint64_t execution_count_{0};

  void validate_input(const mx::array &array, const char *name,
                      mx::Dtype dtype, size_t elements) const;
};

} // namespace glm53::native_execution
