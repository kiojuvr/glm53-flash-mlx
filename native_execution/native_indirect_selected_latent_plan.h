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

// Probe-only substrate that keeps the Q256 union count on device, converts it
// to Metal indirect-dispatch geometry, and gathers the union latent rows into
// a fixed arena.  This proves the execution ABI needed by the later tiled K/V
// projection plan without allocating full-history projected K/V tensors.
class NativeIndirectSelectedLatentPlan {
public:
  explicit NativeIndirectSelectedLatentPlan(int physical_k);

  std::vector<mx::array> execute(
      const mx::array &selected_indices, const mx::array &selected_valid,
      const mx::array &latent);

  int physical_k() const { return physical_k_; }
  uint64_t execution_count() const { return execution_count_; }
  uint64_t dynamic_allocation_count() const { return 0; }
  uint64_t graph_node_count() const { return 0; }
  uint64_t shape_discovery_count() const { return 0; }
  uint64_t host_synchronization_count() const { return 0; }
  uint64_t returned_intermediate_tensor_bytes() const { return 0; }
  uint64_t scratch_bytes() const;
  std::vector<uint64_t> buffer_identities() const;
  mx::array union_count() const { return union_plan_.union_count(); }
  mx::array union_indices() const { return union_plan_.union_indices(); }
  mx::array union_latent() const { return union_latent_; }
  mx::array debug_indirect_arguments() const { return indirect_arguments_; }

private:
  static constexpr int kLatentDim = 512;
  static constexpr int kThreadsPerGroup = 256;
  static constexpr int kRowsPerGroup = 8;

  int physical_k_;
  mx::Stream stream_;
  NativeSelectedUnionPlan union_plan_;
  mx::array indirect_arguments_;
  mx::array union_latent_;
  MTL::ComputePipelineState *arguments_pipeline_{nullptr};
  MTL::ComputePipelineState *gather_pipeline_{nullptr};
  std::vector<uint64_t> initial_buffer_identities_;
  uint64_t execution_count_{0};

  void validate_latent(const mx::array &latent) const;
};

} // namespace glm53::native_execution
