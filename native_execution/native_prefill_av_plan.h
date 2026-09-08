#pragma once

#include <cstdint>
#include <vector>

#include <Metal/Metal.hpp>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/stream.h"

namespace glm53::native_execution {

namespace mx = mlx::core;

// Probe-only exact prefill AV plan. Selected tokens are repacked into their
// original physical BK16 lanes and wholly empty physical blocks are removed.
// A fixed maximum K avoids host readback while retaining Direct accumulation.
// The Steel-compatible kernel gathers selected V rows directly into its BK16
// threadgroup tile; it never materializes the virtual dense value matrix.
class NativeSparsePrefillAVPlan {
public:
  explicit NativeSparsePrefillAVPlan(int physical_k);

  mx::array execute(const mx::array &selected_probabilities,
                    const mx::array &selected_values,
                    const mx::array &selected_indices,
                    const mx::array &selected_valid);
  // Attribution-only entry point. execute() must have prepared the lane map;
  // this isolates AV execution from the map-construction dispatch.
  mx::array execute_prepared(const mx::array &selected_probabilities,
                             const mx::array &selected_values);

  int physical_k() const { return physical_k_; }
  int packed_k() const { return packed_k_; }
  int selected_width() const { return kSelectedWidth; }
  int bk() const { return kBK; }
  int head_tile() const { return kHeads; }
  uint64_t execution_count() const { return execution_count_; }
  uint64_t dynamic_allocation_count() const { return 0; }
  uint64_t graph_node_count() const { return 0; }
  uint64_t shape_discovery_count() const { return 0; }
  uint64_t host_synchronization_count() const { return 0; }
  uint64_t returned_intermediate_tensor_bytes() const { return 0; }
  uint64_t scratch_bytes() const;
  std::vector<uint64_t> buffer_identities() const;
  mx::array debug_lane_to_selected() const { return lane_to_selected_; }

private:
  static constexpr int kHeads = 64;
  static constexpr int kValueDim = 128;
  static constexpr int kSelectedWidth = 2051;
  static constexpr int kBK = 16;
  static constexpr int kMaximumPackedK = kSelectedWidth * kBK;

  int physical_k_;
  int packed_k_;
  mx::Stream stream_;
  mx::array lane_to_selected_;
  mx::array output_;
  MTL::ComputePipelineState *map_pipeline_{nullptr};
  MTL::ComputePipelineState *av_pipeline_{nullptr};
  std::vector<uint64_t> initial_buffer_identities_;
  uint64_t execution_count_{0};
  bool map_prepared_{false};

  void validate_input(const mx::array &array, const char *name,
                      mx::Dtype dtype, size_t elements) const;
  mx::array submit_av(const mx::array &selected_probabilities,
                      const mx::array &selected_values);
};

} // namespace glm53::native_execution
