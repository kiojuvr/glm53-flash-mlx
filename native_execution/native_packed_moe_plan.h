#pragma once

#include <cstdint>
#include <vector>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/stream.h"

namespace glm53::native_execution {

namespace mx = mlx::core;

// Decode-only exact packed MoE plan.  Routing remains the authoritative MLX
// implementation; selected routed experts and the shared expert execute in a
// fixed-address scratch arena under one native command topology.
class NativePackedMoEDecodePlan {
public:
  NativePackedMoEDecodePlan(int hidden_size, int intermediate_size,
                            int shared_intermediate_size, int expert_count,
                            int swiglu_limit);

  mx::array execute(
      const mx::array &x, const mx::array &expert_ids,
      const mx::array &scores, const mx::array &gate_up_weight,
      const mx::array &gate_up_scale_inv, const mx::array &down_weight,
      const mx::array &down_scale_inv, const mx::array &shared_gate_weight,
      const mx::array &shared_gate_scale_inv,
      const mx::array &shared_up_weight,
      const mx::array &shared_up_scale_inv,
      const mx::array &shared_down_weight,
      const mx::array &shared_down_scale_inv);

  int hidden_size() const { return hidden_size_; }
  int intermediate_size() const { return intermediate_size_; }
  int shared_intermediate_size() const { return shared_intermediate_size_; }
  int expert_count() const { return expert_count_; }
  uint64_t execution_count() const { return execution_count_; }
  uint64_t dynamic_allocation_count() const { return 0; }
  uint64_t graph_node_count() const { return 0; }
  uint64_t shape_discovery_count() const { return 0; }
  uint64_t host_synchronization_count() const { return 0; }
  uint64_t returned_intermediate_tensor_bytes() const { return 0; }
  uint64_t scratch_bytes() const;
  std::vector<uint64_t> buffer_identities() const;
  mx::array debug_routed_hidden() const { return routed_hidden_; }
  mx::array debug_routed_down() const { return routed_down_; }
  mx::array debug_routed_output() const { return routed_output_; }
  mx::array debug_shared_hidden() const { return shared_hidden_; }
  mx::array debug_shared_down() const { return shared_down_; }

private:
  static constexpr int kTopK = 8;
  static constexpr int kBlockSize = 128;
  static constexpr int kThreads = 256;

  int hidden_size_;
  int intermediate_size_;
  int shared_intermediate_size_;
  int expert_count_;
  int swiglu_limit_;
  int hidden_scale_rows_;
  int intermediate_scale_rows_;
  int shared_scale_rows_;
  mx::Stream stream_;
  mx::array routed_hidden_;
  mx::array routed_down_;
  mx::array routed_output_;
  mx::array shared_hidden_;
  mx::array shared_down_;
  mx::array output_;
  std::vector<uint64_t> initial_buffer_identities_;
  uint64_t execution_count_{0};

  void validate_input(const mx::array &array, const char *name,
                      mx::Dtype dtype, size_t elements) const;
};

} // namespace glm53::native_execution
