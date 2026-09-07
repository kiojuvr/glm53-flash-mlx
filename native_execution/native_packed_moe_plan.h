#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <Metal/Metal.hpp>

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

  void bind_weights(
      const mx::array &gate_up_weight,
      const mx::array &gate_up_scale_inv, const mx::array &down_weight,
      const mx::array &down_scale_inv, const mx::array &shared_gate_weight,
      const mx::array &shared_gate_scale_inv,
      const mx::array &shared_up_weight,
      const mx::array &shared_up_scale_inv,
      const mx::array &shared_down_weight,
      const mx::array &shared_down_scale_inv);
  mx::array execute_bound(const mx::array &x, const mx::array &expert_ids,
                          const mx::array &scores);

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
  uint64_t static_input_validation_count() const {
    return static_input_validation_count_;
  }
  uint64_t dynamic_input_validation_count() const {
    return dynamic_input_validation_count_;
  }
  uint64_t pipeline_lookup_count() const { return pipeline_lookup_count_; }
  bool weights_bound() const { return bound_weights_.size() == 10; }
  bool bound_weight_identities_stable() const;
  uint64_t scratch_bytes() const;
  std::vector<uint64_t> buffer_identities() const;
  std::vector<uint64_t> bound_weight_identities() const;
  bool uses_shape_specialized_routed_gate_up() const {
    return hidden_size_ == 4096 && intermediate_size_ == 2048 &&
        intermediate_scale_rows_ == 16 && hidden_scale_rows_ == 32 &&
        swiglu_limit_ == 10;
  }
  bool uses_fast_bf16_routed_sigmoid() const {
    return uses_shape_specialized_routed_gate_up();
  }
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
  std::vector<mx::array> bound_weights_;
  std::vector<uint64_t> initial_bound_weight_identities_;
  std::array<MTL::ComputePipelineState *, 6> bound_pipelines_{};
  uint64_t execution_count_{0};
  uint64_t static_input_validation_count_{0};
  uint64_t dynamic_input_validation_count_{0};
  uint64_t pipeline_lookup_count_{0};

  struct WeightRefs {
    const mx::array &gate_up_weight;
    const mx::array &gate_up_scale_inv;
    const mx::array &down_weight;
    const mx::array &down_scale_inv;
    const mx::array &shared_gate_weight;
    const mx::array &shared_gate_scale_inv;
    const mx::array &shared_up_weight;
    const mx::array &shared_up_scale_inv;
    const mx::array &shared_down_weight;
    const mx::array &shared_down_scale_inv;
  };

  void validate_input(const mx::array &array, const char *name,
                      mx::Dtype dtype, size_t elements) const;
  void validate_static_weights(const WeightRefs &weights);
  std::array<MTL::ComputePipelineState *, 6> resolve_pipelines();
  mx::array encode(const mx::array &x, const mx::array &expert_ids,
                   const mx::array &scores, const WeightRefs &weights,
                   const std::array<MTL::ComputePipelineState *, 6> &pipelines);
  bool scratch_buffer_identities_stable() const;
};

// One-shot diagnostic for the GLM-5.3 routed gate/up/SwiGLU boundary.  It is
// intentionally separate from NativePackedMoEDecodePlan so instrumentation
// never changes the rejected plan's command topology or fixed scratch arena.
class NativePackedMoERoutedDiagnostic {
public:
  explicit NativePackedMoERoutedDiagnostic(int expert_count);

  mx::array execute(const mx::array &x, const mx::array &expert_ids,
                    const mx::array &gate_up_weight,
                    const mx::array &gate_up_scale_inv);

  mx::array gate() const { return gate_; }
  mx::array up() const { return up_; }
  mx::array sigmoid() const { return sigmoid_; }
  mx::array silu() const { return silu_; }
  mx::array hidden() const { return hidden_; }
  uint64_t execution_count() const { return execution_count_; }
  uint64_t scratch_bytes() const;
  std::vector<uint64_t> buffer_identities() const;

private:
  static constexpr int kHiddenSize = 4096;
  static constexpr int kIntermediateSize = 2048;
  static constexpr int kTopK = 8;
  static constexpr int kBlockSize = 128;
  int expert_count_;
  mx::Stream stream_;
  mx::array gate_;
  mx::array up_;
  mx::array sigmoid_;
  mx::array silu_;
  mx::array hidden_;
  std::vector<uint64_t> initial_buffer_identities_;
  uint64_t execution_count_{0};
};

class NativeRoutedSigmoidFormulaSweep {
public:
  explicit NativeRoutedSigmoidFormulaSweep(int elements);
  mx::array execute(const mx::array &gate);

  mx::array standard_bf16() const { return standard_bf16_; }
  mx::array precise_bf16() const { return precise_bf16_; }
  mx::array standard_f32() const { return standard_f32_; }
  mx::array precise_f32() const { return precise_f32_; }
  mx::array fast_bf16() const { return fast_bf16_; }
  mx::array fast_f32() const { return fast_f32_; }
  int elements() const { return elements_; }

private:
  int elements_;
  mx::Stream stream_;
  mx::array standard_bf16_;
  mx::array precise_bf16_;
  mx::array standard_f32_;
  mx::array precise_f32_;
  mx::array fast_bf16_;
  mx::array fast_f32_;
};

} // namespace glm53::native_execution
