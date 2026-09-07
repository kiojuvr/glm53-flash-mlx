#include <metal_stdlib>

using namespace metal;

#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/softmax.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_fused.h"
#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_splitk.h"

constant uint kSelectedPools = 512;
constant uint kIndexKPool = 4;
constant uint kSelectedWidth = 2051;
constant uint kIndexerHeads = 32;
constant uint kIndexPoolHeadDim = 128;
constant uint kIndexPoolRawWindow = 19;
constant float kNativeE4M3[256] = {
    0.0f,0.001953125f,0.00390625f,0.005859375f,0.0078125f,0.009765625f,0.01171875f,0.013671875f,0.015625f,0.017578125f,0.01953125f,0.021484375f,0.0234375f,0.025390625f,0.02734375f,0.029296875f,
    0.03125f,0.03515625f,0.0390625f,0.04296875f,0.046875f,0.05078125f,0.0546875f,0.05859375f,0.0625f,0.0703125f,0.078125f,0.0859375f,0.09375f,0.1015625f,0.109375f,0.1171875f,
    0.125f,0.140625f,0.15625f,0.171875f,0.1875f,0.203125f,0.21875f,0.234375f,0.25f,0.28125f,0.3125f,0.34375f,0.375f,0.40625f,0.4375f,0.46875f,
    0.5f,0.5625f,0.625f,0.6875f,0.75f,0.8125f,0.875f,0.9375f,1.0f,1.125f,1.25f,1.375f,1.5f,1.625f,1.75f,1.875f,
    2.0f,2.25f,2.5f,2.75f,3.0f,3.25f,3.5f,3.75f,4.0f,4.5f,5.0f,5.5f,6.0f,6.5f,7.0f,7.5f,
    8.0f,9.0f,10.0f,11.0f,12.0f,13.0f,14.0f,15.0f,16.0f,18.0f,20.0f,22.0f,24.0f,26.0f,28.0f,30.0f,
    32.0f,36.0f,40.0f,44.0f,48.0f,52.0f,56.0f,60.0f,64.0f,72.0f,80.0f,88.0f,96.0f,104.0f,112.0f,120.0f,
    128.0f,144.0f,160.0f,176.0f,192.0f,208.0f,224.0f,240.0f,256.0f,288.0f,320.0f,352.0f,384.0f,416.0f,448.0f,480.0f,
    -0.0f,-0.001953125f,-0.00390625f,-0.005859375f,-0.0078125f,-0.009765625f,-0.01171875f,-0.013671875f,-0.015625f,-0.017578125f,-0.01953125f,-0.021484375f,-0.0234375f,-0.025390625f,-0.02734375f,-0.029296875f,
    -0.03125f,-0.03515625f,-0.0390625f,-0.04296875f,-0.046875f,-0.05078125f,-0.0546875f,-0.05859375f,-0.0625f,-0.0703125f,-0.078125f,-0.0859375f,-0.09375f,-0.1015625f,-0.109375f,-0.1171875f,
    -0.125f,-0.140625f,-0.15625f,-0.171875f,-0.1875f,-0.203125f,-0.21875f,-0.234375f,-0.25f,-0.28125f,-0.3125f,-0.34375f,-0.375f,-0.40625f,-0.4375f,-0.46875f,
    -0.5f,-0.5625f,-0.625f,-0.6875f,-0.75f,-0.8125f,-0.875f,-0.9375f,-1.0f,-1.125f,-1.25f,-1.375f,-1.5f,-1.625f,-1.75f,-1.875f,
    -2.0f,-2.25f,-2.5f,-2.75f,-3.0f,-3.25f,-3.5f,-3.75f,-4.0f,-4.5f,-5.0f,-5.5f,-6.0f,-6.5f,-7.0f,-7.5f,
    -8.0f,-9.0f,-10.0f,-11.0f,-12.0f,-13.0f,-14.0f,-15.0f,-16.0f,-18.0f,-20.0f,-22.0f,-24.0f,-26.0f,-28.0f,-30.0f,
    -32.0f,-36.0f,-40.0f,-44.0f,-48.0f,-52.0f,-56.0f,-60.0f,-64.0f,-72.0f,-80.0f,-88.0f,-96.0f,-104.0f,-112.0f,-120.0f,
    -128.0f,-144.0f,-160.0f,-176.0f,-192.0f,-208.0f,-224.0f,-240.0f,-256.0f,-288.0f,-320.0f,-352.0f,-384.0f,-416.0f,-448.0f,-480.0f};

// Canonical E4M3 decode used by the checkpoint.  Normal values are assembled
// directly as IEEE binary32 so this is bit-equivalent to the probe LUT without
// a transcendental or arithmetic decode in the inner projection loop.
inline float glm53_native_e4m3(uint8_t code) {
  return kNativeE4M3[uint(code)];
}

[[kernel]] void glm53_native_packed_selected8_gate_up_swiglu(
    device const bfloat16_t* x [[buffer(0)]],
    device const uint* expert_ids [[buffer(1)]],
    device const uint8_t* weight [[buffer(2)]],
    device const float* scale_inv [[buffer(3)]],
    device bfloat16_t* hidden [[buffer(4)]],
    constant const int& hidden_size [[buffer(5)]],
    constant const int& intermediate_size [[buffer(6)]],
    constant const int& intermediate_scale_rows [[buffer(7)]],
    constant const int& hidden_scale_rows [[buffer(8)]],
    constant const int& swiglu_limit [[buffer(9)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]) {
  uint selected = group_id / uint(intermediate_size);
  uint out_row = group_id % uint(intermediate_size);
  if (selected >= 8u) return;
  uint expert = expert_ids[selected];
  uint bank_out_features = 2u * uint(intermediate_size);
  uint bank_scale_rows = 2u * uint(intermediate_scale_rows);
  const device uint8_t* gate_wr = weight +
      (size_t(expert) * bank_out_features + out_row) * uint(hidden_size);
  const device uint8_t* up_wr = weight +
      (size_t(expert) * bank_out_features + uint(intermediate_size) + out_row) *
          uint(hidden_size);
  float gate_acc = 0.0f;
  float up_acc = 0.0f;
  for (uint k = tid; k < uint(hidden_size); k += 256u) {
    uint scale_col = k / 128u;
    size_t gate_scale_offset =
        (size_t(expert) * bank_scale_rows + out_row / 128u) *
            uint(hidden_scale_rows) +
        scale_col;
    size_t up_scale_offset =
        (size_t(expert) * bank_scale_rows + uint(intermediate_scale_rows) +
         out_row / 128u) *
            uint(hidden_scale_rows) +
        scale_col;
    gate_acc += float(x[k]) * glm53_native_e4m3(gate_wr[k]) *
        scale_inv[gate_scale_offset];
    up_acc += float(x[k]) * glm53_native_e4m3(up_wr[k]) *
        scale_inv[up_scale_offset];
  }
  gate_acc = simd_sum(gate_acc);
  up_acc = simd_sum(up_acc);
  threadgroup float gate_partial[8];
  threadgroup float up_partial[8];
  if (lane == 0) {
    gate_partial[simd_id] = gate_acc;
    up_partial[simd_id] = up_acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_id == 0) {
    float gate_total = lane < 8u ? gate_partial[lane] : 0.0f;
    float up_total = lane < 8u ? up_partial[lane] : 0.0f;
    gate_total = simd_sum(gate_total);
    up_total = simd_sum(up_total);
    if (lane == 0) {
      bfloat16_t gate_t = bfloat16_t(gate_total);
      bfloat16_t up_t = bfloat16_t(up_total);
      float limit = float(swiglu_limit);
      float gate_value = min(float(gate_t), limit);
      float up_value = clamp(float(up_t), -limit, limit);
      bfloat16_t gate_activation = bfloat16_t(gate_value);
      bfloat16_t up_activation = bfloat16_t(up_value);
      auto sigmoid_tail =
          1 / (1 + metal::exp(metal::abs(gate_activation)));
      bfloat16_t sigmoid_value = gate_activation < 0
          ? sigmoid_tail
          : 1 - sigmoid_tail;
      bfloat16_t silu_value = gate_activation * sigmoid_value;
      bfloat16_t activated = silu_value * up_activation;
      hidden[size_t(selected) * uint(intermediate_size) + out_row] =
          bfloat16_t(activated);
    }
  }
}

// The exact MLX oracle specializes these dimensions as Metal template
// constants.  Keep a GLM-5.3-specific AOT variant so the compiler sees the
// same loop bounds, scale geometry, and SwiGLU limit.  The generic kernel
// remains available for the artificial ABI fixtures and future geometries.
[[kernel]] void glm53_native_glm53_packed_selected8_gate_up_swiglu(
    device const bfloat16_t* x [[buffer(0)]],
    device const uint* expert_ids [[buffer(1)]],
    device const uint8_t* weight [[buffer(2)]],
    device const float* scale_inv [[buffer(3)]],
    device bfloat16_t* hidden [[buffer(4)]],
    constant const int& runtime_hidden_size [[buffer(5)]],
    constant const int& runtime_intermediate_size [[buffer(6)]],
    constant const int& runtime_intermediate_scale_rows [[buffer(7)]],
    constant const int& runtime_hidden_scale_rows [[buffer(8)]],
    constant const int& runtime_swiglu_limit [[buffer(9)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]) {
  constexpr uint kHiddenSize = 4096u;
  constexpr uint kIntermediateSize = 2048u;
  constexpr uint kIntermediateScaleRows = 16u;
  constexpr uint kHiddenScaleRows = 32u;
  constexpr uint kBankOutFeatures = 4096u;
  constexpr uint kBankScaleRows = 32u;
  constexpr uint kThreads = 256u;
  constexpr uint kBlockSize = 128u;
  constexpr uint kTopK = 8u;
  constexpr uint kNSimd = kThreads / 32u;
  constexpr float kSwiGLULimit = 10.0f;
  (void)runtime_hidden_size;
  (void)runtime_intermediate_size;
  (void)runtime_intermediate_scale_rows;
  (void)runtime_hidden_scale_rows;
  (void)runtime_swiglu_limit;

  uint selected = group_id / kIntermediateSize;
  uint out_row = group_id % kIntermediateSize;
  if (selected >= kTopK) return;
  uint expert = expert_ids[selected];
  const device uint8_t* gate_wr = weight +
      (size_t(expert) * kBankOutFeatures + out_row) * kHiddenSize;
  const device uint8_t* up_wr = weight +
      (size_t(expert) * kBankOutFeatures + kIntermediateSize + out_row) *
          kHiddenSize;
  float gate_acc = 0.0f;
  float up_acc = 0.0f;
  for (uint k = tid; k < kHiddenSize; k += kThreads) {
    uint scale_col = k / kBlockSize;
    size_t gate_scale_offset =
        (size_t(expert) * kBankScaleRows + out_row / kBlockSize) *
            kHiddenScaleRows +
        scale_col;
    size_t up_scale_offset =
        (size_t(expert) * kBankScaleRows + kIntermediateScaleRows +
         out_row / kBlockSize) *
            kHiddenScaleRows +
        scale_col;
    gate_acc += float(x[k]) * glm53_native_e4m3(gate_wr[k]) *
        scale_inv[gate_scale_offset];
    up_acc += float(x[k]) * glm53_native_e4m3(up_wr[k]) *
        scale_inv[up_scale_offset];
  }
  gate_acc = simd_sum(gate_acc);
  up_acc = simd_sum(up_acc);
  threadgroup float gate_partial[kNSimd];
  threadgroup float up_partial[kNSimd];
  if (lane == 0) {
    gate_partial[simd_id] = gate_acc;
    up_partial[simd_id] = up_acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_id == 0) {
    float gate_total = lane < kNSimd ? gate_partial[lane] : 0.0f;
    float up_total = lane < kNSimd ? up_partial[lane] : 0.0f;
    gate_total = simd_sum(gate_total);
    up_total = simd_sum(up_total);
    if (lane == 0) {
      bfloat16_t gate_t = bfloat16_t(gate_total);
      bfloat16_t up_t = bfloat16_t(up_total);
      float gate_value = min(float(gate_t), kSwiGLULimit);
      float up_value = clamp(float(up_t), -kSwiGLULimit, kSwiGLULimit);
      bfloat16_t gate_activation = bfloat16_t(gate_value);
      bfloat16_t up_activation = bfloat16_t(up_value);
      // MLX v0.32.2 eager sigmoid rounds through the precise exponential.
      // The ordinary AOT intrinsic differs by one BF16 value for the recorded
      // step-29/layer-41 routed activation.
      auto sigmoid_tail =
          1 / (1 + metal::precise::exp(metal::abs(gate_activation)));
      bfloat16_t sigmoid_value = gate_activation < 0
          ? sigmoid_tail
          : 1 - sigmoid_tail;
      bfloat16_t silu_value = gate_activation * sigmoid_value;
      bfloat16_t activated = silu_value * up_activation;
      hidden[size_t(selected) * kIntermediateSize + out_row] = activated;
    }
  }
}

[[kernel]] void
glm53_native_glm53_packed_selected8_gate_up_swiglu_diagnostic(
    device const bfloat16_t* x [[buffer(0)]],
    device const uint* expert_ids [[buffer(1)]],
    device const uint8_t* weight [[buffer(2)]],
    device const float* scale_inv [[buffer(3)]],
    device bfloat16_t* gate_output [[buffer(4)]],
    device bfloat16_t* up_output [[buffer(5)]],
    device bfloat16_t* sigmoid_output [[buffer(6)]],
    device bfloat16_t* silu_output [[buffer(7)]],
    device bfloat16_t* hidden [[buffer(8)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]) {
  constexpr uint kHiddenSize = 4096u;
  constexpr uint kIntermediateSize = 2048u;
  constexpr uint kIntermediateScaleRows = 16u;
  constexpr uint kHiddenScaleRows = 32u;
  constexpr uint kBankOutFeatures = 4096u;
  constexpr uint kBankScaleRows = 32u;
  constexpr uint kThreads = 256u;
  constexpr uint kBlockSize = 128u;
  constexpr uint kTopK = 8u;
  constexpr uint kNSimd = kThreads / 32u;
  constexpr float kSwiGLULimit = 10.0f;

  uint selected = group_id / kIntermediateSize;
  uint out_row = group_id % kIntermediateSize;
  if (selected >= kTopK) return;
  uint expert = expert_ids[selected];
  const device uint8_t* gate_wr = weight +
      (size_t(expert) * kBankOutFeatures + out_row) * kHiddenSize;
  const device uint8_t* up_wr = weight +
      (size_t(expert) * kBankOutFeatures + kIntermediateSize + out_row) *
          kHiddenSize;
  float gate_acc = 0.0f;
  float up_acc = 0.0f;
  for (uint k = tid; k < kHiddenSize; k += kThreads) {
    uint scale_col = k / kBlockSize;
    size_t gate_scale_offset =
        (size_t(expert) * kBankScaleRows + out_row / kBlockSize) *
            kHiddenScaleRows +
        scale_col;
    size_t up_scale_offset =
        (size_t(expert) * kBankScaleRows + kIntermediateScaleRows +
         out_row / kBlockSize) *
            kHiddenScaleRows +
        scale_col;
    gate_acc += float(x[k]) * glm53_native_e4m3(gate_wr[k]) *
        scale_inv[gate_scale_offset];
    up_acc += float(x[k]) * glm53_native_e4m3(up_wr[k]) *
        scale_inv[up_scale_offset];
  }
  gate_acc = simd_sum(gate_acc);
  up_acc = simd_sum(up_acc);
  threadgroup float gate_partial[kNSimd];
  threadgroup float up_partial[kNSimd];
  if (lane == 0) {
    gate_partial[simd_id] = gate_acc;
    up_partial[simd_id] = up_acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_id == 0) {
    float gate_total = lane < kNSimd ? gate_partial[lane] : 0.0f;
    float up_total = lane < kNSimd ? up_partial[lane] : 0.0f;
    gate_total = simd_sum(gate_total);
    up_total = simd_sum(up_total);
    if (lane == 0) {
      size_t offset = size_t(selected) * kIntermediateSize + out_row;
      bfloat16_t gate_t = bfloat16_t(gate_total);
      bfloat16_t up_t = bfloat16_t(up_total);
      float gate_value = min(float(gate_t), kSwiGLULimit);
      float up_value = clamp(float(up_t), -kSwiGLULimit, kSwiGLULimit);
      bfloat16_t gate_activation = bfloat16_t(gate_value);
      bfloat16_t up_activation = bfloat16_t(up_value);
      auto sigmoid_tail =
          1 / (1 + metal::exp(metal::abs(gate_activation)));
      bfloat16_t sigmoid_value = gate_activation < 0
          ? sigmoid_tail
          : 1 - sigmoid_tail;
      bfloat16_t silu_value = gate_activation * sigmoid_value;
      bfloat16_t activated = silu_value * up_activation;
      gate_output[offset] = gate_t;
      up_output[offset] = up_t;
      sigmoid_output[offset] = sigmoid_value;
      silu_output[offset] = silu_value;
      hidden[offset] = activated;
    }
  }
}

[[kernel]] void glm53_native_packed_selected8_down(
    device const bfloat16_t* hidden [[buffer(0)]],
    device const uint* expert_ids [[buffer(1)]],
    device const uint8_t* weight [[buffer(2)]],
    device const float* scale_inv [[buffer(3)]],
    device bfloat16_t* output [[buffer(4)]],
    constant const int& intermediate_size [[buffer(5)]],
    constant const int& hidden_size [[buffer(6)]],
    constant const int& intermediate_scale_rows [[buffer(7)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]]) {
  uint selected = group_id / uint(hidden_size);
  uint out_row = group_id % uint(hidden_size);
  if (selected >= 8u) return;
  uint expert = expert_ids[selected];
  const device uint8_t* wr = weight +
      (size_t(expert) * uint(hidden_size) + out_row) * uint(intermediate_size);
  const device bfloat16_t* xr =
      hidden + size_t(selected) * uint(intermediate_size);
  float acc = 0.0f;
  for (uint k = tid; k < uint(intermediate_size); k += 256u) {
    size_t scale_offset =
        (size_t(expert) * uint(hidden_size / 128) + out_row / 128u) *
            uint(intermediate_scale_rows) +
        k / 128u;
    acc += float(xr[k]) * glm53_native_e4m3(wr[k]) *
        scale_inv[scale_offset];
  }
  acc = simd_sum(acc);
  threadgroup float partial[8];
  if (lane == 0) partial[simd_id] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_id == 0) {
    float reduced = lane < 8u ? partial[lane] : 0.0f;
    reduced = simd_sum(reduced);
    if (lane == 0) {
      output[size_t(selected) * uint(hidden_size) + out_row] =
          bfloat16_t(reduced);
    }
  }
}

[[kernel]] void glm53_native_packed_selected8_weighted_reduction(
    device const bfloat16_t* down [[buffer(0)]],
    device const float* scores [[buffer(1)]],
    device bfloat16_t* output [[buffer(2)]],
    constant const int& hidden_size [[buffer(3)]],
    uint out_row [[thread_position_in_grid]]) {
  if (out_row >= uint(hidden_size)) return;
  float total = 0.0f;
  for (uint selected = 0; selected < 8u; ++selected) {
    float contribution =
        float(down[size_t(selected) * uint(hidden_size) + out_row]) *
        scores[selected];
    total += contribution;
  }
  output[out_row] = bfloat16_t(total);
}

[[kernel]] void glm53_native_shared_gate_up_swiglu(
    device const bfloat16_t* x [[buffer(0)]],
    device const uint8_t* gate_weight [[buffer(1)]],
    device const float* gate_scale_inv [[buffer(2)]],
    device const uint8_t* up_weight [[buffer(3)]],
    device const float* up_scale_inv [[buffer(4)]],
    device bfloat16_t* hidden [[buffer(5)]],
    constant const int& hidden_size [[buffer(6)]],
    constant const int& intermediate_size [[buffer(7)]],
    constant const int& hidden_scale_rows [[buffer(8)]],
    constant const int& swiglu_limit [[buffer(9)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint out_row [[threadgroup_position_in_grid]]) {
  if (out_row >= uint(intermediate_size)) return;
  const device uint8_t* gate_wr =
      gate_weight + size_t(out_row) * uint(hidden_size);
  const device uint8_t* up_wr =
      up_weight + size_t(out_row) * uint(hidden_size);
  float gate_acc = 0.0f;
  float up_acc = 0.0f;
  for (uint k = tid; k < uint(hidden_size); k += 256u) {
    size_t scale_offset = size_t(out_row / 128u) *
        uint(hidden_scale_rows) + k / 128u;
    gate_acc += float(x[k]) * glm53_native_e4m3(gate_wr[k]) *
        gate_scale_inv[scale_offset];
    up_acc += float(x[k]) * glm53_native_e4m3(up_wr[k]) *
        up_scale_inv[scale_offset];
  }
  gate_acc = simd_sum(gate_acc);
  up_acc = simd_sum(up_acc);
  threadgroup float gate_partial[8];
  threadgroup float up_partial[8];
  if (lane == 0) {
    gate_partial[simd_id] = gate_acc;
    up_partial[simd_id] = up_acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_id == 0) {
    float gate_total = lane < 8u ? gate_partial[lane] : 0.0f;
    float up_total = lane < 8u ? up_partial[lane] : 0.0f;
    gate_total = simd_sum(gate_total);
    up_total = simd_sum(up_total);
    if (lane == 0) {
      bfloat16_t gate_t = bfloat16_t(gate_total);
      bfloat16_t up_t = bfloat16_t(up_total);
      float limit = float(swiglu_limit);
      float gate_value = min(float(gate_t), limit);
      float up_value = clamp(float(up_t), -limit, limit);
      bfloat16_t gate_activation = bfloat16_t(gate_value);
      bfloat16_t up_activation = bfloat16_t(up_value);
      auto sigmoid_tail =
          1 / (1 + metal::exp(metal::abs(gate_activation)));
      bfloat16_t sigmoid_value = gate_activation < 0
          ? sigmoid_tail
          : 1 - sigmoid_tail;
      bfloat16_t silu_value = gate_activation * sigmoid_value;
      hidden[out_row] = bfloat16_t(silu_value * up_activation);
    }
  }
}

[[kernel]] void glm53_native_shared_down(
    device const bfloat16_t* hidden [[buffer(0)]],
    device const uint8_t* weight [[buffer(1)]],
    device const float* scale_inv [[buffer(2)]],
    device bfloat16_t* output [[buffer(3)]],
    constant const int& intermediate_size [[buffer(4)]],
    constant const int& hidden_size [[buffer(5)]],
    constant const int& intermediate_scale_rows [[buffer(6)]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint out_row [[threadgroup_position_in_grid]]) {
  if (out_row >= uint(hidden_size)) return;
  const device uint8_t* wr =
      weight + size_t(out_row) * uint(intermediate_size);
  float acc = 0.0f;
  for (uint k = tid; k < uint(intermediate_size); k += 256u) {
    size_t scale_offset = size_t(out_row / 128u) *
        uint(intermediate_scale_rows) + k / 128u;
    acc += float(hidden[k]) * glm53_native_e4m3(wr[k]) *
        scale_inv[scale_offset];
  }
  acc = simd_sum(acc);
  threadgroup float partial[8];
  if (lane == 0) partial[simd_id] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_id == 0) {
    float reduced = lane < 8u ? partial[lane] : 0.0f;
    reduced = simd_sum(reduced);
    if (lane == 0) output[out_row] = bfloat16_t(reduced);
  }
}

[[kernel]] void glm53_native_add_routed_shared(
    device const bfloat16_t* routed [[buffer(0)]],
    device const bfloat16_t* shared [[buffer(1)]],
    device bfloat16_t* output [[buffer(2)]],
    constant const int& hidden_size [[buffer(3)]],
    uint position [[thread_position_in_grid]]) {
  if (position < uint(hidden_size)) {
    output[position] = bfloat16_t(
        float(routed[position]) + float(shared[position]));
  }
}

// Advance the bounded rollback representation without concatenating a new
// MLX graph.  Keys/gates use a ping-pong destination; metadata is copied by
// the first 19 lanes of the same dispatch.
[[kernel]] void glm53_native_advance_indexpool_raw19(
    device const bfloat16_t* raw_keys [[buffer(0)]],
    device const bfloat16_t* raw_gates [[buffer(1)]],
    device const bool* raw_valid [[buffer(2)]],
    device const long* raw_positions [[buffer(3)]],
    device const bfloat16_t* current_key [[buffer(4)]],
    device const bfloat16_t* current_gate [[buffer(5)]],
    device const bool* current_valid [[buffer(6)]],
    device bfloat16_t* next_keys [[buffer(7)]],
    device bfloat16_t* next_gates [[buffer(8)]],
    device bool* next_valid [[buffer(9)]],
    device long* next_positions [[buffer(10)]],
    constant const int& previous_total_tokens [[buffer(11)]],
    uint position [[thread_position_in_grid]]) {
  constexpr uint kElements = kIndexPoolRawWindow * kIndexPoolHeadDim;
  if (position < kElements) {
    uint row = position / kIndexPoolHeadDim;
    uint column = position % kIndexPoolHeadDim;
    if (row + 1 < kIndexPoolRawWindow) {
      uint source = (row + 1) * kIndexPoolHeadDim + column;
      next_keys[position] = raw_keys[source];
      next_gates[position] = raw_gates[source];
    } else {
      next_keys[position] = current_key[column];
      next_gates[position] = current_gate[column];
    }
  }
  if (position < kIndexPoolRawWindow) {
    if (position + 1 < kIndexPoolRawWindow) {
      next_valid[position] = raw_valid[position + 1];
      next_positions[position] = raw_positions[position + 1];
    } else {
      next_valid[position] = current_valid[0];
      next_positions[position] = long(previous_total_tokens);
    }
  }
}

// Decode-specialized kpool=4 update.  Eager MLX keeps the BF16 boundaries of
// subtraction, exp, reduction, division, product, and final reduction; using
// the FP32-accumulating attention softmax changes pool bytes.  One thread owns
// one head dimension so the four-lane order is explicit and durable.
[[kernel]] void glm53_native_update_indexpool_row_bfloat16(
    device const bfloat16_t* raw_keys [[buffer(0)]],
    device const bfloat16_t* raw_gates [[buffer(1)]],
    device const bool* raw_valid [[buffer(2)]],
    device const bfloat16_t* compress_ape [[buffer(3)]],
    device bfloat16_t* debug_logits [[buffer(4)]],
    device bfloat16_t* debug_probabilities [[buffer(5)]],
    device bfloat16_t* pool_keys [[buffer(6)]],
    device long* pool_indices [[buffer(7)]],
    device bool* pool_valid [[buffer(8)]],
    constant const int& pool_row [[buffer(9)]],
    constant const int& active_count [[buffer(10)]],
    uint dimension [[thread_position_in_grid]]) {
  if (dimension >= kIndexPoolHeadDim) return;
  bfloat16_t logits[kIndexKPool];
  bfloat16_t maximum = Limits<bfloat16_t>::finite_min;
  bool lane_valid[kIndexKPool];
  bool all_valid = active_count == int(kIndexKPool);
  for (uint lane = 0; lane < kIndexKPool; ++lane) {
    int source_row = int(kIndexPoolRawWindow) - active_count + int(lane);
    bool valid = int(lane) < active_count && source_row >= 0 &&
        raw_valid[source_row];
    lane_valid[lane] = valid;
    all_valid = all_valid && valid;
    if (valid) {
      bfloat16_t gate =
          raw_gates[uint(source_row) * kIndexPoolHeadDim + dimension];
      bfloat16_t ape = compress_ape[lane * kIndexPoolHeadDim + dimension];
      logits[lane] = bfloat16_t(float(gate) + float(ape));
    } else {
      logits[lane] = bfloat16_t(-1.0e30f);
    }
    maximum = maximum < logits[lane] ? logits[lane] : maximum;
    debug_logits[dimension * kIndexKPool + lane] = logits[lane];
  }
  bfloat16_t exponentials[kIndexKPool];
  bfloat16_t normalizer = bfloat16_t(0.0f);
  for (uint lane = 0; lane < kIndexKPool; ++lane) {
    bfloat16_t delta = bfloat16_t(float(logits[lane]) - float(maximum));
    exponentials[lane] = bfloat16_t(fast::exp(float(delta)));
    normalizer =
        bfloat16_t(float(normalizer) + float(exponentials[lane]));
  }
  bfloat16_t total = bfloat16_t(0.0f);
  for (uint lane = 0; lane < kIndexKPool; ++lane) {
    bfloat16_t probability = bfloat16_t(
        float(exponentials[lane]) / float(normalizer));
    debug_probabilities[dimension * kIndexKPool + lane] = probability;
    int source_row = int(kIndexPoolRawWindow) - active_count + int(lane);
    // Eager take_along_axis clips missing lanes to the final suffix token.
    // This matters when every lane is invalid: softmax is uniform and the
    // invalid pool row still has an authoritative (though unselectable) key.
    uint safe_row = lane_valid[lane]
        ? uint(source_row)
        : uint(kIndexPoolRawWindow - 1);
    bfloat16_t key = raw_keys[safe_row * kIndexPoolHeadDim + dimension];
    bfloat16_t product =
        bfloat16_t(float(probability) * float(key));
    total = bfloat16_t(float(total) + float(product));
  }
  pool_keys[size_t(pool_row) * kIndexPoolHeadDim + dimension] = total;
  if (dimension < kIndexKPool) {
    uint lane = dimension;
    pool_indices[size_t(pool_row) * kIndexKPool + lane] = lane_valid[lane]
        ? long(pool_row * int(kIndexKPool) + int(lane))
        : long(-1);
  }
  if (dimension == 0) pool_valid[pool_row] = all_valid;
}

// The matmul itself is the same MLX Steel kernel selected on apple-gpu-d for
// this BF16 nt geometry.  This tail fixes the eager BF16 materialization
// points after that exact matmul output.
[[kernel]] void glm53_native_finish_pooled_score_bfloat16_pool32(
    device const bfloat16_t* head_scores [[buffer(0)]],
    device const bfloat16_t* mixture_weights [[buffer(1)]],
    device const bool* pool_valid [[buffer(2)]],
    device bfloat16_t* output [[buffer(3)]],
    constant const int& logical_pool_rows [[buffer(4)]],
    constant const int& physical_pool_rows [[buffer(5)]],
    constant const float& softmax_scale_fp32 [[buffer(6)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]) {
  constexpr uint kPoolsPerGroup = 32;
  constexpr uint kSimdGroups = 8;
  constexpr uint kPoolsPerSimd = kPoolsPerGroup / kSimdGroups;
  threadgroup bfloat16_t shared[kIndexerHeads * kPoolsPerGroup];

  uint linear_lane = simd_group * 32 + simd_lane;
  uint head = linear_lane / 8;
  uint pool_lane = (linear_lane % 8) * kPoolsPerSimd;
  uint pool_base = group.x * kPoolsPerGroup;
  uint row = group.y;

  bfloat16_t scale = bfloat16_t(softmax_scale_fp32);
  for (uint item = 0; item < kPoolsPerSimd; ++item) {
    uint pool = pool_base + pool_lane + item;
    bfloat16_t weighted = bfloat16_t(0.0f);
    if (pool < uint(physical_pool_rows)) {
      size_t source =
          (size_t(row) * kIndexerHeads + head) *
              uint(physical_pool_rows) +
          pool;
      bfloat16_t scaled =
          bfloat16_t(float(head_scores[source]) * float(scale));
      bfloat16_t clipped = scaled > bfloat16_t(0.0f)
          ? scaled
          : bfloat16_t(0.0f);
      weighted = bfloat16_t(
          float(mixture_weights[size_t(row) * kIndexerHeads + head]) *
          float(clipped));
    }
    shared[head * kPoolsPerGroup + pool_lane + item] = weighted;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint item = 0; item < kPoolsPerSimd; ++item) {
    uint within_tile = simd_group * kPoolsPerSimd + item;
    uint pool = pool_base + within_tile;
    bfloat16_t total = simd_sum(
        shared[simd_lane * kPoolsPerGroup + within_tile]);
    if (simd_lane == 0 && pool < uint(physical_pool_rows)) {
      output[size_t(row) * uint(physical_pool_rows) + pool] =
          pool < uint(logical_pool_rows) && pool_valid[pool]
          ? total
          : bfloat16_t(-1.0e30f);
    }
  }
}

template <typename T>
[[kernel]] void native_exact_partial_topk_512(
    device const T* scores [[buffer(0)]],
    device uint* indices [[buffer(1)]],
    constant const int& logical_count [[buffer(2)]],
    constant const int& score_stride [[buffer(3)]],
    uint2 lane_position [[thread_position_in_threadgroup]],
    uint2 group_position [[threadgroup_position_in_grid]]) {
  uint lane = lane_position.x;
  uint row = group_position.y;
  const device T* row_scores = scores + size_t(row) * uint(score_stride);
  device uint* row_output = indices + size_t(row) * kSelectedPools;
  threadgroup atomic_uint histogram[16];
  threadgroup atomic_uint candidate_count;
  threadgroup ulong prefix_shared;
  threadgroup uint rank_shared;
  threadgroup ulong candidates[kSelectedPools];

  if (lane == 0) {
    prefix_shared = 0;
    rank_shared = kSelectedPools;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int shift = 48; shift >= 0; shift -= 4) {
    if (lane < 16) {
      atomic_store_explicit(&histogram[lane], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    ulong prefix = prefix_shared;
    for (uint index = lane; index < uint(logical_count); index += kSelectedPools) {
      float value = float(row_scores[index]);
      uint bits = as_type<uint>(value);
      if ((bits & 0x7fffffffu) == 0u) bits = 0u;
      uint ordered = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
      ulong key = (ulong(ordered) << 17) | ulong(0x1ffffu - index);
      bool matches = shift == 48 || (key >> uint(shift + 4)) == prefix;
      if (matches) {
        uint digit = uint((key >> uint(shift)) & 0xful);
        atomic_fetch_add_explicit(&histogram[digit], 1u, memory_order_relaxed);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
      uint rank = rank_shared;
      uint skipped = 0;
      uint chosen = 0;
      for (int digit = 15; digit >= 0; --digit) {
        uint in_bin = atomic_load_explicit(
            &histogram[digit], memory_order_relaxed);
        if (rank > skipped && rank <= skipped + in_bin) {
          chosen = uint(digit);
          rank_shared = rank - skipped;
          break;
        }
        skipped += in_bin;
      }
      prefix_shared = (prefix_shared << 4) | ulong(chosen);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (lane == 0) {
    atomic_store_explicit(&candidate_count, 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ulong threshold = prefix_shared;
  for (uint index = lane; index < uint(logical_count); index += kSelectedPools) {
    float value = float(row_scores[index]);
    uint bits = as_type<uint>(value);
    if ((bits & 0x7fffffffu) == 0u) bits = 0u;
    uint ordered = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
    ulong key = (ulong(ordered) << 17) | ulong(0x1ffffu - index);
    if (key >= threshold) {
      uint slot = atomic_fetch_add_explicit(
          &candidate_count, 1u, memory_order_relaxed);
      if (slot < kSelectedPools) candidates[slot] = key;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint span = 2; span <= kSelectedPools; span <<= 1) {
    for (uint stride = span >> 1; stride > 0; stride >>= 1) {
      uint peer = lane ^ stride;
      if (peer > lane) {
        ulong left = candidates[lane];
        ulong right = candidates[peer];
        bool ascending = (lane & span) == 0;
        bool swap = ascending ? (left > right) : (left < right);
        if (swap) {
          candidates[lane] = right;
          candidates[peer] = left;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  ulong key = candidates[kSelectedPools - 1 - lane];
  row_output[lane] = 0x1ffffu - uint(key & 0x1fffful);
}

[[kernel]] void glm53_native_expand_selected_pools(
    device const uint* selected [[buffer(0)]],
    device const long* pool_indices [[buffer(1)]],
    device const bool* pool_valid [[buffer(2)]],
    device const long* raw_positions [[buffer(3)]],
    device const bool* raw_valid [[buffer(4)]],
    device const bool* current_valid [[buffer(5)]],
    device int* output [[buffer(6)]],
    device bool* output_valid [[buffer(7)]],
    constant const int& logical_pool_rows [[buffer(8)]],
    constant const int& kv_len [[buffer(9)]],
    constant const int& active_tail_count [[buffer(10)]],
    constant const int& raw_width [[buffer(11)]],
    constant const int& raw_rows [[buffer(12)]],
    uint2 position [[thread_position_in_grid]]) {
  uint column = position.x;
  uint row = position.y;
  if (column >= kSelectedWidth) return;
  size_t destination = size_t(row) * kSelectedWidth + column;
  bool valid = current_valid[row];
  long token = -1;
  if (column < kSelectedPools * kIndexKPool) {
    uint selected_slot = column / kIndexKPool;
    uint lane = column % kIndexKPool;
    uint pool = selected[size_t(row) * kSelectedPools + selected_slot];
    valid = valid && pool < uint(logical_pool_rows) && pool_valid[pool];
    if (valid) token = pool_indices[size_t(pool) * kIndexKPool + lane];
  } else {
    int tail_slot = int(column - kSelectedPools * kIndexKPool);
    valid = valid && tail_slot < active_tail_count;
    int source = raw_width - active_tail_count + tail_slot;
    int raw_row = raw_rows == 1 ? 0 : int(row);
    int raw_offset = raw_row * raw_width + source;
    valid = valid && source >= 0 && raw_valid[raw_offset];
    if (valid) token = raw_positions[raw_offset];
  }
  valid = valid && token >= 0 && token < kv_len;
  output[destination] = valid ? int(token) : -1;
  output_valid[destination] = valid;
}

#define instantiate_native_topk(type_name, type)                             \
  instantiate_kernel(                                                       \
      "glm53_native_exact_partial_topk_512_" #type_name,                    \
      native_exact_partial_topk_512,                                        \
      type)

instantiate_native_topk(float32, float);
instantiate_native_topk(bfloat16, bfloat16_t);

instantiate_kernel(
    "glm53_native_steel_gemm_nt_bfloat16_bfloat16_bm64_bn64_bk16_wm1_wn2",
    gemm,
    bfloat16_t,
    64,
    64,
    16,
    1,
    2,
    false,
    true,
    float);

// Decode-only sparse DSA attention uses head dimension 512, which is not a
// fused-SDPA geometry in the pinned MLX 0.32.2 runtime.  Instantiate the exact
// Steel/precise-softmax sequence selected by that fallback so the native plan
// can encode it without returning its gathered latent or score scratch to MLX.
[[kernel]] void glm53_native_prepare_sparse_attention_bfloat16(
    device const int* selected_indices [[buffer(0)]],
    device const bool* selected_valid [[buffer(1)]],
    device const bfloat16_t* query [[buffer(2)]],
    device const bfloat16_t* latent [[buffer(3)]],
    device bfloat16_t* scaled_query [[buffer(4)]],
    device bfloat16_t* gathered_latent [[buffer(5)]],
    constant const float& scale_fp32 [[buffer(6)]],
    constant const int& physical_kv_rows [[buffer(7)]],
    uint position [[thread_position_in_grid]]) {
  constexpr uint kAttentionHeads = 64;
  constexpr uint kAttentionDim = 512;
  constexpr uint kQueryElements = kAttentionHeads * kAttentionDim;
  constexpr uint kGatherElements = kSelectedWidth * kAttentionDim;
  bfloat16_t scale = bfloat16_t(scale_fp32);
  if (position < kQueryElements) {
    scaled_query[position] =
        bfloat16_t(float(query[position]) * float(scale));
  }
  if (position < kGatherElements) {
    uint slot = position / kAttentionDim;
    uint column = position % kAttentionDim;
    int index = selected_indices[slot];
    bool valid = selected_valid[slot] && index >= 0 && index < physical_kv_rows;
    // The eager gather sanitizes invalid indices to row zero and applies a
    // separate attention mask.  Preserve that exact intermediate contract.
    uint source_row = valid ? uint(index) : 0u;
    gathered_latent[position] =
        latent[size_t(source_row) * kAttentionDim + column];
  }
}

[[kernel]] void glm53_native_mask_sparse_attention_scores_bfloat16(
    device bfloat16_t* scores [[buffer(0)]],
    device const bool* selected_valid [[buffer(1)]],
    uint position [[thread_position_in_grid]]) {
  constexpr uint kAttentionHeads = 64;
  constexpr uint kScoreElements = kAttentionHeads * kSelectedWidth;
  if (position >= kScoreElements) return;
  uint slot = position % kSelectedWidth;
  if (!selected_valid[slot]) {
    scores[position] = Limits<bfloat16_t>::finite_min;
  }
}

instantiate_kernel(
    "glm53_native_attention_gemm_nt_bfloat16_bfloat16_bm64_bn32_bk32_wm2_wn2",
    gemm,
    bfloat16_t,
    64,
    32,
    32,
    2,
    2,
    false,
    true,
    float);

instantiate_kernel(
    "glm53_native_block_softmax_precise_bfloat16",
    softmax_single_row,
    bfloat16_t,
    float,
    SOFTMAX_N_READS);

instantiate_kernel(
    "glm53_native_attention_gemm_splitk_nn_bfloat16_float32_bm32_bn32_bk16_wm2_wn2_MN_taligned_K_naligned",
    gemm_splitk,
    bfloat16_t,
    float,
    32,
    32,
    16,
    2,
    2,
    false,
    false,
    true,
    false);

instantiate_kernel(
    "glm53_native_attention_gemm_splitk_accum_bfloat16_float32",
    gemm_splitk_accum,
    float,
    bfloat16_t);
