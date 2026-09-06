#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

constant uint kSelectedPools = 512;
constant uint kIndexKPool = 4;
constant uint kSelectedWidth = 2051;

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
