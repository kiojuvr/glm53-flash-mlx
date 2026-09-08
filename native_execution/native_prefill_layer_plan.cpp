#include "native_prefill_layer_plan.h"

#include <dlfcn.h>
#include <filesystem>
#include <stdexcept>

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"

namespace glm53::native_execution {

namespace {

std::string current_binary_dir() {
  static const std::string directory = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void *>(&current_binary_dir), &info)) {
      throw std::runtime_error("cannot resolve native execution binary path");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return directory;
}

size_t checked_elements(const mx::Shape &shape) {
  size_t result = 1;
  for (auto extent : shape) {
    if (extent <= 0) {
      throw std::invalid_argument("native buffer extents must be positive");
    }
    result *= static_cast<size_t>(extent);
  }
  return result;
}

mx::array owned_array(const mx::Shape &shape, mx::Dtype dtype) {
  const auto elements = checked_elements(shape);
  return mx::array(
      mx::allocator::malloc(elements * mx::size_of(dtype)), shape, dtype);
}

uint64_t buffer_identity(const mx::array &array) {
  return reinterpret_cast<uint64_t>(array.buffer().ptr());
}

} // namespace

NativePrefillLayerSubstrate::NativePrefillLayerSubstrate()
    : stream_(mx::default_stream(mx::Device(mx::Device::gpu))),
      hidden_ping_(owned_array({kQueryRows, kHiddenSize}, mx::bfloat16)),
      hidden_pong_(owned_array({kQueryRows, kHiddenSize}, mx::bfloat16)),
      dsa_score_scratch_(owned_array(
          {kQueryBlockRows, kPhysicalPoolRows}, mx::float32)),
      selected_indices_(owned_array(
          {kQueryRows, kSelectedWidth}, mx::int32)),
      selected_valid_(owned_array(
          {kQueryRows, kSelectedWidth}, mx::bool_)),
      route_experts_(owned_array({kRouteRows}, mx::uint32)),
      route_scores_(owned_array({kRouteRows}, mx::bfloat16)),
      route_order_(owned_array({kRouteRows}, mx::uint32)),
      moe_hidden_scratch_(owned_array(
          {kRouteRows, kIntermediateSize}, mx::bfloat16)),
      moe_down_scratch_(owned_array(
          {kRouteRows, kHiddenSize}, mx::bfloat16)) {
  auto &device = mx::metal::device(stream_.device);
  auto *library = device.get_library(
      "glm53_native_execution", current_binary_dir());
  copy_pipeline_ = device.get_kernel(
      "glm53_native_prefill_copy_bfloat16", library);
  initial_buffer_identities_ = buffer_identities();
}

void NativePrefillLayerSubstrate::validate_hidden(
    const mx::array &hidden) const {
  if (hidden.dtype() != mx::bfloat16 ||
      hidden.ndim() != 2 || hidden.shape(0) != kQueryRows ||
      hidden.shape(1) != kHiddenSize) {
    throw std::invalid_argument(
        "native prefill hidden must be [256, 4096] bfloat16");
  }
  if (!hidden.flags().row_contiguous) {
    throw std::invalid_argument("native prefill hidden must be row-contiguous");
  }
  if (hidden.status() == mx::array::Status::unscheduled) {
    throw std::invalid_argument(
        "native prefill hidden must be scheduled before submission");
  }
}

std::vector<mx::array>
NativePrefillLayerSubstrate::execute(const mx::array &hidden) {
  validate_hidden(hidden);
  auto &encoder = mx::metal::get_command_encoder(stream_);
  constexpr uint32_t elements = kQueryRows * kHiddenSize;
  constexpr uint32_t threads = 256;
  constexpr uint32_t groups = (elements + threads - 1) / threads;

  // The three stages deliberately share one encoder scope. DSA and MoE math
  // replace the middle copies without exposing their arena buffers to MLX.
  encoder.set_compute_pipeline_state(copy_pipeline_);
  encoder.set_input_array(hidden, 0);
  encoder.set_output_array(hidden_ping_, 1);
  encoder.set_bytes(elements, 2);
  encoder.dispatch_threadgroups(
      MTL::Size(groups, 1, 1), MTL::Size(threads, 1, 1));
  encoder.barrier();

  encoder.set_compute_pipeline_state(copy_pipeline_);
  encoder.set_input_array(hidden_ping_, 0);
  encoder.set_output_array(hidden_pong_, 1);
  encoder.set_bytes(elements, 2);
  encoder.dispatch_threadgroups(
      MTL::Size(groups, 1, 1), MTL::Size(threads, 1, 1));
  encoder.barrier();

  encoder.set_compute_pipeline_state(copy_pipeline_);
  encoder.set_input_array(hidden_pong_, 0);
  encoder.set_output_array(hidden_ping_, 1);
  encoder.set_bytes(elements, 2);
  encoder.dispatch_threadgroups(
      MTL::Size(groups, 1, 1), MTL::Size(threads, 1, 1));

  ++execution_count_;
  return {hidden_ping_};
}

uint64_t NativePrefillLayerSubstrate::dsa_score_scratch_bytes() const {
  return dsa_score_scratch_.nbytes();
}

uint64_t NativePrefillLayerSubstrate::scratch_bytes() const {
  return dsa_score_scratch_.nbytes() + selected_indices_.nbytes() +
      selected_valid_.nbytes() + route_experts_.nbytes() +
      route_scores_.nbytes() + route_order_.nbytes() +
      moe_hidden_scratch_.nbytes() + moe_down_scratch_.nbytes();
}

uint64_t NativePrefillLayerSubstrate::arena_bytes() const {
  return hidden_ping_.nbytes() + hidden_pong_.nbytes() + scratch_bytes();
}

std::vector<uint64_t> NativePrefillLayerSubstrate::buffer_identities() const {
  return {
      buffer_identity(hidden_ping_),
      buffer_identity(hidden_pong_),
      buffer_identity(dsa_score_scratch_),
      buffer_identity(selected_indices_),
      buffer_identity(selected_valid_),
      buffer_identity(route_experts_),
      buffer_identity(route_scores_),
      buffer_identity(route_order_),
      buffer_identity(moe_hidden_scratch_),
      buffer_identity(moe_down_scratch_),
  };
}

bool NativePrefillLayerSubstrate::buffer_identities_stable() const {
  return buffer_identities() == initial_buffer_identities_;
}

std::vector<std::string> NativePrefillLayerSubstrate::fixed_topology() const {
  return {
      "ingress_to_hidden_ping",
      "dsa_region_hidden_ping_to_pong",
      "moe_region_hidden_pong_to_layer_output",
  };
}

} // namespace glm53::native_execution
