#include "native_prefill_moe_route_plan.h"

#include <dlfcn.h>
#include <filesystem>
#include <stdexcept>
#include <string>

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

mx::array owned_array(const mx::Shape &shape, mx::Dtype dtype) {
  size_t elements = 1;
  for (auto extent : shape) {
    if (extent <= 0) {
      throw std::invalid_argument("native buffer extents must be positive");
    }
    elements *= static_cast<size_t>(extent);
  }
  return mx::array(
      mx::allocator::malloc(elements * mx::size_of(dtype)), shape, dtype);
}

uint64_t buffer_identity(const mx::array &array) {
  return reinterpret_cast<uint64_t>(array.buffer().ptr());
}

} // namespace

NativePrefillMoERoutePlan::NativePrefillMoERoutePlan(int expert_count)
    : expert_count_(expert_count),
      descriptor_capacity_((kRouteRows + kTileRows - 1) / kTileRows +
                           expert_count),
      stream_(mx::default_stream(mx::Device(mx::Device::gpu))),
      sorted_route_order_(owned_array({kRouteRows}, mx::uint32)),
      inverse_route_order_(owned_array({kRouteRows}, mx::uint32)),
      sorted_experts_(owned_array({kRouteRows}, mx::uint32)),
      sorted_scores_(owned_array({kRouteRows}, mx::float32)),
      expert_offsets_(owned_array({expert_count + 1}, mx::uint32)),
      tile_experts_(owned_array({descriptor_capacity_}, mx::uint32)),
      tile_starts_(owned_array({descriptor_capacity_}, mx::uint32)),
      tile_lengths_(owned_array({descriptor_capacity_}, mx::uint32)),
      descriptor_count_(owned_array({1}, mx::uint32)),
      invalid_route_count_(owned_array({1}, mx::uint32)) {
  if (expert_count_ < kTopK || expert_count_ > 1024) {
    throw std::invalid_argument("expert_count must be in [8, 1024]");
  }
  auto &device = mx::metal::device(stream_.device);
  auto *library =
      device.get_library("glm53_native_execution", current_binary_dir());
  group_pipeline_ = device.get_kernel(
      "glm53_native_prefill_moe_count_routes", library);
  offset_pipeline_ = device.get_kernel(
      "glm53_native_prefill_moe_expert_prefix", library);
  scatter_pipeline_ = device.get_kernel(
      "glm53_native_prefill_moe_stable_scatter_routes", library);
  descriptor_pipeline_ = device.get_kernel(
      "glm53_native_prefill_moe_tile_descriptors", library);
  initial_buffer_identities_ = buffer_identities();
}

void NativePrefillMoERoutePlan::validate_input(
    const mx::array &array, const char *name, mx::Dtype dtype) const {
  if (array.dtype() != dtype || array.size() != kRouteRows) {
    throw std::invalid_argument(std::string(name) + " shape/dtype mismatch");
  }
  if (!array.flags().row_contiguous) {
    throw std::invalid_argument(std::string(name) + " must be row-contiguous");
  }
  if (array.status() == mx::array::Status::unscheduled) {
    throw std::invalid_argument(
        std::string(name) + " must be scheduled before submission");
  }
}

std::vector<mx::array> NativePrefillMoERoutePlan::execute(
    const mx::array &expert_ids, const mx::array &scores) {
  validate_input(expert_ids, "expert_ids", mx::uint32);
  validate_input(scores, "scores", mx::float32);
  auto &encoder = mx::metal::get_command_encoder(stream_);

  encoder.set_compute_pipeline_state(group_pipeline_);
  encoder.set_input_array(expert_ids, 0);
  encoder.set_output_array(expert_offsets_, 1);
  encoder.set_output_array(invalid_route_count_, 2);
  encoder.set_bytes(expert_count_, 3);
  encoder.set_bytes(kRouteRows, 4);
  encoder.dispatch_threads(
      MTL::Size(expert_count_ + 1, 1, 1), MTL::Size(256, 1, 1));
  encoder.barrier();

  encoder.set_compute_pipeline_state(offset_pipeline_);
  encoder.set_input_array(expert_offsets_, 0);
  encoder.set_bytes(expert_count_, 1);
  encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
  encoder.barrier();

  encoder.set_compute_pipeline_state(scatter_pipeline_);
  encoder.set_input_array(expert_ids, 0);
  encoder.set_input_array(scores, 1);
  encoder.set_input_array(expert_offsets_, 2);
  encoder.set_output_array(sorted_route_order_, 3);
  encoder.set_output_array(inverse_route_order_, 4);
  encoder.set_output_array(sorted_experts_, 5);
  encoder.set_output_array(sorted_scores_, 6);
  encoder.set_bytes(expert_count_, 7);
  encoder.set_bytes(kRouteRows, 8);
  encoder.dispatch_threads(
      MTL::Size(expert_count_, 1, 1), MTL::Size(256, 1, 1));
  encoder.barrier();

  encoder.set_compute_pipeline_state(descriptor_pipeline_);
  encoder.set_input_array(expert_offsets_, 0);
  encoder.set_output_array(tile_experts_, 1);
  encoder.set_output_array(tile_starts_, 2);
  encoder.set_output_array(tile_lengths_, 3);
  encoder.set_output_array(descriptor_count_, 4);
  encoder.set_bytes(expert_count_, 5);
  encoder.set_bytes(kRouteRows, 6);
  encoder.set_bytes(kTileRows, 7);
  encoder.set_bytes(descriptor_capacity_, 8);
  encoder.dispatch_threads(
      MTL::Size(descriptor_capacity_, 1, 1), MTL::Size(256, 1, 1));

  ++execution_count_;
  if (buffer_identities() != initial_buffer_identities_) {
    throw std::runtime_error("native prefill MoE route buffer identity changed");
  }
  return {sorted_route_order_, inverse_route_order_, sorted_experts_,
          sorted_scores_, expert_offsets_, tile_experts_, tile_starts_,
          tile_lengths_, descriptor_count_, invalid_route_count_};
}

uint64_t NativePrefillMoERoutePlan::scratch_bytes() const {
  return sorted_route_order_.nbytes() + inverse_route_order_.nbytes() +
      sorted_experts_.nbytes() + sorted_scores_.nbytes() +
      expert_offsets_.nbytes() + tile_experts_.nbytes() +
      tile_starts_.nbytes() + tile_lengths_.nbytes() +
      descriptor_count_.nbytes() + invalid_route_count_.nbytes();
}

std::vector<uint64_t> NativePrefillMoERoutePlan::buffer_identities() const {
  return {
      buffer_identity(sorted_route_order_), buffer_identity(inverse_route_order_),
      buffer_identity(sorted_experts_),     buffer_identity(sorted_scores_),
      buffer_identity(expert_offsets_),     buffer_identity(tile_experts_),
      buffer_identity(tile_starts_),        buffer_identity(tile_lengths_),
      buffer_identity(descriptor_count_),   buffer_identity(invalid_route_count_),
  };
}

} // namespace glm53::native_execution
