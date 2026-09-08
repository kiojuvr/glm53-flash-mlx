#include "native_selected_union_plan.h"

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

int checked_physical_k(int value) {
  if (value <= 0 || value > (512 << 10)) {
    throw std::invalid_argument("physical K must be in [1, 512K]");
  }
  return value;
}

} // namespace

NativeSelectedUnionPlan::NativeSelectedUnionPlan(int physical_k)
    : physical_k_(checked_physical_k(physical_k)),
      word_count_(static_cast<uint32_t>((physical_k_ + 31) / 32)),
      block_count_(
          static_cast<uint32_t>((physical_k_ + kTokensPerBlock - 1) /
                                kTokensPerBlock)),
      stream_(mx::default_stream(mx::Device(mx::Device::gpu))),
      membership_words_(owned_array(
          {static_cast<int>(word_count_)}, mx::uint32)),
      block_counts_(owned_array(
          {static_cast<int>(block_count_)}, mx::uint32)),
      block_prefix_(owned_array(
          {static_cast<int>(block_count_)}, mx::uint32)),
      union_indices_(owned_array({physical_k_}, mx::int32)),
      physical_to_union_(owned_array({physical_k_}, mx::int32)),
      query_union_slots_(owned_array(
          {kQueryRows, kSelectedWidth}, mx::int32)),
      union_count_(owned_array({1}, mx::uint32)) {
  auto &device = mx::metal::device(stream_.device);
  auto *library = device.get_library(
      "glm53_native_execution", current_binary_dir());
  mark_pipeline_ = device.get_kernel(
      "glm53_native_mark_selected_union", library);
  count_pipeline_ = device.get_kernel(
      "glm53_native_count_selected_union_blocks", library);
  prefix_pipeline_ = device.get_kernel(
      "glm53_native_prefix_selected_union_blocks", library);
  scatter_pipeline_ = device.get_kernel(
      "glm53_native_scatter_selected_union", library);
  map_pipeline_ = device.get_kernel(
      "glm53_native_map_queries_to_selected_union", library);
  initial_buffer_identities_ = buffer_identities();
}

void NativeSelectedUnionPlan::validate_input(
    const mx::array &array, const char *name, mx::Dtype dtype) const {
  if (array.dtype() != dtype ||
      array.size() != static_cast<size_t>(kQueryRows) * kSelectedWidth) {
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

std::vector<mx::array> NativeSelectedUnionPlan::execute(
    const mx::array &selected_indices, const mx::array &selected_valid) {
  validate_input(selected_indices, "selected_indices", mx::int32);
  validate_input(selected_valid, "selected_valid", mx::bool_);
  constexpr uint32_t selected_elements = kQueryRows * kSelectedWidth;
  auto &encoder = mx::metal::get_command_encoder(stream_);

  encoder.set_compute_pipeline_state(mark_pipeline_);
  encoder.set_input_array(selected_indices, 0);
  encoder.set_input_array(selected_valid, 1);
  encoder.set_output_array(membership_words_, 2);
  encoder.set_bytes(physical_k_, 3);
  encoder.set_bytes(word_count_, 4);
  encoder.dispatch_threads(
      MTL::Size(word_count_, 1, 1), MTL::Size(256, 1, 1));
  encoder.barrier();

  encoder.set_compute_pipeline_state(count_pipeline_);
  encoder.set_input_array(membership_words_, 0);
  encoder.set_output_array(block_counts_, 1);
  encoder.set_bytes(word_count_, 2);
  encoder.set_bytes(block_count_, 3);
  encoder.dispatch_threads(
      MTL::Size(block_count_, 1, 1), MTL::Size(256, 1, 1));
  encoder.barrier();

  encoder.set_compute_pipeline_state(prefix_pipeline_);
  encoder.set_input_array(block_counts_, 0);
  encoder.set_output_array(block_prefix_, 1);
  encoder.set_output_array(union_count_, 2);
  encoder.set_bytes(block_count_, 3);
  encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
  encoder.barrier();

  encoder.set_compute_pipeline_state(scatter_pipeline_);
  encoder.set_input_array(membership_words_, 0);
  encoder.set_input_array(block_prefix_, 1);
  encoder.set_output_array(union_indices_, 2);
  encoder.set_output_array(physical_to_union_, 3);
  encoder.set_bytes(physical_k_, 4);
  encoder.dispatch_threads(
      MTL::Size(physical_k_, 1, 1), MTL::Size(256, 1, 1));
  encoder.barrier();

  encoder.set_compute_pipeline_state(map_pipeline_);
  encoder.set_input_array(selected_indices, 0);
  encoder.set_input_array(selected_valid, 1);
  encoder.set_input_array(physical_to_union_, 2);
  encoder.set_output_array(query_union_slots_, 3);
  encoder.set_bytes(physical_k_, 4);
  encoder.set_bytes(selected_elements, 5);
  encoder.dispatch_threads(
      MTL::Size(selected_elements, 1, 1), MTL::Size(256, 1, 1));

  ++execution_count_;
  if (buffer_identities() != initial_buffer_identities_) {
    throw std::runtime_error("selected union buffer identity changed");
  }
  return {union_indices_, union_count_, query_union_slots_};
}

uint64_t NativeSelectedUnionPlan::scratch_bytes() const {
  return membership_words_.nbytes() + block_counts_.nbytes() +
      block_prefix_.nbytes() + union_indices_.nbytes() +
      physical_to_union_.nbytes() + query_union_slots_.nbytes() +
      union_count_.nbytes();
}

std::vector<uint64_t> NativeSelectedUnionPlan::buffer_identities() const {
  return {
      buffer_identity(membership_words_), buffer_identity(block_counts_),
      buffer_identity(block_prefix_), buffer_identity(union_indices_),
      buffer_identity(physical_to_union_),
      buffer_identity(query_union_slots_), buffer_identity(union_count_),
  };
}

} // namespace glm53::native_execution
