#include "native_prefill_av_plan.h"

#include <algorithm>
#include <dlfcn.h>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/steel/gemm/params.h"

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
  return mx::array(
      mx::allocator::malloc(checked_elements(shape) * mx::size_of(dtype)),
      shape, dtype);
}

uint64_t buffer_identity(const mx::array &array) {
  return reinterpret_cast<uint64_t>(array.buffer().ptr());
}

int checked_physical_k(int physical_k) {
  if (physical_k <= 0 || physical_k > (512 << 10)) {
    throw std::invalid_argument("physical K must be in [1, 512K]");
  }
  return physical_k;
}

} // namespace

NativeSparsePrefillAVPlan::NativeSparsePrefillAVPlan(int physical_k)
    : physical_k_(checked_physical_k(physical_k)),
      packed_k_(std::min(
          ((physical_k_ + kBK - 1) / kBK) * kBK, kMaximumPackedK)),
      stream_(mx::default_stream(mx::Device(mx::Device::gpu))),
      lane_to_selected_(owned_array({packed_k_}, mx::int32)),
      output_(owned_array({1, kHeads, 1, kValueDim}, mx::bfloat16)) {
  auto &device = mx::metal::device(stream_.device);
  if (device.get_architecture().back() != 'd' ||
      device.get_architecture_gen() >= 17) {
    throw std::runtime_error(
        "prefill AV Steel geometry is qualified only for pre-NAX apple-gpu-d");
  }
  auto *library =
      device.get_library("glm53_native_execution", current_binary_dir());
  map_pipeline_ = device.get_kernel(
      "glm53_native_build_virtual_bk16_map", library);
  av_pipeline_ = device.get_kernel(
      "glm53_native_virtual_bk16_av_bfloat16", library);
  initial_buffer_identities_ = buffer_identities();
}

void NativeSparsePrefillAVPlan::validate_input(
    const mx::array &array, const char *name, mx::Dtype dtype,
    size_t elements) const {
  if (array.dtype() != dtype) {
    throw std::invalid_argument(std::string(name) + " dtype mismatch");
  }
  if (array.size() != elements) {
    throw std::invalid_argument(std::string(name) + " shape/size mismatch");
  }
  if (!array.flags().row_contiguous) {
    throw std::invalid_argument(std::string(name) + " must be row-contiguous");
  }
  if (array.status() == mx::array::Status::unscheduled) {
    throw std::invalid_argument(
        std::string(name) + " must be scheduled before native submission");
  }
}

mx::array NativeSparsePrefillAVPlan::execute(
    const mx::array &selected_probabilities,
    const mx::array &selected_values,
    const mx::array &selected_indices,
    const mx::array &selected_valid) {
  validate_input(selected_probabilities, "selected_probabilities",
                 mx::bfloat16,
                 static_cast<size_t>(kHeads) * kSelectedWidth);
  validate_input(selected_values, "selected_values", mx::bfloat16,
                 static_cast<size_t>(kHeads) * kSelectedWidth * kValueDim);
  validate_input(selected_indices, "selected_indices", mx::int32,
                 kSelectedWidth);
  validate_input(selected_valid, "selected_valid", mx::bool_,
                 kSelectedWidth);

  auto &encoder = mx::metal::get_command_encoder(stream_);
  encoder.set_compute_pipeline_state(map_pipeline_);
  encoder.set_input_array(selected_indices, 0);
  encoder.set_input_array(selected_valid, 1);
  encoder.set_output_array(lane_to_selected_, 2);
  encoder.set_bytes(physical_k_, 3);
  encoder.set_bytes(packed_k_, 4);
  encoder.dispatch_threadgroups(MTL::Size(1, 1, 1), MTL::Size(256, 1, 1));
  encoder.barrier();

  map_prepared_ = true;
  return submit_av(selected_probabilities, selected_values);
}

mx::array NativeSparsePrefillAVPlan::execute_prepared(
    const mx::array &selected_probabilities,
    const mx::array &selected_values) {
  if (!map_prepared_) {
    throw std::runtime_error("native prefill AV lane map is not prepared");
  }
  validate_input(selected_probabilities, "selected_probabilities",
                 mx::bfloat16,
                 static_cast<size_t>(kHeads) * kSelectedWidth);
  validate_input(selected_values, "selected_values", mx::bfloat16,
                 static_cast<size_t>(kHeads) * kSelectedWidth * kValueDim);
  return submit_av(selected_probabilities, selected_values);
}

mx::array NativeSparsePrefillAVPlan::submit_av(
    const mx::array &selected_probabilities,
    const mx::array &selected_values) {
  auto &encoder = mx::metal::get_command_encoder(stream_);

  encoder.set_compute_pipeline_state(av_pipeline_);
  encoder.set_input_array(selected_probabilities, 0);
  encoder.set_input_array(selected_values, 1);
  encoder.set_input_array(lane_to_selected_, 2);
  encoder.set_output_array(output_, 3);
  encoder.set_bytes(packed_k_, 4);
  encoder.dispatch_threadgroups(
      MTL::Size(1, 1, kHeads), MTL::Size(32, 4, 1));

  execution_count_++;
  if (buffer_identities() != initial_buffer_identities_) {
    throw std::runtime_error("native prefill AV buffer identity changed");
  }
  return output_;
}

uint64_t NativeSparsePrefillAVPlan::scratch_bytes() const {
  return lane_to_selected_.nbytes();
}

std::vector<uint64_t> NativeSparsePrefillAVPlan::buffer_identities() const {
  return {
      buffer_identity(lane_to_selected_),
      buffer_identity(output_),
  };
}

} // namespace glm53::native_execution
