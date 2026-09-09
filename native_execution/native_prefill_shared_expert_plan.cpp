#include "native_prefill_shared_expert_plan.h"

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
  for (auto extent : shape) elements *= static_cast<size_t>(extent);
  return mx::array(mx::allocator::malloc(elements * mx::size_of(dtype)),
                   shape, dtype);
}
uint64_t buffer_identity(const mx::array &array) {
  return reinterpret_cast<uint64_t>(array.buffer().ptr());
}
} // namespace

NativePrefillSharedExpertPlan::NativePrefillSharedExpertPlan()
    : stream_(mx::default_stream(mx::Device(mx::Device::gpu))),
      activated_(owned_array({kQueryRows, kIntermediateSize}, mx::bfloat16)),
      output_(owned_array({kQueryRows, kHiddenSize}, mx::bfloat16)) {
  auto &device = mx::metal::device(stream_.device);
  auto *library =
      device.get_library("glm53_native_execution", current_binary_dir());
  gate_up_pipeline_ = device.get_kernel(
      "glm53_native_prefill_shared_bm8_gate_up_swiglu", library);
  down_pipeline_ = device.get_kernel(
      "glm53_native_prefill_shared_bm8_down", library);
  initial_buffer_identities_ = buffer_identities();
}

void NativePrefillSharedExpertPlan::validate_input(
    const mx::array &array, const char *name, mx::Dtype dtype,
    size_t elements) const {
  if (array.dtype() != dtype || array.size() != elements) {
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

mx::array NativePrefillSharedExpertPlan::execute(
    const mx::array &hidden, const mx::array &gate_weight,
    const mx::array &gate_scale_inv, const mx::array &up_weight,
    const mx::array &up_scale_inv, const mx::array &down_weight,
    const mx::array &down_scale_inv) {
  validate_input(hidden, "hidden", mx::bfloat16,
                 static_cast<size_t>(kQueryRows) * kHiddenSize);
  validate_input(gate_weight, "gate_weight", mx::uint8,
                 static_cast<size_t>(kIntermediateSize) * kHiddenSize);
  validate_input(up_weight, "up_weight", mx::uint8,
                 static_cast<size_t>(kIntermediateSize) * kHiddenSize);
  validate_input(gate_scale_inv, "gate_scale_inv", mx::float32, 16 * 32);
  validate_input(up_scale_inv, "up_scale_inv", mx::float32, 16 * 32);
  validate_input(down_weight, "down_weight", mx::uint8,
                 static_cast<size_t>(kHiddenSize) * kIntermediateSize);
  validate_input(down_scale_inv, "down_scale_inv", mx::float32, 32 * 16);

  auto &encoder = mx::metal::get_command_encoder(stream_);
  encoder.set_compute_pipeline_state(gate_up_pipeline_);
  encoder.set_input_array(hidden, 0);
  encoder.set_input_array(gate_weight, 1);
  encoder.set_input_array(gate_scale_inv, 2);
  encoder.set_input_array(up_weight, 3);
  encoder.set_input_array(up_scale_inv, 4);
  encoder.set_output_array(activated_, 5);
  encoder.dispatch_threadgroups(
      MTL::Size((kQueryRows / kTileRows) * kIntermediateSize, 1, 1),
      MTL::Size(256, 1, 1));
  encoder.barrier();
  encoder.set_compute_pipeline_state(down_pipeline_);
  encoder.set_input_array(activated_, 0);
  encoder.set_input_array(down_weight, 1);
  encoder.set_input_array(down_scale_inv, 2);
  encoder.set_output_array(output_, 3);
  encoder.dispatch_threadgroups(
      MTL::Size((kQueryRows / kTileRows) * kHiddenSize, 1, 1),
      MTL::Size(256, 1, 1));
  ++execution_count_;
  if (buffer_identities() != initial_buffer_identities_) {
    throw std::runtime_error("native shared expert buffer identity changed");
  }
  return output_;
}

uint64_t NativePrefillSharedExpertPlan::scratch_bytes() const {
  return activated_.nbytes() + output_.nbytes();
}
std::vector<uint64_t> NativePrefillSharedExpertPlan::buffer_identities() const {
  return {buffer_identity(activated_), buffer_identity(output_)};
}

} // namespace glm53::native_execution
