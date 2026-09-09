#include "native_prefill_45_layer_submission_plan.h"

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

mx::array owned_hidden() {
  constexpr size_t elements = 256u * 4096u;
  return mx::array(
      mx::allocator::malloc(elements * mx::size_of(mx::bfloat16)),
      {256, 4096}, mx::bfloat16);
}

uint64_t identity(const mx::array &value) {
  return reinterpret_cast<uint64_t>(value.buffer().ptr());
}
} // namespace

NativePrefill45LayerSubmissionPlan::NativePrefill45LayerSubmissionPlan()
    : stream_(mx::default_stream(mx::Device(mx::Device::gpu))),
      hidden_ping_(owned_hidden()), hidden_pong_(owned_hidden()) {
  auto &device = mx::metal::device(stream_.device);
  auto *library = device.get_library(
      "glm53_native_execution", current_binary_dir());
  copy_pipeline_ = device.get_kernel(
      "glm53_native_prefill_copy_bfloat16", library);
  initial_buffer_identities_ = buffer_identities();
}

void NativePrefill45LayerSubmissionPlan::validate_hidden(
    const mx::array &hidden) const {
  if (hidden.dtype() != mx::bfloat16 || hidden.ndim() != 2 ||
      hidden.shape(0) != kQueryRows || hidden.shape(1) != kHiddenSize) {
    throw std::invalid_argument(
        "45-layer native prefill hidden must be [256, 4096] bfloat16");
  }
  if (!hidden.flags().row_contiguous) {
    throw std::invalid_argument(
        "45-layer native prefill hidden must be row-contiguous");
  }
  if (hidden.status() == mx::array::Status::unscheduled) {
    throw std::invalid_argument(
        "45-layer native prefill hidden must be scheduled before submission");
  }
}

mx::array NativePrefill45LayerSubmissionPlan::encode_layer(
    const mx::array &hidden, int layer_index) {
  if (layer_index < 0 || layer_index >= kLayerCount) {
    throw std::out_of_range("native prefill layer index must be in [0, 45)");
  }
  auto &encoder = mx::metal::get_command_encoder(stream_);
  auto &destination = (layer_index % 2 == 0) ? hidden_ping_ : hidden_pong_;
  constexpr uint32_t elements = kQueryRows * kHiddenSize;
  constexpr uint32_t threads = 256;
  constexpr uint32_t groups = (elements + threads - 1) / threads;
  encoder.set_compute_pipeline_state(copy_pipeline_);
  encoder.set_input_array(hidden, 0);
  encoder.set_output_array(destination, 1);
  encoder.set_bytes(elements, 2);
  encoder.dispatch_threadgroups(
      MTL::Size(groups, 1, 1), MTL::Size(threads, 1, 1));
  encoder.barrier();
  return destination;
}

mx::array NativePrefill45LayerSubmissionPlan::execute_layer(
    const mx::array &hidden, int layer_index) {
  validate_hidden(hidden);
  auto output = encode_layer(hidden, layer_index);
  ++layer_execution_count_;
  return output;
}

mx::array NativePrefill45LayerSubmissionPlan::execute_all(
    const mx::array &hidden) {
  validate_hidden(hidden);
  mx::array current = hidden;
  for (int layer = 0; layer < kLayerCount; ++layer) {
    current = encode_layer(current, layer);
  }
  ++all_execution_count_;
  if (!buffer_identities_stable()) {
    throw std::runtime_error("45-layer native prefill arena identity changed");
  }
  return current;
}

uint64_t NativePrefill45LayerSubmissionPlan::arena_bytes() const {
  return hidden_ping_.nbytes() + hidden_pong_.nbytes();
}

std::vector<uint64_t>
NativePrefill45LayerSubmissionPlan::buffer_identities() const {
  return {identity(hidden_ping_), identity(hidden_pong_)};
}

bool NativePrefill45LayerSubmissionPlan::buffer_identities_stable() const {
  return buffer_identities() == initial_buffer_identities_;
}

std::vector<std::string>
NativePrefill45LayerSubmissionPlan::attention_types() const {
  std::vector<std::string> result;
  result.reserve(kLayerCount);
  for (int layer = 0; layer < kLayerCount; ++layer) {
    result.push_back(layer % 4 == 3 ? "dsa" : "kda");
  }
  return result;
}

std::vector<std::string>
NativePrefill45LayerSubmissionPlan::ffn_types() const {
  std::vector<std::string> result;
  result.reserve(kLayerCount);
  for (int layer = 0; layer < kLayerCount; ++layer) {
    result.push_back(layer < 3 ? "dense" : "moe");
  }
  return result;
}

} // namespace glm53::native_execution
