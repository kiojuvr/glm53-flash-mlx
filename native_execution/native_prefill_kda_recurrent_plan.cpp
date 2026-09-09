#include "native_prefill_kda_recurrent_plan.h"

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

uint64_t identity(const mx::array &value) {
  return reinterpret_cast<uint64_t>(value.buffer().ptr());
}
} // namespace

NativePrefillKDARecurrentPlan::NativePrefillKDARecurrentPlan()
    : stream_(mx::default_stream(mx::Device(mx::Device::gpu))),
      output_(owned_array(
          {1, kQueryRows, kHeads, kValueDim}, mx::bfloat16)),
      next_state_(owned_array(
          {1, kHeads, kValueDim, kKeyDim}, mx::float32)) {
  auto &device = mx::metal::device(stream_.device);
  auto *library = device.get_library(
      "glm53_native_execution", current_binary_dir());
  pipeline_ = device.get_kernel(
      "glm53_native_prefill_kda_recurrent_r4_bfloat16", library);
  initial_buffer_identities_ = buffer_identities();
}

void NativePrefillKDARecurrentPlan::validate(
    const mx::array &value, const char *name, mx::Dtype dtype,
    size_t elements) const {
  if (value.dtype() != dtype || value.size() != elements) {
    throw std::invalid_argument(std::string(name) + " shape/dtype mismatch");
  }
  if (!value.flags().row_contiguous) {
    throw std::invalid_argument(std::string(name) + " must be row-contiguous");
  }
  if (value.status() == mx::array::Status::unscheduled) {
    throw std::invalid_argument(
        std::string(name) + " must be scheduled before native submission");
  }
}

std::vector<mx::array> NativePrefillKDARecurrentPlan::execute(
    const mx::array &q, const mx::array &k, const mx::array &v,
    const mx::array &g, const mx::array &beta, const mx::array &state,
    const mx::array &mask, bool has_mask) {
  constexpr size_t sequence =
      static_cast<size_t>(kQueryRows) * kHeads * kKeyDim;
  constexpr size_t state_elements =
      static_cast<size_t>(kHeads) * kValueDim * kKeyDim;
  validate(q, "q", mx::bfloat16, sequence);
  validate(k, "k", mx::bfloat16, sequence);
  validate(v, "v", mx::bfloat16, sequence);
  validate(g, "g", mx::float32, sequence);
  validate(beta, "beta", mx::bfloat16,
           static_cast<size_t>(kQueryRows) * kHeads);
  validate(state, "state", mx::float32, state_elements);
  validate(mask, "mask", mx::bool_, has_mask ? kQueryRows : 1);

  auto &encoder = mx::metal::get_command_encoder(stream_);
  encoder.set_compute_pipeline_state(pipeline_);
  encoder.set_input_array(q, 0);
  encoder.set_input_array(k, 1);
  encoder.set_input_array(v, 2);
  encoder.set_input_array(g, 3);
  encoder.set_input_array(beta, 4);
  encoder.set_input_array(state, 5);
  encoder.set_input_array(mask, 6);
  encoder.set_output_array(output_, 7);
  encoder.set_output_array(next_state_, 8);
  encoder.set_bytes(has_mask, 9);
  encoder.dispatch_threadgroups(
      MTL::Size(1, kValueDim / kRowBlock, kHeads),
      MTL::Size(32, 1, 1));
  encoder.barrier();

  ++execution_count_;
  if (!buffer_identities_stable()) {
    throw std::runtime_error("native KDA recurrent arena identity changed");
  }
  return {output_, next_state_};
}

uint64_t NativePrefillKDARecurrentPlan::arena_bytes() const {
  return output_.nbytes() + next_state_.nbytes();
}

std::vector<uint64_t>
NativePrefillKDARecurrentPlan::buffer_identities() const {
  return {identity(output_), identity(next_state_)};
}

bool NativePrefillKDARecurrentPlan::buffer_identities_stable() const {
  return buffer_identities() == initial_buffer_identities_;
}

} // namespace glm53::native_execution
