#include "native_prefill_moe_plan.h"

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
  for (auto extent : shape) elements *= static_cast<size_t>(extent);
  return mx::array(mx::allocator::malloc(elements * mx::size_of(dtype)),
                   shape, dtype);
}
uint64_t identity(const mx::array &array) {
  return reinterpret_cast<uint64_t>(array.buffer().ptr());
}
} // namespace

NativePrefillMoEPlan::NativePrefillMoEPlan(int expert_count)
    : stream_(mx::default_stream(mx::Device(mx::Device::gpu))),
      routed_plan_(expert_count), shared_plan_(),
      output_(owned_array({kQueryRows, kHiddenSize}, mx::bfloat16)) {
  auto &device = mx::metal::device(stream_.device);
  auto *library =
      device.get_library("glm53_native_execution", current_binary_dir());
  add_pipeline_ = device.get_kernel("glm53_native_add_routed_shared", library);
  initial_buffer_identities_ = buffer_identities();
}

mx::array NativePrefillMoEPlan::execute(
    const mx::array &hidden, const mx::array &expert_ids,
    const mx::array &scores, const mx::array &gate_up_weight,
    const mx::array &gate_up_scale_inv, const mx::array &down_weight,
    const mx::array &down_scale_inv, const mx::array &shared_gate_weight,
    const mx::array &shared_gate_scale_inv,
    const mx::array &shared_up_weight, const mx::array &shared_up_scale_inv,
    const mx::array &shared_down_weight,
    const mx::array &shared_down_scale_inv) {
  auto routed = routed_plan_.execute_routed(
      hidden, expert_ids, scores, gate_up_weight, gate_up_scale_inv,
      down_weight, down_scale_inv);
  auto shared = shared_plan_.execute(
      hidden, shared_gate_weight, shared_gate_scale_inv, shared_up_weight,
      shared_up_scale_inv, shared_down_weight, shared_down_scale_inv);
  return finish(routed, shared);
}

mx::array NativePrefillMoEPlan::execute_fused_down_reduce(
    const mx::array &hidden, const mx::array &expert_ids,
    const mx::array &scores, const mx::array &gate_up_weight,
    const mx::array &gate_up_scale_inv, const mx::array &down_weight,
    const mx::array &down_scale_inv, const mx::array &shared_gate_weight,
    const mx::array &shared_gate_scale_inv,
    const mx::array &shared_up_weight, const mx::array &shared_up_scale_inv,
    const mx::array &shared_down_weight,
    const mx::array &shared_down_scale_inv) {
  auto routed = routed_plan_.execute_routed_fused(
      hidden, expert_ids, scores, gate_up_weight, gate_up_scale_inv,
      down_weight, down_scale_inv);
  auto shared = shared_plan_.execute(
      hidden, shared_gate_weight, shared_gate_scale_inv, shared_up_weight,
      shared_up_scale_inv, shared_down_weight, shared_down_scale_inv);
  return finish(routed, shared);
}

mx::array NativePrefillMoEPlan::execute_indirect(
    const mx::array &hidden, const mx::array &expert_ids,
    const mx::array &scores, const mx::array &gate_up_weight,
    const mx::array &gate_up_scale_inv, const mx::array &down_weight,
    const mx::array &down_scale_inv, const mx::array &shared_gate_weight,
    const mx::array &shared_gate_scale_inv,
    const mx::array &shared_up_weight, const mx::array &shared_up_scale_inv,
    const mx::array &shared_down_weight,
    const mx::array &shared_down_scale_inv) {
  auto routed = routed_plan_.execute_routed_indirect(
      hidden, expert_ids, scores, gate_up_weight, gate_up_scale_inv,
      down_weight, down_scale_inv);
  auto shared = shared_plan_.execute(
      hidden, shared_gate_weight, shared_gate_scale_inv, shared_up_weight,
      shared_up_scale_inv, shared_down_weight, shared_down_scale_inv);
  return finish(routed, shared);
}

mx::array NativePrefillMoEPlan::finish(
    const mx::array &routed, const mx::array &shared) {
  auto &encoder = mx::metal::get_command_encoder(stream_);
  encoder.barrier();
  encoder.set_compute_pipeline_state(add_pipeline_);
  encoder.set_input_array(routed, 0);
  encoder.set_input_array(shared, 1);
  encoder.set_output_array(output_, 2);
  const int elements = kQueryRows * kHiddenSize;
  encoder.set_bytes(elements, 3);
  encoder.dispatch_threads(MTL::Size(elements, 1, 1), MTL::Size(256, 1, 1));
  ++execution_count_;
  if (buffer_identities() != initial_buffer_identities_) {
    throw std::runtime_error("native prefill MoE buffer identity changed");
  }
  return output_;
}

uint64_t NativePrefillMoEPlan::scratch_bytes() const {
  return routed_plan_.scratch_bytes() + shared_plan_.scratch_bytes() +
      output_.nbytes();
}
std::vector<uint64_t> NativePrefillMoEPlan::buffer_identities() const {
  auto values = routed_plan_.buffer_identities();
  auto shared = shared_plan_.buffer_identities();
  values.insert(values.end(), shared.begin(), shared.end());
  values.push_back(identity(output_));
  return values;
}
} // namespace glm53::native_execution
