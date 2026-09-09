#include "native_prefill_moe_gate_up_plan.h"

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
  return mx::array(
      mx::allocator::malloc(elements * mx::size_of(dtype)), shape, dtype);
}

uint64_t buffer_identity(const mx::array &array) {
  return reinterpret_cast<uint64_t>(array.buffer().ptr());
}

} // namespace

NativePrefillMoEGateUpPlan::NativePrefillMoEGateUpPlan(int expert_count)
    : expert_count_(expert_count),
      stream_(mx::default_stream(mx::Device(mx::Device::gpu))),
      route_plan_(expert_count),
      activated_(owned_array({kRouteRows, kIntermediateSize}, mx::bfloat16)),
      routed_down_(owned_array({kRouteRows, kHiddenSize}, mx::bfloat16)),
      routed_output_(owned_array({kQueryRows, kHiddenSize}, mx::bfloat16)) {
  auto &device = mx::metal::device(stream_.device);
  auto *library =
      device.get_library("glm53_native_execution", current_binary_dir());
  gate_up_pipeline_ = device.get_kernel(
      "glm53_native_prefill_moe_bm8_gate_up_swiglu", library);
  down_pipeline_ = device.get_kernel(
      "glm53_native_prefill_moe_bm8_down", library);
  reduce_pipeline_ = device.get_kernel(
      "glm53_native_prefill_moe_direct_order_reduce", library);
  initial_buffer_identities_ = buffer_identities();
}

void NativePrefillMoEGateUpPlan::validate_input(
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

mx::array NativePrefillMoEGateUpPlan::execute(
    const mx::array &hidden, const mx::array &expert_ids,
    const mx::array &scores, const mx::array &gate_up_weight,
    const mx::array &gate_up_scale_inv) {
  encode_ingress(hidden, expert_ids, scores, gate_up_weight,
                 gate_up_scale_inv);
  ++execution_count_;
  if (buffer_identities() != initial_buffer_identities_) {
    throw std::runtime_error("native prefill MoE gate/up buffer identity changed");
  }
  return activated_;
}

void NativePrefillMoEGateUpPlan::encode_ingress(
    const mx::array &hidden, const mx::array &expert_ids,
    const mx::array &scores, const mx::array &gate_up_weight,
    const mx::array &gate_up_scale_inv) {
  validate_input(hidden, "hidden", mx::bfloat16,
                 static_cast<size_t>(kQueryRows) * kHiddenSize);
  validate_input(expert_ids, "expert_ids", mx::uint32, kRouteRows);
  validate_input(scores, "scores", mx::float32, kRouteRows);
  validate_input(gate_up_weight, "gate_up_weight", mx::uint8,
                 static_cast<size_t>(expert_count_) * 2 * kIntermediateSize *
                     kHiddenSize);
  validate_input(gate_up_scale_inv, "gate_up_scale_inv", mx::float32,
                 static_cast<size_t>(expert_count_) * kScaleRows * kScaleCols);

  // This call only encodes into the current MLX command buffer. Its owned
  // metadata is consumed below without synchronization or Python inspection.
  route_plan_.execute(expert_ids, scores);
  auto &encoder = mx::metal::get_command_encoder(stream_);
  encoder.barrier();
  encoder.set_compute_pipeline_state(gate_up_pipeline_);
  encoder.set_input_array(hidden, 0);
  encoder.set_input_array(route_plan_.sorted_route_order(), 1);
  encoder.set_input_array(route_plan_.tile_experts(), 2);
  encoder.set_input_array(route_plan_.tile_starts(), 3);
  encoder.set_input_array(route_plan_.tile_lengths(), 4);
  encoder.set_input_array(route_plan_.expert_offsets(), 5);
  encoder.set_input_array(gate_up_weight, 6);
  encoder.set_input_array(gate_up_scale_inv, 7);
  encoder.set_output_array(activated_, 8);
  const int descriptor_capacity = route_plan_.descriptor_capacity();
  encoder.dispatch_threadgroups(
      MTL::Size(static_cast<NS::UInteger>(descriptor_capacity) *
                    kIntermediateSize,
                1, 1),
      MTL::Size(256, 1, 1));

}

mx::array NativePrefillMoEGateUpPlan::execute_routed(
    const mx::array &hidden, const mx::array &expert_ids,
    const mx::array &scores, const mx::array &gate_up_weight,
    const mx::array &gate_up_scale_inv, const mx::array &down_weight,
    const mx::array &down_scale_inv) {
  encode_ingress(hidden, expert_ids, scores, gate_up_weight,
                 gate_up_scale_inv);
  validate_input(down_weight, "down_weight", mx::uint8,
                 static_cast<size_t>(expert_count_) * kHiddenSize *
                     kIntermediateSize);
  validate_input(down_scale_inv, "down_scale_inv", mx::float32,
                 static_cast<size_t>(expert_count_) * kScaleCols * 16);
  auto &encoder = mx::metal::get_command_encoder(stream_);
  encoder.barrier();
  encoder.set_compute_pipeline_state(down_pipeline_);
  encoder.set_input_array(activated_, 0);
  encoder.set_input_array(route_plan_.tile_experts(), 1);
  encoder.set_input_array(route_plan_.tile_starts(), 2);
  encoder.set_input_array(route_plan_.tile_lengths(), 3);
  encoder.set_input_array(route_plan_.expert_offsets(), 4);
  encoder.set_input_array(down_weight, 5);
  encoder.set_input_array(down_scale_inv, 6);
  encoder.set_output_array(routed_down_, 7);
  encoder.dispatch_threadgroups(
      MTL::Size(static_cast<NS::UInteger>(route_plan_.descriptor_capacity()) *
                    kHiddenSize,
                1, 1),
      MTL::Size(256, 1, 1));
  encoder.barrier();
  encoder.set_compute_pipeline_state(reduce_pipeline_);
  encoder.set_input_array(routed_down_, 0);
  encoder.set_input_array(expert_ids, 1);
  encoder.set_input_array(scores, 2);
  encoder.set_input_array(route_plan_.inverse_route_order(), 3);
  encoder.set_output_array(routed_output_, 4);
  encoder.dispatch_threadgroups(
      MTL::Size(kQueryRows, 1, 1), MTL::Size(256, 1, 1));
  ++execution_count_;
  if (buffer_identities() != initial_buffer_identities_) {
    throw std::runtime_error("native prefill routed MoE buffer identity changed");
  }
  return routed_output_;
}

uint64_t NativePrefillMoEGateUpPlan::scratch_bytes() const {
  return route_plan_.scratch_bytes() + activated_.nbytes() +
      routed_down_.nbytes() + routed_output_.nbytes();
}

std::vector<uint64_t> NativePrefillMoEGateUpPlan::buffer_identities() const {
  auto identities = route_plan_.buffer_identities();
  identities.push_back(buffer_identity(activated_));
  identities.push_back(buffer_identity(routed_down_));
  identities.push_back(buffer_identity(routed_output_));
  return identities;
}

} // namespace glm53::native_execution
