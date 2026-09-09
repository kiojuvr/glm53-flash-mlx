#include "native_prefill_dsa_output_plan.h"

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

NativePrefillDSAOutputPlan::NativePrefillDSAOutputPlan()
    : stream_(mx::default_stream(mx::Device(mx::Device::gpu))),
      row_major_(owned_array({kQueryRows, kInputSize}, mx::bfloat16)),
      output_(owned_array({kQueryRows, kHiddenSize}, mx::bfloat16)),
      hc_output_(owned_array(
          {kQueryRows, 4, kHiddenSize}, mx::bfloat16)) {
  auto &device = mx::metal::device(stream_.device);
  auto *library = device.get_library(
      "glm53_native_execution", current_binary_dir());
  transpose_pipeline_ = device.get_kernel(
      "glm53_native_prefill_dsa_head_to_row_bfloat16", library);
  projection_pipeline_ = device.get_kernel(
      "glm53_native_prefill_dsa_o_proj_e4m3", library);
  hc_expand_pipeline_ = device.get_kernel(
      "glm53_native_prefill_post_attention_hc_expand", library);
  initial_buffer_identities_ = buffer_identities();
}

void NativePrefillDSAOutputPlan::validate(
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
        std::string(name) + " must be scheduled before submission");
  }
}

mx::array NativePrefillDSAOutputPlan::execute(
    const mx::array &head_major, const mx::array &weight,
    const mx::array &scale_inv) {
  encode_projection(head_major, weight, scale_inv);
  ++execution_count_;
  if (buffer_identities() != initial_buffer_identities_) {
    throw std::runtime_error("native DSA output buffer identity changed");
  }
  return output_;
}

void NativePrefillDSAOutputPlan::encode_projection(
    const mx::array &head_major, const mx::array &weight,
    const mx::array &scale_inv) {
  validate(head_major, "head_major", mx::bfloat16,
           static_cast<size_t>(kHeads) * kQueryRows * kValueDim);
  validate(weight, "weight", mx::uint8,
           static_cast<size_t>(kHiddenSize) * kInputSize);
  validate(scale_inv, "scale_inv", mx::float32,
           static_cast<size_t>(kHiddenSize / 128) * (kInputSize / 128));

  auto &encoder = mx::metal::get_command_encoder(stream_);
  encoder.set_compute_pipeline_state(transpose_pipeline_);
  encoder.set_input_array(head_major, 0);
  encoder.set_output_array(row_major_, 1);
  constexpr uint32_t elements = kHeads * kQueryRows * kValueDim;
  encoder.set_bytes(elements, 2);
  encoder.dispatch_threads(MTL::Size(elements, 1, 1), MTL::Size(256, 1, 1));
  encoder.barrier();
  encoder.set_compute_pipeline_state(projection_pipeline_);
  encoder.set_input_array(row_major_, 0);
  encoder.set_input_array(weight, 1);
  encoder.set_input_array(scale_inv, 2);
  encoder.set_output_array(output_, 3);
  constexpr int tile_rows = 8;
  constexpr int groups = (kQueryRows / tile_rows) * kHiddenSize;
  encoder.dispatch_threadgroups(
      MTL::Size(groups, 1, 1), MTL::Size(256, 1, 1));
}

mx::array NativePrefillDSAOutputPlan::execute_hc(
    const mx::array &head_major, const mx::array &weight,
    const mx::array &scale_inv, const mx::array &residual,
    const mx::array &post, const mx::array &comb) {
  encode_projection(head_major, weight, scale_inv);
  validate(residual, "residual", mx::bfloat16,
           static_cast<size_t>(kQueryRows) * 4 * kHiddenSize);
  validate(post, "post", mx::float32,
           static_cast<size_t>(kQueryRows) * 4);
  validate(comb, "comb", mx::float32,
           static_cast<size_t>(kQueryRows) * 4 * 4);
  auto &encoder = mx::metal::get_command_encoder(stream_);
  encoder.barrier();
  encoder.set_compute_pipeline_state(hc_expand_pipeline_);
  encoder.set_input_array(output_, 0);
  encoder.set_input_array(residual, 1);
  encoder.set_input_array(post, 2);
  encoder.set_input_array(comb, 3);
  encoder.set_output_array(hc_output_, 4);
  constexpr uint32_t elements = kQueryRows * 4 * kHiddenSize;
  encoder.set_bytes(elements, 5);
  encoder.dispatch_threads(MTL::Size(elements, 1, 1), MTL::Size(256, 1, 1));
  ++execution_count_;
  if (buffer_identities() != initial_buffer_identities_) {
    throw std::runtime_error("native post-attention HC buffer identity changed");
  }
  return hc_output_;
}

std::vector<uint64_t> NativePrefillDSAOutputPlan::buffer_identities() const {
  return {identity(row_major_), identity(output_), identity(hc_output_)};
}

} // namespace glm53::native_execution
