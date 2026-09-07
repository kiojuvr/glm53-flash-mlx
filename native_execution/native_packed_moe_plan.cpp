#include "native_packed_moe_plan.h"

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

int checked_dimension(int value, const char *name) {
  if (value <= 0 || value % 128 != 0) {
    throw std::invalid_argument(std::string(name) +
                                " must be positive and block-128 aligned");
  }
  return value;
}

} // namespace

NativePackedMoEDecodePlan::NativePackedMoEDecodePlan(
    int hidden_size, int intermediate_size, int shared_intermediate_size,
    int expert_count, int swiglu_limit)
    : hidden_size_(checked_dimension(hidden_size, "hidden_size")),
      intermediate_size_(
          checked_dimension(intermediate_size, "intermediate_size")),
      shared_intermediate_size_(checked_dimension(shared_intermediate_size,
                                                   "shared_intermediate_size")),
      expert_count_(expert_count), swiglu_limit_(swiglu_limit),
      hidden_scale_rows_(hidden_size_ / kBlockSize),
      intermediate_scale_rows_(intermediate_size_ / kBlockSize),
      shared_scale_rows_(shared_intermediate_size_ / kBlockSize),
      stream_(mx::default_stream(mx::Device(mx::Device::gpu))),
      routed_hidden_(
          owned_array({kTopK, intermediate_size_}, mx::bfloat16)),
      routed_down_(owned_array({kTopK, hidden_size_}, mx::bfloat16)),
      routed_output_(owned_array({hidden_size_}, mx::bfloat16)),
      shared_hidden_(
          owned_array({shared_intermediate_size_}, mx::bfloat16)),
      shared_down_(owned_array({hidden_size_}, mx::bfloat16)),
      output_(owned_array({1, 1, hidden_size_}, mx::bfloat16)) {
  if (expert_count_ < kTopK) {
    throw std::invalid_argument("expert_count must cover selected top-8");
  }
  if (swiglu_limit_ <= 0) {
    throw std::invalid_argument("SwiGLU limit must be a positive integer");
  }
  initial_buffer_identities_ = buffer_identities();
}

void NativePackedMoEDecodePlan::validate_input(
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
    throw std::invalid_argument(std::string(name) +
                                " must be scheduled before native submission");
  }
}

mx::array NativePackedMoEDecodePlan::execute(
    const mx::array &x, const mx::array &expert_ids, const mx::array &scores,
    const mx::array &gate_up_weight, const mx::array &gate_up_scale_inv,
    const mx::array &down_weight, const mx::array &down_scale_inv,
    const mx::array &shared_gate_weight,
    const mx::array &shared_gate_scale_inv,
    const mx::array &shared_up_weight,
    const mx::array &shared_up_scale_inv,
    const mx::array &shared_down_weight,
    const mx::array &shared_down_scale_inv) {
  validate_input(x, "x", mx::bfloat16, hidden_size_);
  validate_input(expert_ids, "expert_ids", mx::uint32, kTopK);
  validate_input(scores, "scores", mx::float32, kTopK);
  validate_input(gate_up_weight, "gate_up_weight", mx::uint8,
                 static_cast<size_t>(expert_count_) * 2 * intermediate_size_ *
                     hidden_size_);
  validate_input(gate_up_scale_inv, "gate_up_scale_inv", mx::float32,
                 static_cast<size_t>(expert_count_) * 2 *
                     intermediate_scale_rows_ * hidden_scale_rows_);
  validate_input(down_weight, "down_weight", mx::uint8,
                 static_cast<size_t>(expert_count_) * hidden_size_ *
                     intermediate_size_);
  validate_input(down_scale_inv, "down_scale_inv", mx::float32,
                 static_cast<size_t>(expert_count_) * hidden_scale_rows_ *
                     intermediate_scale_rows_);
  validate_input(shared_gate_weight, "shared_gate_weight", mx::uint8,
                 static_cast<size_t>(shared_intermediate_size_) * hidden_size_);
  validate_input(shared_gate_scale_inv, "shared_gate_scale_inv", mx::float32,
                 static_cast<size_t>(shared_scale_rows_) * hidden_scale_rows_);
  validate_input(shared_up_weight, "shared_up_weight", mx::uint8,
                 static_cast<size_t>(shared_intermediate_size_) * hidden_size_);
  validate_input(shared_up_scale_inv, "shared_up_scale_inv", mx::float32,
                 static_cast<size_t>(shared_scale_rows_) * hidden_scale_rows_);
  validate_input(shared_down_weight, "shared_down_weight", mx::uint8,
                 static_cast<size_t>(hidden_size_) * shared_intermediate_size_);
  validate_input(shared_down_scale_inv, "shared_down_scale_inv", mx::float32,
                 static_cast<size_t>(hidden_scale_rows_) * shared_scale_rows_);

  auto &device = mx::metal::device(stream_.device);
  auto *library =
      device.get_library("glm53_native_execution", current_binary_dir());
  const char *routed_gate_up_name = uses_shape_specialized_routed_gate_up()
      ? "glm53_native_glm53_packed_selected8_gate_up_swiglu"
      : "glm53_native_packed_selected8_gate_up_swiglu";
  auto *routed_gate_up = device.get_kernel(routed_gate_up_name, library);
  auto *routed_down =
      device.get_kernel("glm53_native_packed_selected8_down", library);
  auto *aggregate = device.get_kernel(
      "glm53_native_packed_selected8_weighted_reduction", library);
  auto *shared_gate_up = device.get_kernel(
      "glm53_native_shared_gate_up_swiglu", library);
  auto *shared_down =
      device.get_kernel("glm53_native_shared_down", library);
  auto *add = device.get_kernel("glm53_native_add_routed_shared", library);
  auto &encoder = mx::metal::get_command_encoder(stream_);

  encoder.set_compute_pipeline_state(routed_gate_up);
  encoder.set_input_array(x, 0);
  encoder.set_input_array(expert_ids, 1);
  encoder.set_input_array(gate_up_weight, 2);
  encoder.set_input_array(gate_up_scale_inv, 3);
  encoder.set_output_array(routed_hidden_, 4);
  encoder.set_bytes(hidden_size_, 5);
  encoder.set_bytes(intermediate_size_, 6);
  encoder.set_bytes(intermediate_scale_rows_, 7);
  encoder.set_bytes(hidden_scale_rows_, 8);
  encoder.set_bytes(swiglu_limit_, 9);
  encoder.dispatch_threadgroups(
      MTL::Size(kTopK * intermediate_size_, 1, 1),
      MTL::Size(kThreads, 1, 1));
  encoder.barrier();

  encoder.set_compute_pipeline_state(routed_down);
  encoder.set_input_array(routed_hidden_, 0);
  encoder.set_input_array(expert_ids, 1);
  encoder.set_input_array(down_weight, 2);
  encoder.set_input_array(down_scale_inv, 3);
  encoder.set_output_array(routed_down_, 4);
  encoder.set_bytes(intermediate_size_, 5);
  encoder.set_bytes(hidden_size_, 6);
  encoder.set_bytes(intermediate_scale_rows_, 7);
  encoder.dispatch_threadgroups(MTL::Size(kTopK * hidden_size_, 1, 1),
                                MTL::Size(kThreads, 1, 1));
  encoder.barrier();

  encoder.set_compute_pipeline_state(aggregate);
  encoder.set_input_array(routed_down_, 0);
  encoder.set_input_array(scores, 1);
  encoder.set_output_array(routed_output_, 2);
  encoder.set_bytes(hidden_size_, 3);
  encoder.dispatch_threads(MTL::Size(hidden_size_, 1, 1),
                           MTL::Size(kThreads, 1, 1));
  encoder.barrier();

  encoder.set_compute_pipeline_state(shared_gate_up);
  encoder.set_input_array(x, 0);
  encoder.set_input_array(shared_gate_weight, 1);
  encoder.set_input_array(shared_gate_scale_inv, 2);
  encoder.set_input_array(shared_up_weight, 3);
  encoder.set_input_array(shared_up_scale_inv, 4);
  encoder.set_output_array(shared_hidden_, 5);
  encoder.set_bytes(hidden_size_, 6);
  encoder.set_bytes(shared_intermediate_size_, 7);
  encoder.set_bytes(hidden_scale_rows_, 8);
  encoder.set_bytes(swiglu_limit_, 9);
  encoder.dispatch_threadgroups(MTL::Size(shared_intermediate_size_, 1, 1),
                                MTL::Size(kThreads, 1, 1));
  encoder.barrier();

  encoder.set_compute_pipeline_state(shared_down);
  encoder.set_input_array(shared_hidden_, 0);
  encoder.set_input_array(shared_down_weight, 1);
  encoder.set_input_array(shared_down_scale_inv, 2);
  encoder.set_output_array(shared_down_, 3);
  encoder.set_bytes(shared_intermediate_size_, 4);
  encoder.set_bytes(hidden_size_, 5);
  encoder.set_bytes(shared_scale_rows_, 6);
  encoder.dispatch_threadgroups(MTL::Size(hidden_size_, 1, 1),
                                MTL::Size(kThreads, 1, 1));
  encoder.barrier();

  encoder.set_compute_pipeline_state(add);
  encoder.set_input_array(routed_output_, 0);
  encoder.set_input_array(shared_down_, 1);
  encoder.set_output_array(output_, 2);
  encoder.set_bytes(hidden_size_, 3);
  encoder.dispatch_threads(MTL::Size(hidden_size_, 1, 1),
                           MTL::Size(kThreads, 1, 1));

  execution_count_++;
  if (buffer_identities() != initial_buffer_identities_) {
    throw std::runtime_error(
        "native packed MoE plan buffer identity changed during execute");
  }
  return output_;
}

uint64_t NativePackedMoEDecodePlan::scratch_bytes() const {
  return routed_hidden_.nbytes() + routed_down_.nbytes() +
      routed_output_.nbytes() + shared_hidden_.nbytes() +
      shared_down_.nbytes() + output_.nbytes();
}

std::vector<uint64_t> NativePackedMoEDecodePlan::buffer_identities() const {
  return {buffer_identity(routed_hidden_), buffer_identity(routed_down_),
          buffer_identity(routed_output_), buffer_identity(shared_hidden_),
          buffer_identity(shared_down_), buffer_identity(output_)};
}

} // namespace glm53::native_execution
