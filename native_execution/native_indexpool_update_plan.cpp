#include "native_indexpool_update_plan.h"

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

} // namespace

NativeIndexPoolUpdateSelectionPlan::NativeIndexPoolUpdateSelectionPlan(
    int physical_pool_rows, float softmax_scale)
    : physical_pool_rows_(physical_pool_rows),
      stream_(mx::default_stream(mx::Device(mx::Device::gpu))),
      score_plan_("decode", 1, physical_pool_rows, softmax_scale),
      raw_keys_a_(owned_array({1, kRawWindow, kHeadDim}, mx::bfloat16)),
      raw_keys_b_(owned_array({1, kRawWindow, kHeadDim}, mx::bfloat16)),
      raw_gates_a_(owned_array({1, kRawWindow, kHeadDim}, mx::bfloat16)),
      raw_gates_b_(owned_array({1, kRawWindow, kHeadDim}, mx::bfloat16)),
      raw_valid_a_(owned_array({1, kRawWindow}, mx::bool_)),
      raw_valid_b_(owned_array({1, kRawWindow}, mx::bool_)),
      raw_positions_a_(owned_array({1, kRawWindow}, mx::int64)),
      raw_positions_b_(owned_array({1, kRawWindow}, mx::int64)),
      pool_logits_(owned_array({kHeadDim, kIndexKPool}, mx::bfloat16)),
      pool_probabilities_(
          owned_array({kHeadDim, kIndexKPool}, mx::bfloat16)),
      current_raw_keys_(&raw_keys_b_), current_raw_gates_(&raw_gates_b_),
      current_raw_valid_(&raw_valid_b_),
      current_raw_positions_(&raw_positions_b_) {
  initial_buffer_identities_ = buffer_identities();
}

void NativeIndexPoolUpdateSelectionPlan::validate_input(
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

std::vector<mx::array> NativeIndexPoolUpdateSelectionPlan::execute(
    const mx::array &key, const mx::array &gate,
    const mx::array &current_valid, const mx::array &query,
    const mx::array &mixture_weights, mx::array &pool_keys,
    mx::array &pool_indices, mx::array &pool_valid,
    const mx::array &raw_keys, const mx::array &raw_gates,
    const mx::array &raw_valid, const mx::array &raw_positions,
    const mx::array &compress_ape, int previous_total_tokens) {
  validate_input(key, "key", mx::bfloat16, kHeadDim);
  validate_input(gate, "gate", mx::bfloat16, kHeadDim);
  validate_input(current_valid, "current_valid", mx::bool_, 1);
  validate_input(query, "query", mx::bfloat16, 32 * kHeadDim);
  validate_input(mixture_weights, "mixture_weights", mx::bfloat16, 32);
  validate_input(pool_keys, "pool_keys", mx::bfloat16,
                 static_cast<size_t>(physical_pool_rows_) * kHeadDim);
  validate_input(pool_indices, "pool_indices", mx::int64,
                 static_cast<size_t>(physical_pool_rows_) * kIndexKPool);
  validate_input(pool_valid, "pool_valid", mx::bool_, physical_pool_rows_);
  validate_input(raw_keys, "raw_keys", mx::bfloat16,
                 kRawWindow * kHeadDim);
  validate_input(raw_gates, "raw_gates", mx::bfloat16,
                 kRawWindow * kHeadDim);
  validate_input(raw_valid, "raw_valid", mx::bool_, kRawWindow);
  validate_input(raw_positions, "raw_positions", mx::int64, kRawWindow);
  validate_input(compress_ape, "compress_ape", mx::bfloat16,
                 kIndexKPool * kHeadDim);
  if (previous_total_tokens < 2048) {
    throw std::out_of_range(
        "native IndexPool update begins only at the sparse decode boundary");
  }
  const int pool_row = previous_total_tokens / kIndexKPool;
  if (pool_row >= physical_pool_rows_) {
    throw std::out_of_range("native IndexPool pool row exceeds capacity");
  }

  // Never overwrite a source bank that is also the current input.  A cache
  // imported from MLX uses A on the first call; subsequent calls alternate.
  const uint64_t source_keys = buffer_identity(raw_keys);
  const bool source_is_a = source_keys == buffer_identity(raw_keys_a_);
  const bool source_is_b = source_keys == buffer_identity(raw_keys_b_);
  const bool any_source_is_a =
      source_is_a || buffer_identity(raw_gates) == buffer_identity(raw_gates_a_) ||
      buffer_identity(raw_valid) == buffer_identity(raw_valid_a_) ||
      buffer_identity(raw_positions) == buffer_identity(raw_positions_a_);
  const bool any_source_is_b =
      source_is_b || buffer_identity(raw_gates) == buffer_identity(raw_gates_b_) ||
      buffer_identity(raw_valid) == buffer_identity(raw_valid_b_) ||
      buffer_identity(raw_positions) == buffer_identity(raw_positions_b_);
  const bool complete_source_is_a =
      source_is_a && buffer_identity(raw_gates) == buffer_identity(raw_gates_a_) &&
      buffer_identity(raw_valid) == buffer_identity(raw_valid_a_) &&
      buffer_identity(raw_positions) == buffer_identity(raw_positions_a_);
  const bool complete_source_is_b =
      source_is_b && buffer_identity(raw_gates) == buffer_identity(raw_gates_b_) &&
      buffer_identity(raw_valid) == buffer_identity(raw_valid_b_) &&
      buffer_identity(raw_positions) == buffer_identity(raw_positions_b_);
  if ((any_source_is_a || any_source_is_b) &&
      !complete_source_is_a && !complete_source_is_b) {
    throw std::invalid_argument(
        "native IndexPool raw state mixes incompatible buffer generations");
  }
  mx::array *next_keys = source_is_a ? &raw_keys_b_ : &raw_keys_a_;
  mx::array *next_gates = source_is_a ? &raw_gates_b_ : &raw_gates_a_;
  mx::array *next_valid = source_is_a ? &raw_valid_b_ : &raw_valid_a_;
  mx::array *next_positions =
      source_is_a ? &raw_positions_b_ : &raw_positions_a_;
  if (source_is_b) {
    next_keys = &raw_keys_a_;
    next_gates = &raw_gates_a_;
    next_valid = &raw_valid_a_;
    next_positions = &raw_positions_a_;
  }

  auto &device = mx::metal::device(stream_.device);
  auto *library =
      device.get_library("glm53_native_execution", current_binary_dir());
  auto *advance =
      device.get_kernel("glm53_native_advance_indexpool_raw19", library);
  auto *update_row = device.get_kernel(
      "glm53_native_update_indexpool_row_bfloat16", library);
  auto &encoder = mx::metal::get_command_encoder(stream_);

  encoder.set_compute_pipeline_state(advance);
  encoder.set_input_array(raw_keys, 0);
  encoder.set_input_array(raw_gates, 1);
  encoder.set_input_array(raw_valid, 2);
  encoder.set_input_array(raw_positions, 3);
  encoder.set_input_array(key, 4);
  encoder.set_input_array(gate, 5);
  encoder.set_input_array(current_valid, 6);
  encoder.set_output_array(*next_keys, 7);
  encoder.set_output_array(*next_gates, 8);
  encoder.set_output_array(*next_valid, 9);
  encoder.set_output_array(*next_positions, 10);
  encoder.set_bytes(previous_total_tokens, 11);
  encoder.dispatch_threads(MTL::Size(kRawWindow * kHeadDim, 1, 1),
                           MTL::Size(256, 1, 1));
  encoder.barrier();

  const int active_count = previous_total_tokens % kIndexKPool + 1;
  encoder.set_compute_pipeline_state(update_row);
  encoder.set_input_array(*next_keys, 0);
  encoder.set_input_array(*next_gates, 1);
  encoder.set_input_array(*next_valid, 2);
  encoder.set_input_array(compress_ape, 3);
  encoder.set_output_array(pool_logits_, 4);
  encoder.set_output_array(pool_probabilities_, 5);
  encoder.set_output_array(pool_keys, 6);
  encoder.set_output_array(pool_indices, 7);
  encoder.set_output_array(pool_valid, 8);
  encoder.set_bytes(pool_row, 9);
  encoder.set_bytes(active_count, 10);
  encoder.dispatch_threads(MTL::Size(kHeadDim, 1, 1),
                           MTL::Size(128, 1, 1));
  encoder.barrier();

  const int next_total = previous_total_tokens + 1;
  const int logical_pool_rows =
      (next_total + kIndexKPool - 1) / kIndexKPool;
  const int active_tail_count = next_total % kIndexKPool;
  auto selected = score_plan_.execute(
      query, mixture_weights, pool_keys, pool_indices, pool_valid,
      *next_positions, *next_valid, current_valid, logical_pool_rows,
      next_total, active_tail_count);

  current_raw_keys_ = next_keys;
  current_raw_gates_ = next_gates;
  current_raw_valid_ = next_valid;
  current_raw_positions_ = next_positions;
  execution_count_++;
  if (buffer_identities() != initial_buffer_identities_) {
    throw std::runtime_error(
        "native IndexPool update plan buffer identity changed during execute");
  }
  return {selected[0], selected[1], *next_keys, *next_gates, *next_valid,
          *next_positions};
}

uint64_t NativeIndexPoolUpdateSelectionPlan::scratch_bytes() const {
  return raw_keys_a_.nbytes() + raw_keys_b_.nbytes() + raw_gates_a_.nbytes() +
      raw_gates_b_.nbytes() + raw_valid_a_.nbytes() + raw_valid_b_.nbytes() +
      raw_positions_a_.nbytes() + raw_positions_b_.nbytes() +
      pool_logits_.nbytes() + pool_probabilities_.nbytes() +
      score_plan_.scratch_bytes();
}

std::vector<uint64_t>
NativeIndexPoolUpdateSelectionPlan::buffer_identities() const {
  auto identities = std::vector<uint64_t>{
      buffer_identity(raw_keys_a_),       buffer_identity(raw_keys_b_),
      buffer_identity(raw_gates_a_),      buffer_identity(raw_gates_b_),
      buffer_identity(raw_valid_a_),      buffer_identity(raw_valid_b_),
      buffer_identity(raw_positions_a_),  buffer_identity(raw_positions_b_),
      buffer_identity(pool_logits_),      buffer_identity(pool_probabilities_),
  };
  auto score_identities = score_plan_.buffer_identities();
  identities.insert(identities.end(), score_identities.begin(),
                    score_identities.end());
  return identities;
}

} // namespace glm53::native_execution
