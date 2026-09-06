#include "native_indexer_plan.h"

#include <dlfcn.h>
#include <filesystem>
#include <stdexcept>
#include <utility>

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"

namespace glm53::native_execution {

namespace {

std::string current_binary_dir() {
  static const std::string directory = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir), &info)) {
      throw std::runtime_error("cannot resolve native execution binary path");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return directory;
}

size_t checked_elements(const mx::Shape& shape) {
  size_t result = 1;
  for (auto extent : shape) {
    if (extent <= 0) {
      throw std::invalid_argument("native buffer extents must be positive");
    }
    result *= static_cast<size_t>(extent);
  }
  return result;
}

mx::array owned_array(const mx::Shape& shape, mx::Dtype dtype) {
  auto elements = checked_elements(shape);
  return mx::array(
      mx::allocator::malloc(elements * mx::size_of(dtype)), shape, dtype);
}

uint64_t buffer_identity(const mx::array& array) {
  return reinterpret_cast<uint64_t>(array.buffer().ptr());
}

int checked_query_rows(const std::string& mode, int query_rows) {
  if (mode != "prefill" && mode != "decode") {
    throw std::invalid_argument("native mode must be prefill or decode");
  }
  if (query_rows <= 0 || (mode == "decode" && query_rows != 1)) {
    throw std::invalid_argument("native query-row geometry is invalid");
  }
  return query_rows;
}

int checked_pool_rows(int physical_pool_rows) {
  if (physical_pool_rows < 512 || physical_pool_rows > 65600 ||
      physical_pool_rows % 64 != 0) {
    throw std::invalid_argument(
        "physical pool rows must be 64-aligned in [512, 65600]");
  }
  return physical_pool_rows;
}

mx::Dtype checked_score_dtype(const std::string& score_dtype) {
  if (score_dtype == "bfloat16") return mx::bfloat16;
  if (score_dtype == "float32") return mx::float32;
  throw std::invalid_argument("score dtype must be bfloat16 or float32");
}

} // namespace

NativeIndexSelectionPlan::NativeIndexSelectionPlan(
    std::string mode,
    int query_rows,
    int physical_pool_rows,
    std::string score_dtype)
    : mode_(std::move(mode)),
      query_rows_(checked_query_rows(mode_, query_rows)),
      physical_pool_rows_(checked_pool_rows(physical_pool_rows)),
      score_dtype_(checked_score_dtype(score_dtype)),
      stream_(mx::default_stream(mx::Device(mx::Device::gpu))),
      selected_pool_scratch_(owned_array(
          {1, query_rows, kSelectedPools}, mx::uint32)),
      selected_token_indices_(owned_array(
          {1, query_rows, kSelectedWidth}, mx::int32)),
      selected_token_valid_(owned_array(
          {1, query_rows, kSelectedWidth}, mx::bool_)) {
  initial_buffer_identities_ = buffer_identities();
}

void NativeIndexSelectionPlan::validate_input(
    const mx::array& array,
    const char* name,
    mx::Dtype dtype,
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

std::vector<mx::array> NativeIndexSelectionPlan::execute(
    const mx::array& scores,
    const mx::array& pool_indices,
    const mx::array& pool_valid,
    const mx::array& raw_positions,
    const mx::array& raw_valid,
    const mx::array& current_valid,
    int logical_pool_rows,
    int kv_len,
    int active_tail_count) {
  validate_input(
      scores,
      "scores",
      score_dtype_,
      static_cast<size_t>(query_rows_) * physical_pool_rows_);
  validate_input(
      pool_indices,
      "pool_indices",
      mx::int64,
      static_cast<size_t>(physical_pool_rows_) * kIndexKPool);
  validate_input(
      pool_valid,
      "pool_valid",
      mx::bool_,
      static_cast<size_t>(physical_pool_rows_));
  if (raw_positions.dtype() != mx::int64 ||
      raw_positions.shape(-1) < kIndexKPool) {
    throw std::invalid_argument("raw_positions must contain the active pool");
  }
  if (raw_valid.dtype() != mx::bool_ ||
      raw_valid.size() != raw_positions.size()) {
    throw std::invalid_argument("raw_valid must match raw_positions");
  }
  validate_input(
      current_valid,
      "current_valid",
      mx::bool_,
      static_cast<size_t>(query_rows_));
  if (!raw_positions.flags().row_contiguous ||
      !raw_valid.flags().row_contiguous ||
      raw_positions.status() == mx::array::Status::unscheduled ||
      raw_valid.status() == mx::array::Status::unscheduled) {
    throw std::invalid_argument("raw tail inputs must be scheduled row-major arrays");
  }
  if (logical_pool_rows < kSelectedPools ||
      logical_pool_rows > physical_pool_rows_) {
    throw std::out_of_range("logical pool rows are outside the native plan");
  }
  if (kv_len <= 0 || active_tail_count < 0 ||
      active_tail_count > kTailWidth) {
    throw std::out_of_range("KV length or active tail count is invalid");
  }

  auto& device = mx::metal::device(stream_.device);
  auto* library = device.get_library(
      "glm53_native_execution", current_binary_dir());
  const std::string suffix =
      score_dtype_ == mx::bfloat16 ? "bfloat16" : "float32";
  auto* topk = device.get_kernel(
      "glm53_native_exact_partial_topk_512_" + suffix, library);
  auto* expand = device.get_kernel(
      "glm53_native_expand_selected_pools", library);
  auto& encoder = mx::metal::get_command_encoder(stream_);

  encoder.set_compute_pipeline_state(topk);
  encoder.set_input_array(scores, 0);
  encoder.set_output_array(selected_pool_scratch_, 1);
  encoder.set_bytes(logical_pool_rows, 2);
  encoder.set_bytes(physical_pool_rows_, 3);
  encoder.dispatch_threads(
      MTL::Size(kSelectedPools, query_rows_, 1),
      MTL::Size(kSelectedPools, 1, 1));
  encoder.barrier();

  const int raw_width = static_cast<int>(raw_positions.shape(-1));
  const int raw_rows = static_cast<int>(raw_positions.size()) / raw_width;
  if (raw_rows != 1 && raw_rows != query_rows_) {
    throw std::invalid_argument(
        "raw tail batch must be shared or match native query rows");
  }
  encoder.set_compute_pipeline_state(expand);
  encoder.set_input_array(selected_pool_scratch_, 0);
  encoder.set_input_array(pool_indices, 1);
  encoder.set_input_array(pool_valid, 2);
  encoder.set_input_array(raw_positions, 3);
  encoder.set_input_array(raw_valid, 4);
  encoder.set_input_array(current_valid, 5);
  encoder.set_output_array(selected_token_indices_, 6);
  encoder.set_output_array(selected_token_valid_, 7);
  encoder.set_bytes(logical_pool_rows, 8);
  encoder.set_bytes(kv_len, 9);
  encoder.set_bytes(active_tail_count, 10);
  encoder.set_bytes(raw_width, 11);
  encoder.set_bytes(raw_rows, 12);
  encoder.dispatch_threads(
      MTL::Size(kSelectedWidth, query_rows_, 1), MTL::Size(256, 1, 1));

  execution_count_++;
  if (buffer_identities() != initial_buffer_identities_) {
    throw std::runtime_error("native plan buffer identity changed during execute");
  }
  return {selected_token_indices_, selected_token_valid_};
}

std::vector<uint64_t> NativeIndexSelectionPlan::buffer_identities() const {
  return {
      buffer_identity(selected_pool_scratch_),
      buffer_identity(selected_token_indices_),
      buffer_identity(selected_token_valid_),
  };
}

} // namespace glm53::native_execution
