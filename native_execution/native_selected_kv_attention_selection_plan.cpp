#include "native_selected_kv_attention_selection_plan.h"

#include <cmath>
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
    if (!dladdr(reinterpret_cast<void *>(
                    &current_binary_dir), &info)) {
      throw std::runtime_error("cannot resolve native execution binary path");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return directory;
}

mx::array owned_array(const mx::Shape &shape, mx::Dtype dtype) {
  size_t elements = 1;
  for (auto extent : shape) {
    if (extent <= 0) {
      throw std::invalid_argument("native buffer extents must be positive");
    }
    elements *= static_cast<size_t>(extent);
  }
  return mx::array(
      mx::allocator::malloc(elements * mx::size_of(dtype)), shape, dtype);
}

uint64_t buffer_identity(const mx::array &array) {
  return reinterpret_cast<uint64_t>(array.buffer().ptr());
}

int checked_kv_rows(int rows) {
  if (rows <= 0 || rows > (512 << 10)) {
    throw std::invalid_argument("physical KV rows must be in [1, 512K]");
  }
  return rows;
}

} // namespace

NativeSelectedKVAttentionSelectionPlan::
    NativeSelectedKVAttentionSelectionPlan(
        int physical_pool_rows, int physical_kv_rows,
        float indexer_softmax_scale, float attention_scale)
    : physical_kv_rows_(checked_kv_rows(physical_kv_rows)),
      attention_scale_(attention_scale),
      stream_(mx::default_stream(mx::Device(mx::Device::gpu))),
      score_plan_(
          "prefill", kQueryRows, physical_pool_rows,
          indexer_softmax_scale),
      attention_plan_(physical_kv_rows_, true),
      attention_indices_(owned_array(
          {1, kQueryRows, kSelectedWidth}, mx::int32)),
      attention_valid_(owned_array(
          {1, kQueryRows, kSelectedWidth}, mx::bool_)),
      selected_latent_(owned_array(
          {kQueryRows, kSelectedWidth, kLatentDim}, mx::bfloat16)) {
  if (!std::isfinite(attention_scale_) || !(attention_scale_ > 0.0f)) {
    throw std::invalid_argument("attention scale must be finite and positive");
  }
  auto &device = mx::metal::device(stream_.device);
  auto *library = device.get_library(
      "glm53_native_execution", current_binary_dir());
  gather_pipeline_ = device.get_kernel(
      "glm53_native_gather_prefill_selected_latent_bfloat16", library);
  order_pipeline_ = device.get_kernel(
      "glm53_native_order_selected_pools_for_prefill_attention", library);
  initial_buffer_identities_ = buffer_identities();
}

void NativeSelectedKVAttentionSelectionPlan::validate_latent(
    const mx::array &latent) const {
  if (latent.dtype() != mx::bfloat16 ||
      latent.size() !=
          static_cast<size_t>(physical_kv_rows_) * kLatentDim) {
    throw std::invalid_argument("latent shape/dtype mismatch");
  }
  if (!latent.flags().row_contiguous) {
    throw std::invalid_argument("latent must be row-contiguous");
  }
  if (latent.status() == mx::array::Status::unscheduled) {
    throw std::invalid_argument("latent must be scheduled before submission");
  }
}

mx::array NativeSelectedKVAttentionSelectionPlan::execute(
    const mx::array &index_query, const mx::array &mixture_weights,
    const mx::array &pool_keys, const mx::array &pool_indices,
    const mx::array &pool_valid, const mx::array &raw_positions,
    const mx::array &raw_valid, const mx::array &current_valid,
    const mx::array &latent, const mx::array &key_weight,
    const mx::array &value_weight, const mx::array &attention_query,
    int logical_pool_rows, int kv_len, int active_tail_count) {
  validate_latent(latent);
  if (kv_len <= 0 || kv_len > physical_kv_rows_) {
    throw std::out_of_range("logical KV length is outside the native plan");
  }

  const auto selected = score_plan_.execute(
      index_query, mixture_weights, pool_keys, pool_indices, pool_valid,
      raw_positions, raw_valid, current_valid, logical_pool_rows, kv_len,
      active_tail_count);
  const auto &score_order_indices = selected[0];
  const auto &score_order_valid = selected[1];
  (void)score_order_indices;
  (void)score_order_valid;

  auto &encoder = mx::metal::get_command_encoder(stream_);
  // NativeDSAScoreSelectionPlan's terminal expansion has no downstream
  // consumer in its standalone contract.  Composition adds one, so make the
  // selected index/valid writes visible before gathering latent rows.
  encoder.barrier();
  const int raw_width = static_cast<int>(raw_positions.shape(-1));
  const int raw_rows = static_cast<int>(raw_positions.size()) / raw_width;
  encoder.set_compute_pipeline_state(order_pipeline_);
  encoder.set_input_array(score_plan_.debug_selected_pools(), 0);
  encoder.set_input_array(pool_indices, 1);
  encoder.set_input_array(pool_valid, 2);
  encoder.set_input_array(raw_positions, 3);
  encoder.set_input_array(raw_valid, 4);
  encoder.set_input_array(current_valid, 5);
  encoder.set_output_array(attention_indices_, 6);
  encoder.set_output_array(attention_valid_, 7);
  encoder.set_bytes(logical_pool_rows, 8);
  encoder.set_bytes(kv_len, 9);
  encoder.set_bytes(active_tail_count, 10);
  encoder.set_bytes(raw_width, 11);
  encoder.set_bytes(raw_rows, 12);
  encoder.dispatch_threadgroups(
      MTL::Size(1, kQueryRows, 1), MTL::Size(512, 1, 1));
  encoder.barrier();

  constexpr uint32_t elements =
      kQueryRows * kSelectedWidth * kLatentDim;
  encoder.set_compute_pipeline_state(gather_pipeline_);
  encoder.set_input_array(latent, 0);
  encoder.set_input_array(attention_indices_, 1);
  encoder.set_input_array(attention_valid_, 2);
  encoder.set_output_array(selected_latent_, 3);
  encoder.set_bytes(physical_kv_rows_, 4);
  encoder.dispatch_threads(
      MTL::Size(elements, 1, 1), MTL::Size(256, 1, 1));
  encoder.barrier();

  auto output = attention_plan_.execute_attention(
      selected_latent_, key_weight, value_weight, attention_query,
      attention_indices_, attention_valid_, attention_scale_);

  ++execution_count_;
  if (buffer_identities() != initial_buffer_identities_) {
    throw std::runtime_error(
        "composed selected-K/V attention buffer identity changed");
  }
  return output;
}

uint64_t NativeSelectedKVAttentionSelectionPlan::scratch_bytes() const {
  return score_plan_.scratch_bytes() + selected_latent_.nbytes() +
      attention_indices_.nbytes() + attention_valid_.nbytes() +
      attention_plan_.scratch_bytes();
}

std::vector<uint64_t>
NativeSelectedKVAttentionSelectionPlan::buffer_identities() const {
  auto result = score_plan_.buffer_identities();
  result.push_back(buffer_identity(attention_indices_));
  result.push_back(buffer_identity(attention_valid_));
  result.push_back(buffer_identity(selected_latent_));
  const auto attention = attention_plan_.buffer_identities();
  result.insert(result.end(), attention.begin(), attention.end());
  return result;
}

} // namespace glm53::native_execution
