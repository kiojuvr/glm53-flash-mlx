#include "native_selected_v_av_plan.h"

#include <algorithm>
#include <cmath>
#include <dlfcn.h>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/steel/gemm/params.h"

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

int checked_physical_k(int physical_k) {
  if (physical_k <= 0 || physical_k > (512 << 10)) {
    throw std::invalid_argument("physical K must be in [1, 512K]");
  }
  return physical_k;
}

} // namespace

NativeSelectedVProjectionAVPlan::NativeSelectedVProjectionAVPlan(
    int physical_k, bool attention_enabled)
    : physical_k_(checked_physical_k(physical_k)),
      packed_k_(std::min(
          ((physical_k_ + kBK - 1) / kBK) * kBK, kMaximumPackedK)),
      attention_enabled_(attention_enabled),
      stream_(mx::default_stream(mx::Device(mx::Device::gpu))),
      projected_key_(owned_array(
          attention_enabled_
              ? mx::Shape{kHeads, kSelectedWidth, kLatentDim}
              : mx::Shape{1},
          mx::bfloat16)),
      projected_value_(owned_array(
          {kHeads, kSelectedWidth, kValueDim}, mx::bfloat16)),
      scaled_query_(owned_array(
          attention_enabled_ ? mx::Shape{kHeads, kLatentDim} : mx::Shape{1},
          mx::bfloat16)),
      attention_scores_(owned_array(
          attention_enabled_ ? mx::Shape{kHeads, kSelectedWidth} : mx::Shape{1},
          mx::bfloat16)),
      attention_probabilities_(owned_array(
          attention_enabled_ ? mx::Shape{kHeads, kSelectedWidth} : mx::Shape{1},
          mx::bfloat16)),
      lane_to_selected_(owned_array({packed_k_}, mx::int32)),
      output_(owned_array(
          {kQueryRows, kHeads, 1, kValueDim}, mx::bfloat16)) {
  auto &device = mx::metal::device(stream_.device);
  if (device.get_architecture().back() != 'd' ||
      device.get_architecture_gen() >= 17) {
    throw std::runtime_error(
        "selected V/AV Steel geometry is qualified only for pre-NAX apple-gpu-d");
  }
  const bool has_batch = false;
  const bool use_out_source = false;
  const bool do_axpby = false;
  const bool align_m = false;
  const bool align_n = true;
  const bool align_k = true;
  mx::metal::MTLFCList constants = {
      {&has_batch, MTL::DataType::DataTypeBool, 10},
      {&use_out_source, MTL::DataType::DataTypeBool, 100},
      {&do_axpby, MTL::DataType::DataTypeBool, 110},
      {&align_m, MTL::DataType::DataTypeBool, 200},
      {&align_n, MTL::DataType::DataTypeBool, 201},
      {&align_k, MTL::DataType::DataTypeBool, 202},
  };
  const std::string projection_name =
      "glm53_native_steel_gemm_nt_bfloat16_bfloat16_bm64_bn64_bk16_wm1_wn2";
  const std::string projection_hash = projection_name +
      "_has_batch_f_use_out_source_f_do_axpby_f_align_M_f_align_N_t_align_K_t";
  auto *library =
      device.get_library("glm53_native_execution", current_binary_dir());
  const std::string key_projection_name =
      "glm53_native_steel_gemm_nn_bfloat16_bfloat16_bm64_bn64_bk16_wm1_wn2";
  const std::string key_projection_hash = key_projection_name +
      "_has_batch_f_use_out_source_f_do_axpby_f_align_M_f_align_N_t_align_K_t";
  key_projection_pipeline_ = device.get_kernel(
      key_projection_name, library, key_projection_hash, constants);
  projection_pipeline_ = device.get_kernel(
      projection_name, library, projection_hash, constants);
  query_scale_pipeline_ = device.get_kernel(
      "glm53_native_prepare_prefill_attention_query_bfloat16", library);
  const std::string qk_name =
      "glm53_native_gemv_bfloat16_bm4_bn1_sm1_sn32_tm4_tn4_nc0_axpby0";
  qk_pipeline_ = device.get_kernel(qk_name, library);
  mask_pipeline_ = device.get_kernel(
      "glm53_native_mask_sparse_attention_scores_bfloat16", library);
  softmax_pipeline_ = device.get_kernel(
      "glm53_native_block_softmax_precise_bfloat16", library);
  map_pipeline_ = device.get_kernel(
      "glm53_native_build_virtual_bk16_map", library);
  av_pipeline_ = device.get_kernel(
      "glm53_native_virtual_bk16_av_bfloat16", library);
  initial_buffer_identities_ = buffer_identities();
}

void NativeSelectedVProjectionAVPlan::validate_input(
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
    throw std::invalid_argument(
        std::string(name) + " must be scheduled before native submission");
  }
}

mx::array NativeSelectedVProjectionAVPlan::execute(
    const mx::array &selected_probabilities,
    const mx::array &selected_latent,
    const mx::array &value_weight,
    const mx::array &selected_indices,
    const mx::array &selected_valid) {
  validate_input(
      selected_probabilities, "selected_probabilities", mx::bfloat16,
      static_cast<size_t>(kQueryRows) * kHeads * kSelectedWidth);
  validate_input(
      selected_latent, "selected_latent", mx::bfloat16,
      static_cast<size_t>(kQueryRows) * kSelectedWidth * kLatentDim);
  validate_input(
      value_weight, "value_weight", mx::bfloat16,
      static_cast<size_t>(kHeads) * kValueDim * kLatentDim);
  validate_input(
      selected_indices, "selected_indices", mx::int32,
      static_cast<size_t>(kQueryRows) * kSelectedWidth);
  validate_input(
      selected_valid, "selected_valid", mx::bool_,
      static_cast<size_t>(kQueryRows) * kSelectedWidth);

  constexpr int bm = 64;
  constexpr int bn = 64;
  constexpr int bk = 16;
  constexpr int wn = 2;
  constexpr int tiles_m = (kSelectedWidth + bm - 1) / bm;
  constexpr int tiles_n = kValueDim / bn;
  constexpr int64_t latent_row_bytes =
      static_cast<int64_t>(kSelectedWidth) * kLatentDim * sizeof(uint16_t);
  constexpr int64_t probability_row_bytes =
      static_cast<int64_t>(kHeads) * kSelectedWidth * sizeof(uint16_t);
  constexpr int64_t index_row_bytes =
      static_cast<int64_t>(kSelectedWidth) * sizeof(int32_t);
  constexpr int64_t valid_row_bytes =
      static_cast<int64_t>(kSelectedWidth) * sizeof(bool);
  constexpr int64_t output_row_bytes =
      static_cast<int64_t>(kHeads) * kValueDim * sizeof(uint16_t);
  mlx::steel::GEMMParams projection_params{
      kSelectedWidth,
      kValueDim,
      kLatentDim,
      kLatentDim,
      kLatentDim,
      kValueDim,
      tiles_n,
      tiles_m,
      0,
      static_cast<int64_t>(kValueDim) * kLatentDim,
      static_cast<int64_t>(kSelectedWidth) * kValueDim,
      0,
      kLatentDim / bk,
      1};

  auto &encoder = mx::metal::get_command_encoder(stream_);
  for (int row = 0; row < kQueryRows; ++row) {
    encoder.set_compute_pipeline_state(projection_pipeline_);
    encoder.set_input_array(selected_latent, 0, row * latent_row_bytes);
    encoder.set_input_array(value_weight, 1);
    encoder.set_output_array(projected_value_, 3);
    encoder.set_bytes(projection_params, 4);
    encoder.dispatch_threadgroups(
        MTL::Size(tiles_n, tiles_m, kHeads), MTL::Size(32, wn, 1));
    encoder.barrier();

    encoder.set_compute_pipeline_state(map_pipeline_);
    encoder.set_input_array(selected_indices, 0, row * index_row_bytes);
    encoder.set_input_array(selected_valid, 1, row * valid_row_bytes);
    encoder.set_output_array(lane_to_selected_, 2);
    encoder.set_bytes(physical_k_, 3);
    encoder.set_bytes(packed_k_, 4);
    encoder.dispatch_threadgroups(
        MTL::Size(1, 1, 1), MTL::Size(256, 1, 1));
    encoder.barrier();

    encoder.set_compute_pipeline_state(av_pipeline_);
    encoder.set_input_array(
        selected_probabilities, 0, row * probability_row_bytes);
    encoder.set_input_array(projected_value_, 1);
    encoder.set_input_array(lane_to_selected_, 2);
    encoder.set_output_array(output_, 3, row * output_row_bytes);
    encoder.set_bytes(packed_k_, 4);
    encoder.dispatch_threadgroups(
        MTL::Size(1, 1, kHeads), MTL::Size(32, 4, 1));
    encoder.barrier();
  }

  ++execution_count_;
  if (buffer_identities() != initial_buffer_identities_) {
    throw std::runtime_error("selected V/AV native buffer identity changed");
  }
  return output_;
}

mx::array NativeSelectedVProjectionAVPlan::execute_attention(
    const mx::array &selected_latent,
    const mx::array &key_weight,
    const mx::array &value_weight,
    const mx::array &attention_query,
    const mx::array &selected_indices,
    const mx::array &selected_valid,
    float attention_scale) {
  if (!attention_enabled_) {
    throw std::runtime_error(
        "selected K/V attention region was not enabled at plan construction");
  }
  validate_input(
      selected_latent, "selected_latent", mx::bfloat16,
      static_cast<size_t>(kQueryRows) * kSelectedWidth * kLatentDim);
  validate_input(
      key_weight, "key_weight", mx::bfloat16,
      static_cast<size_t>(kHeads) * kLatentDim * kLatentDim);
  validate_input(
      value_weight, "value_weight", mx::bfloat16,
      static_cast<size_t>(kHeads) * kValueDim * kLatentDim);
  validate_input(
      attention_query, "attention_query", mx::bfloat16,
      static_cast<size_t>(kHeads) * kQueryRows * kLatentDim);
  validate_input(
      selected_indices, "selected_indices", mx::int32,
      static_cast<size_t>(kQueryRows) * kSelectedWidth);
  validate_input(
      selected_valid, "selected_valid", mx::bool_,
      static_cast<size_t>(kQueryRows) * kSelectedWidth);
  if (!std::isfinite(attention_scale) || !(attention_scale > 0.0f)) {
    throw std::invalid_argument("attention scale must be finite and positive");
  }

  constexpr int projection_bm = 64;
  constexpr int projection_bn = 64;
  constexpr int projection_bk = 16;
  constexpr int projection_wn = 2;
  constexpr int projection_tiles_m =
      (kSelectedWidth + projection_bm - 1) / projection_bm;
  constexpr int key_tiles_n = kLatentDim / projection_bn;
  constexpr int value_tiles_n = kValueDim / projection_bn;
  constexpr int qk_output_rows_per_group = 16;
  constexpr int qk_groups =
      (kSelectedWidth + qk_output_rows_per_group - 1) /
      qk_output_rows_per_group;
  constexpr int64_t latent_row_bytes =
      static_cast<int64_t>(kSelectedWidth) * kLatentDim * sizeof(uint16_t);
  constexpr int64_t index_row_bytes =
      static_cast<int64_t>(kSelectedWidth) * sizeof(int32_t);
  constexpr int64_t valid_row_bytes =
      static_cast<int64_t>(kSelectedWidth) * sizeof(bool);
  constexpr int64_t output_row_bytes =
      static_cast<int64_t>(kHeads) * kValueDim * sizeof(uint16_t);
  mlx::steel::GEMMParams key_projection_params{
      kSelectedWidth,
      kLatentDim,
      kLatentDim,
      kLatentDim,
      kLatentDim,
      kLatentDim,
      key_tiles_n,
      projection_tiles_m,
      0,
      static_cast<int64_t>(kLatentDim) * kLatentDim,
      static_cast<int64_t>(kSelectedWidth) * kLatentDim,
      0,
      kLatentDim / projection_bk,
      1};
  mlx::steel::GEMMParams value_projection_params{
      kSelectedWidth,
      kValueDim,
      kLatentDim,
      kLatentDim,
      kLatentDim,
      kValueDim,
      value_tiles_n,
      projection_tiles_m,
      0,
      static_cast<int64_t>(kValueDim) * kLatentDim,
      static_cast<int64_t>(kSelectedWidth) * kValueDim,
      0,
      kLatentDim / projection_bk,
      1};
  constexpr int qk_input_size = kLatentDim;
  constexpr int qk_output_size = kSelectedWidth;
  constexpr int qk_matrix_ld = kLatentDim;
  constexpr float qk_alpha = 1.0f;
  constexpr float qk_beta = 0.0f;
  constexpr int qk_batch_ndim = 1;
  constexpr int qk_batch_shape = kHeads;
  constexpr int64_t qk_vector_batch_stride = kLatentDim;
  constexpr int64_t qk_matrix_batch_stride =
      static_cast<int64_t>(kSelectedWidth) * kLatentDim;
  constexpr int64_t qk_bias_batch_stride = 0;
  constexpr int qk_bias_stride = 0;
  constexpr int softmax_reads = 4;
  constexpr int softmax_simd = 32;
  constexpr int softmax_threads_needed =
      (kSelectedWidth + softmax_reads - 1) / softmax_reads;
  constexpr int softmax_groups_needed =
      (softmax_threads_needed + softmax_simd - 1) / softmax_simd;
  constexpr int softmax_group_size = softmax_simd * softmax_groups_needed;

  auto &encoder = mx::metal::get_command_encoder(stream_);
  for (int row = 0; row < kQueryRows; ++row) {
    const int64_t latent_offset = row * latent_row_bytes;
    const int64_t valid_offset = row * valid_row_bytes;

    encoder.set_compute_pipeline_state(key_projection_pipeline_);
    encoder.set_input_array(selected_latent, 0, latent_offset);
    encoder.set_input_array(key_weight, 1);
    encoder.set_output_array(projected_key_, 3);
    encoder.set_bytes(key_projection_params, 4);
    encoder.dispatch_threadgroups(
        MTL::Size(key_tiles_n, projection_tiles_m, kHeads),
        MTL::Size(32, projection_wn, 1));
    encoder.barrier();

    encoder.set_compute_pipeline_state(query_scale_pipeline_);
    encoder.set_input_array(attention_query, 0);
    encoder.set_output_array(scaled_query_, 1);
    encoder.set_bytes(row, 2);
    encoder.set_bytes(attention_scale, 3);
    encoder.dispatch_threads(
        MTL::Size(kHeads * kLatentDim, 1, 1), MTL::Size(256, 1, 1));
    encoder.barrier();

    encoder.set_compute_pipeline_state(qk_pipeline_);
    encoder.set_input_array(projected_key_, 0);
    encoder.set_input_array(scaled_query_, 1);
    // Bias is disabled by the pipeline specialization but Metal still
    // requires a bound resource for the declared argument.
    encoder.set_input_array(scaled_query_, 2);
    encoder.set_output_array(attention_scores_, 3);
    encoder.set_bytes(qk_input_size, 4);
    encoder.set_bytes(qk_output_size, 5);
    encoder.set_bytes(qk_matrix_ld, 6);
    encoder.set_bytes(qk_alpha, 7);
    encoder.set_bytes(qk_beta, 8);
    encoder.set_bytes(qk_batch_ndim, 9);
    encoder.set_bytes(qk_batch_shape, 10);
    encoder.set_bytes(qk_vector_batch_stride, 11);
    encoder.set_bytes(qk_matrix_batch_stride, 12);
    encoder.set_bytes(qk_bias_batch_stride, 13);
    encoder.set_bytes(qk_bias_stride, 14);
    encoder.dispatch_threadgroups(
        MTL::Size(qk_groups, 1, kHeads), MTL::Size(32, 4, 1));
    encoder.barrier();

    encoder.set_compute_pipeline_state(mask_pipeline_);
    encoder.set_output_array(attention_scores_, 0);
    encoder.set_input_array(selected_valid, 1, valid_offset);
    encoder.dispatch_threads(
        MTL::Size(kHeads * kSelectedWidth, 1, 1), MTL::Size(256, 1, 1));
    encoder.barrier();

    encoder.set_compute_pipeline_state(softmax_pipeline_);
    encoder.set_input_array(attention_scores_, 0);
    encoder.set_output_array(attention_probabilities_, 1);
    encoder.set_bytes(kSelectedWidth, 2);
    encoder.dispatch_threads(
        MTL::Size(kHeads * softmax_group_size, 1, 1),
        MTL::Size(softmax_group_size, 1, 1));
    encoder.barrier();

    encoder.set_compute_pipeline_state(projection_pipeline_);
    encoder.set_input_array(selected_latent, 0, latent_offset);
    encoder.set_input_array(value_weight, 1);
    encoder.set_output_array(projected_value_, 3);
    encoder.set_bytes(value_projection_params, 4);
    encoder.dispatch_threadgroups(
        MTL::Size(value_tiles_n, projection_tiles_m, kHeads),
        MTL::Size(32, projection_wn, 1));
    encoder.barrier();

    encoder.set_compute_pipeline_state(map_pipeline_);
    encoder.set_input_array(selected_indices, 0, row * index_row_bytes);
    encoder.set_input_array(selected_valid, 1, valid_offset);
    encoder.set_output_array(lane_to_selected_, 2);
    encoder.set_bytes(physical_k_, 3);
    encoder.set_bytes(packed_k_, 4);
    encoder.dispatch_threadgroups(
        MTL::Size(1, 1, 1), MTL::Size(256, 1, 1));
    encoder.barrier();

    encoder.set_compute_pipeline_state(av_pipeline_);
    encoder.set_input_array(attention_probabilities_, 0);
    encoder.set_input_array(projected_value_, 1);
    encoder.set_input_array(lane_to_selected_, 2);
    encoder.set_output_array(output_, 3, row * output_row_bytes);
    encoder.set_bytes(packed_k_, 4);
    encoder.dispatch_threadgroups(
        MTL::Size(1, 1, kHeads), MTL::Size(32, 4, 1));
    encoder.barrier();
  }

  ++execution_count_;
  if (buffer_identities() != initial_buffer_identities_) {
    throw std::runtime_error("selected K/V attention buffer identity changed");
  }
  return output_;
}

uint64_t NativeSelectedVProjectionAVPlan::scratch_bytes() const {
  return projected_key_.nbytes() + projected_value_.nbytes() +
      scaled_query_.nbytes() + attention_scores_.nbytes() +
      attention_probabilities_.nbytes() + lane_to_selected_.nbytes();
}

std::vector<uint64_t> NativeSelectedVProjectionAVPlan::buffer_identities() const {
  return {
      buffer_identity(projected_key_),
      buffer_identity(projected_value_),
      buffer_identity(scaled_query_),
      buffer_identity(attention_scores_),
      buffer_identity(attention_probabilities_),
      buffer_identity(lane_to_selected_),
      buffer_identity(output_),
  };
}

} // namespace glm53::native_execution
