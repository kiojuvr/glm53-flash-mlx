#include "native_dsa_attention_plan.h"

#include <cmath>
#include <dlfcn.h>
#include <filesystem>
#include <stdexcept>

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

int checked_kv_rows(int value) {
  if (value <= 0) {
    throw std::invalid_argument("physical KV rows must be positive");
  }
  return value;
}

} // namespace

NativeDSASparseAttentionPlan::NativeDSASparseAttentionPlan(
    int physical_pool_rows, int physical_kv_rows, float indexer_softmax_scale,
    float attention_scale)
    : physical_kv_rows_(checked_kv_rows(physical_kv_rows)),
      attention_scale_(attention_scale),
      stream_(mx::default_stream(mx::Device(mx::Device::gpu))),
      score_plan_("decode", 1, physical_pool_rows, indexer_softmax_scale),
      scaled_query_(
          owned_array({kAttentionHeads, kAttentionDim}, mx::bfloat16)),
      gathered_latent_(
          owned_array({kSelectedWidth, kAttentionDim}, mx::bfloat16)),
      attention_scores_(
          owned_array({kAttentionHeads, kSelectedWidth}, mx::bfloat16)),
      splitk_accum_(owned_array(
          {kSplitKPartitions, kAttentionHeads, kAttentionDim}, mx::float32)),
      attention_output_(
          owned_array({1, kAttentionHeads, 1, kAttentionDim}, mx::bfloat16)) {
  if (!std::isfinite(attention_scale_) || !(attention_scale_ > 0.0f)) {
    throw std::invalid_argument("attention scale must be finite and positive");
  }
  initial_buffer_identities_ = buffer_identities();
}

void NativeDSASparseAttentionPlan::validate_input(const mx::array &array,
                                                  const char *name,
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
    throw std::invalid_argument(std::string(name) +
                                " must be scheduled before native submission");
  }
}

mx::array NativeDSASparseAttentionPlan::execute(
    const mx::array &index_query, const mx::array &mixture_weights,
    const mx::array &pool_keys, const mx::array &pool_indices,
    const mx::array &pool_valid, const mx::array &raw_positions,
    const mx::array &raw_valid, const mx::array &current_valid,
    const mx::array &attention_query, const mx::array &latent,
    int logical_pool_rows, int kv_len, int active_tail_count) {
  validate_input(attention_query, "attention_query", mx::bfloat16,
                 static_cast<size_t>(kAttentionHeads) * kAttentionDim);
  validate_input(latent, "latent", mx::bfloat16,
                 static_cast<size_t>(physical_kv_rows_) * kAttentionDim);
  if (kv_len <= 0 || kv_len > physical_kv_rows_) {
    throw std::out_of_range("logical KV length is outside the native plan");
  }

  auto selected =
      score_plan_.execute(index_query, mixture_weights, pool_keys, pool_indices,
                          pool_valid, raw_positions, raw_valid, current_valid,
                          logical_pool_rows, kv_len, active_tail_count);
  const auto &selected_indices = selected[0];
  const auto &selected_valid = selected[1];

  auto &device = mx::metal::device(stream_.device);
  auto *library =
      device.get_library("glm53_native_execution", current_binary_dir());
  auto *prepare = device.get_kernel(
      "glm53_native_prepare_sparse_attention_bfloat16", library);
  auto *mask = device.get_kernel(
      "glm53_native_mask_sparse_attention_scores_bfloat16", library);

  constexpr int first_bm = 64;
  constexpr int first_bn = 32;
  constexpr int first_bk = 32;
  constexpr int first_wm = 2;
  constexpr int first_wn = 2;
  const bool has_batch = false;
  const bool use_out_source = false;
  const bool do_axpby = false;
  const bool align_m = true;
  const bool align_n = false;
  const bool align_k = true;
  mx::metal::MTLFCList first_constants = {
      {&has_batch, MTL::DataType::DataTypeBool, 10},
      {&use_out_source, MTL::DataType::DataTypeBool, 100},
      {&do_axpby, MTL::DataType::DataTypeBool, 110},
      {&align_m, MTL::DataType::DataTypeBool, 200},
      {&align_n, MTL::DataType::DataTypeBool, 201},
      {&align_k, MTL::DataType::DataTypeBool, 202},
  };
  const std::string first_name =
      "glm53_native_attention_gemm_nt_bfloat16_bfloat16_bm64_bn32_bk32_wm2_wn2";
  auto *first_gemm =
      device.get_kernel(first_name, library,
                        first_name + "_has_batch_f_use_out_source_f_do_axpby_f_"
                                     "align_M_t_align_N_f_align_K_t",
                        first_constants);
  auto *softmax =
      device.get_kernel("glm53_native_block_softmax_precise_bfloat16", library);
  auto *second_gemm =
      device.get_kernel("glm53_native_attention_gemm_splitk_nn_bfloat16_"
                        "float32_bm32_bn32_bk16_wm2_wn2_MN_taligned_K_naligned",
                        library);
  auto *second_accum = device.get_kernel(
      "glm53_native_attention_gemm_splitk_accum_bfloat16_float32", library);
  auto &encoder = mx::metal::get_command_encoder(stream_);

  encoder.set_compute_pipeline_state(prepare);
  encoder.set_input_array(selected_indices, 0);
  encoder.set_input_array(selected_valid, 1);
  encoder.set_input_array(attention_query, 2);
  encoder.set_input_array(latent, 3);
  encoder.set_output_array(scaled_query_, 4);
  encoder.set_output_array(gathered_latent_, 5);
  encoder.set_bytes(attention_scale_, 6);
  encoder.set_bytes(physical_kv_rows_, 7);
  encoder.dispatch_threads(MTL::Size(kSelectedWidth * kAttentionDim, 1, 1),
                           MTL::Size(256, 1, 1));
  encoder.barrier();

  constexpr int first_tiles_n = (kSelectedWidth + first_bn - 1) / first_bn;
  constexpr int first_tiles_m = 1;
  mlx::steel::GEMMParams first_params{kAttentionHeads,
                                      kSelectedWidth,
                                      kAttentionDim,
                                      kAttentionDim,
                                      kAttentionDim,
                                      kSelectedWidth,
                                      first_tiles_n,
                                      first_tiles_m,
                                      0,
                                      0,
                                      static_cast<int64_t>(kAttentionHeads) *
                                          kSelectedWidth,
                                      0,
                                      kAttentionDim / first_bk,
                                      1};
  encoder.set_compute_pipeline_state(first_gemm);
  encoder.set_input_array(scaled_query_, 0);
  encoder.set_input_array(gathered_latent_, 1);
  encoder.set_output_array(attention_scores_, 3);
  encoder.set_bytes(first_params, 4);
  encoder.dispatch_threadgroups(MTL::Size(first_tiles_n, first_tiles_m, 1),
                                MTL::Size(32, first_wn, first_wm));
  encoder.barrier();

  encoder.set_compute_pipeline_state(mask);
  encoder.set_output_array(attention_scores_, 0);
  encoder.set_input_array(selected_valid, 1);
  encoder.dispatch_threads(MTL::Size(kAttentionHeads * kSelectedWidth, 1, 1),
                           MTL::Size(256, 1, 1));
  encoder.barrier();

  constexpr int softmax_reads = 4;
  constexpr int softmax_simd = 32;
  constexpr int softmax_threads_needed =
      (kSelectedWidth + softmax_reads - 1) / softmax_reads;
  constexpr int softmax_groups_needed =
      (softmax_threads_needed + softmax_simd - 1) / softmax_simd;
  constexpr int softmax_group_size = softmax_simd * softmax_groups_needed;
  encoder.set_compute_pipeline_state(softmax);
  encoder.set_input_array(attention_scores_, 0);
  encoder.set_output_array(attention_scores_, 1);
  encoder.set_bytes(kSelectedWidth, 2);
  encoder.dispatch_threads(
      MTL::Size(kAttentionHeads * softmax_group_size, 1, 1),
      MTL::Size(softmax_group_size, 1, 1));
  encoder.barrier();

  constexpr int second_bm = 32;
  constexpr int second_bn = 32;
  constexpr int second_bk = 16;
  constexpr int second_wm = 2;
  constexpr int second_wn = 2;
  constexpr int second_tiles_m = (kAttentionHeads + second_bm - 1) / second_bm;
  constexpr int second_tiles_n = (kAttentionDim + second_bn - 1) / second_bn;
  constexpr int partition_stride = kAttentionHeads * kAttentionDim;
  constexpr int gemm_iterations =
      (kSelectedWidth / second_bk) / kSplitKPartitions;
  constexpr int partition_size = gemm_iterations * second_bk;
  mlx::steel::GEMMSpiltKParams second_params{
      kAttentionHeads,   kAttentionDim,    kSelectedWidth, kSelectedWidth,
      kAttentionDim,     kAttentionDim,    second_tiles_n, second_tiles_m,
      kSplitKPartitions, partition_stride, partition_size, 0,
      gemm_iterations};
  encoder.set_compute_pipeline_state(second_gemm);
  encoder.set_input_array(attention_scores_, 0);
  encoder.set_input_array(gathered_latent_, 1);
  encoder.set_output_array(splitk_accum_, 2);
  encoder.set_bytes(second_params, 3);
  encoder.dispatch_threadgroups(
      MTL::Size(second_tiles_n, second_tiles_m, kSplitKPartitions),
      MTL::Size(32, second_wn, second_wm));
  encoder.barrier();

  encoder.set_compute_pipeline_state(second_accum);
  encoder.set_input_array(splitk_accum_, 0);
  encoder.set_output_array(attention_output_, 1);
  encoder.set_bytes(kSplitKPartitions, 2);
  encoder.set_bytes(partition_stride, 3);
  encoder.set_bytes(kAttentionDim, 4);
  encoder.dispatch_threads(MTL::Size(kAttentionDim, kAttentionHeads, 1),
                           MTL::Size(32, 8, 1));

  execution_count_++;
  if (buffer_identities() != initial_buffer_identities_) {
    throw std::runtime_error(
        "native sparse-attention buffer identity changed during execute");
  }
  return attention_output_;
}

std::vector<mx::array> NativeDSASparseAttentionPlan::debug_prepare_inputs(
    const mx::array &selected_indices, const mx::array &selected_valid,
    const mx::array &attention_query, const mx::array &latent, int kv_len) {
  validate_input(selected_indices, "selected_indices", mx::int32,
                 kSelectedWidth);
  validate_input(selected_valid, "selected_valid", mx::bool_, kSelectedWidth);
  validate_input(attention_query, "attention_query", mx::bfloat16,
                 static_cast<size_t>(kAttentionHeads) * kAttentionDim);
  validate_input(latent, "latent", mx::bfloat16,
                 static_cast<size_t>(physical_kv_rows_) * kAttentionDim);
  if (kv_len <= 0 || kv_len > physical_kv_rows_) {
    throw std::out_of_range("logical KV length is outside the native plan");
  }

  auto &device = mx::metal::device(stream_.device);
  auto *library =
      device.get_library("glm53_native_execution", current_binary_dir());
  auto *prepare = device.get_kernel(
      "glm53_native_prepare_sparse_attention_bfloat16", library);
  auto &encoder = mx::metal::get_command_encoder(stream_);
  encoder.set_compute_pipeline_state(prepare);
  encoder.set_input_array(selected_indices, 0);
  encoder.set_input_array(selected_valid, 1);
  encoder.set_input_array(attention_query, 2);
  encoder.set_input_array(latent, 3);
  encoder.set_output_array(scaled_query_, 4);
  encoder.set_output_array(gathered_latent_, 5);
  encoder.set_bytes(attention_scale_, 6);
  encoder.set_bytes(physical_kv_rows_, 7);
  encoder.dispatch_threads(MTL::Size(kSelectedWidth * kAttentionDim, 1, 1),
                           MTL::Size(256, 1, 1));
  return {scaled_query_, gathered_latent_};
}

mx::array NativeDSASparseAttentionPlan::debug_attention_math(
    const mx::array &scaled_query, const mx::array &gathered_latent,
    const mx::array &selected_valid) {
  validate_input(scaled_query, "scaled_query", mx::bfloat16,
                 static_cast<size_t>(kAttentionHeads) * kAttentionDim);
  validate_input(gathered_latent, "gathered_latent", mx::bfloat16,
                 static_cast<size_t>(kSelectedWidth) * kAttentionDim);
  validate_input(selected_valid, "selected_valid", mx::bool_, kSelectedWidth);

  auto &device = mx::metal::device(stream_.device);
  auto *library =
      device.get_library("glm53_native_execution", current_binary_dir());
  auto *mask = device.get_kernel(
      "glm53_native_mask_sparse_attention_scores_bfloat16", library);

  constexpr int first_bm = 64;
  constexpr int first_bn = 32;
  constexpr int first_bk = 32;
  constexpr int first_wm = 2;
  constexpr int first_wn = 2;
  const bool has_batch = false;
  const bool use_out_source = false;
  const bool do_axpby = false;
  const bool align_m = true;
  const bool align_n = false;
  const bool align_k = true;
  mx::metal::MTLFCList first_constants = {
      {&has_batch, MTL::DataType::DataTypeBool, 10},
      {&use_out_source, MTL::DataType::DataTypeBool, 100},
      {&do_axpby, MTL::DataType::DataTypeBool, 110},
      {&align_m, MTL::DataType::DataTypeBool, 200},
      {&align_n, MTL::DataType::DataTypeBool, 201},
      {&align_k, MTL::DataType::DataTypeBool, 202},
  };
  const std::string first_name =
      "glm53_native_attention_gemm_nt_bfloat16_bfloat16_bm64_bn32_bk32_wm2_wn2";
  auto *first_gemm =
      device.get_kernel(first_name, library,
                        first_name + "_has_batch_f_use_out_source_f_do_axpby_f_"
                                     "align_M_t_align_N_f_align_K_t",
                        first_constants);
  auto *softmax =
      device.get_kernel("glm53_native_block_softmax_precise_bfloat16", library);
  auto *second_gemm =
      device.get_kernel("glm53_native_attention_gemm_splitk_nn_bfloat16_"
                        "float32_bm32_bn32_bk16_wm2_wn2_MN_taligned_K_naligned",
                        library);
  auto *second_accum = device.get_kernel(
      "glm53_native_attention_gemm_splitk_accum_bfloat16_float32", library);
  auto &encoder = mx::metal::get_command_encoder(stream_);

  constexpr int first_tiles_n = (kSelectedWidth + first_bn - 1) / first_bn;
  constexpr int first_tiles_m = 1;
  mlx::steel::GEMMParams first_params{kAttentionHeads,
                                      kSelectedWidth,
                                      kAttentionDim,
                                      kAttentionDim,
                                      kAttentionDim,
                                      kSelectedWidth,
                                      first_tiles_n,
                                      first_tiles_m,
                                      0,
                                      0,
                                      static_cast<int64_t>(kAttentionHeads) *
                                          kSelectedWidth,
                                      0,
                                      kAttentionDim / first_bk,
                                      1};
  encoder.set_compute_pipeline_state(first_gemm);
  encoder.set_input_array(scaled_query, 0);
  encoder.set_input_array(gathered_latent, 1);
  encoder.set_output_array(attention_scores_, 3);
  encoder.set_bytes(first_params, 4);
  encoder.dispatch_threadgroups(MTL::Size(first_tiles_n, first_tiles_m, 1),
                                MTL::Size(32, first_wn, first_wm));
  encoder.barrier();

  encoder.set_compute_pipeline_state(mask);
  encoder.set_output_array(attention_scores_, 0);
  encoder.set_input_array(selected_valid, 1);
  encoder.dispatch_threads(MTL::Size(kAttentionHeads * kSelectedWidth, 1, 1),
                           MTL::Size(256, 1, 1));
  encoder.barrier();

  constexpr int softmax_reads = 4;
  constexpr int softmax_simd = 32;
  constexpr int softmax_threads_needed =
      (kSelectedWidth + softmax_reads - 1) / softmax_reads;
  constexpr int softmax_groups_needed =
      (softmax_threads_needed + softmax_simd - 1) / softmax_simd;
  constexpr int softmax_group_size = softmax_simd * softmax_groups_needed;
  encoder.set_compute_pipeline_state(softmax);
  encoder.set_input_array(attention_scores_, 0);
  encoder.set_output_array(attention_scores_, 1);
  encoder.set_bytes(kSelectedWidth, 2);
  encoder.dispatch_threads(
      MTL::Size(kAttentionHeads * softmax_group_size, 1, 1),
      MTL::Size(softmax_group_size, 1, 1));
  encoder.barrier();

  constexpr int second_bm = 32;
  constexpr int second_bn = 32;
  constexpr int second_bk = 16;
  constexpr int second_wm = 2;
  constexpr int second_wn = 2;
  constexpr int second_tiles_m = (kAttentionHeads + second_bm - 1) / second_bm;
  constexpr int second_tiles_n = (kAttentionDim + second_bn - 1) / second_bn;
  constexpr int partition_stride = kAttentionHeads * kAttentionDim;
  constexpr int gemm_iterations =
      (kSelectedWidth / second_bk) / kSplitKPartitions;
  constexpr int partition_size = gemm_iterations * second_bk;
  mlx::steel::GEMMSpiltKParams second_params{
      kAttentionHeads,   kAttentionDim,    kSelectedWidth, kSelectedWidth,
      kAttentionDim,     kAttentionDim,    second_tiles_n, second_tiles_m,
      kSplitKPartitions, partition_stride, partition_size, 0,
      gemm_iterations};
  encoder.set_compute_pipeline_state(second_gemm);
  encoder.set_input_array(attention_scores_, 0);
  encoder.set_input_array(gathered_latent, 1);
  encoder.set_output_array(splitk_accum_, 2);
  encoder.set_bytes(second_params, 3);
  encoder.dispatch_threadgroups(
      MTL::Size(second_tiles_n, second_tiles_m, kSplitKPartitions),
      MTL::Size(32, second_wn, second_wm));
  encoder.barrier();

  encoder.set_compute_pipeline_state(second_accum);
  encoder.set_input_array(splitk_accum_, 0);
  encoder.set_output_array(attention_output_, 1);
  encoder.set_bytes(kSplitKPartitions, 2);
  encoder.set_bytes(partition_stride, 3);
  encoder.set_bytes(kAttentionDim, 4);
  encoder.dispatch_threads(MTL::Size(kAttentionDim, kAttentionHeads, 1),
                           MTL::Size(32, 8, 1));
  return attention_output_;
}

uint64_t NativeDSASparseAttentionPlan::scratch_bytes() const {
  return score_plan_.scratch_bytes() + scaled_query_.nbytes() +
         gathered_latent_.nbytes() + attention_scores_.nbytes() +
         splitk_accum_.nbytes() +
         score_plan_.debug_selected_indices().nbytes() +
         score_plan_.debug_selected_valid().nbytes();
}

mx::array NativeDSASparseAttentionPlan::debug_selected_indices() const {
  return score_plan_.debug_selected_indices();
}

mx::array NativeDSASparseAttentionPlan::debug_selected_valid() const {
  return score_plan_.debug_selected_valid();
}

std::vector<uint64_t> NativeDSASparseAttentionPlan::buffer_identities() const {
  auto result = score_plan_.buffer_identities();
  result.insert(result.end(), {buffer_identity(scaled_query_),
                               buffer_identity(gathered_latent_),
                               buffer_identity(attention_scores_),
                               buffer_identity(splitk_accum_),
                               buffer_identity(attention_output_)});
  return result;
}

} // namespace glm53::native_execution
