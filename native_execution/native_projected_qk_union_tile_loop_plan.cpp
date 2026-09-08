#include "native_projected_qk_union_tile_loop_plan.h"

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

int checked_physical_k(int value) {
  if (value <= 4096 || value > (512 << 10)) {
    throw std::invalid_argument(
        "multi-tile projected-QK physical K must be in (4096, 512K]");
  }
  return value;
}

int checked_attention_query_rows(int value) {
  if (value != 4 && value != 256) {
    throw std::invalid_argument(
        "projected-QK attention query rows must be 4 or 256");
  }
  return value;
}

int checked_tile_rows(int value) {
  if (value < 4096 || value > 65536 || (value & (value - 1)) != 0) {
    throw std::invalid_argument(
        "projected-QK tile rows must be a power of two in [4096, 65536]");
  }
  return value;
}

} // namespace

NativeProjectedQKUnionTileLoopPlan::NativeProjectedQKUnionTileLoopPlan(
    int physical_k, int attention_query_rows, int tile_rows,
    bool softmax_enabled, bool value_enabled)
    : physical_k_(checked_physical_k(physical_k)),
      attention_query_rows_(
          checked_attention_query_rows(attention_query_rows)),
      tile_rows_(checked_tile_rows(tile_rows)),
      tile_count_((physical_k_ + tile_rows_ - 1) / tile_rows_),
      packed_k_(std::min(
          ((physical_k_ + 15) / 16) * 16, kSelectedWidth * 16)),
      softmax_enabled_(softmax_enabled || value_enabled),
      value_enabled_(value_enabled),
      stream_(mx::default_stream(mx::Device(mx::Device::gpu))),
      union_plan_(physical_k_),
      union_latent_tile_(
          owned_array({tile_rows_, kLatentDim}, mx::bfloat16)),
      projected_union_key_tile_(owned_array(
          {kHeads, tile_rows_, kLatentDim}, mx::bfloat16)),
      scaled_queries_(owned_array(
          {attention_query_rows_, kHeads, kLatentDim}, mx::bfloat16)),
      attention_scores_(owned_array(
          {attention_query_rows_, kHeads, kSelectedWidth}, mx::bfloat16)),
      attention_probabilities_(owned_array(
          softmax_enabled_
              ? mx::Shape{attention_query_rows_, kHeads, kSelectedWidth}
              : mx::Shape{1},
          mx::bfloat16)),
      projected_union_value_tile_(owned_array(
          value_enabled_
              ? mx::Shape{kHeads, tile_rows_, kValueDim}
              : mx::Shape{1},
          mx::bfloat16)),
      selected_values_(owned_array(
          value_enabled_
              ? mx::Shape{
                    attention_query_rows_, kHeads, kSelectedWidth, kValueDim}
              : mx::Shape{1},
          mx::bfloat16)),
      lane_to_selected_(owned_array(
          value_enabled_ ? mx::Shape{packed_k_} : mx::Shape{1}, mx::int32)),
      attention_output_(owned_array(
          value_enabled_
              ? mx::Shape{attention_query_rows_, kHeads, 1, kValueDim}
              : mx::Shape{1},
          mx::bfloat16)) {
  auto &device = mx::metal::device(stream_.device);
  if (device.get_architecture().back() != 'd' ||
      device.get_architecture_gen() >= 17) {
    throw std::runtime_error(
        "projected-QK tile loop is qualified only for pre-NAX apple-gpu-d");
  }
  const bool has_batch = false;
  const bool use_out_source = false;
  const bool do_axpby = false;
  const bool align_m = true;
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
  auto *library = device.get_library(
      "glm53_native_execution", current_binary_dir());
  const std::string projection_name =
      "glm53_native_steel_gemm_nn_bfloat16_bfloat16_bm64_bn64_bk16_wm1_wn2";
  const std::string projection_hash = projection_name +
      "_has_batch_f_use_out_source_f_do_axpby_f_align_M_t_align_N_t_align_K_t";
  clear_scores_pipeline_ = device.get_kernel(
      "glm53_native_clear_q4_union_scores_bfloat16", library);
  gather_tile_pipeline_ = device.get_kernel(
      "glm53_native_gather_selected_union_latent_tile_bfloat16", library);
  projection_pipeline_ = device.get_kernel(
      projection_name, library, projection_hash, constants);
  query_scale_pipeline_ = device.get_kernel(
      "glm53_native_prepare_union_prefill_attention_query_bfloat16", library);
  projected_qk_pipeline_ = device.get_kernel(
      "glm53_native_projected_union_qk_tile_bfloat16", library);
  softmax_pipeline_ = device.get_kernel(
      "glm53_native_block_softmax_precise_bfloat16", library);
  const std::string value_projection_name =
      "glm53_native_steel_gemm_nt_bfloat16_bfloat16_bm64_bn64_bk16_wm1_wn2";
  const std::string value_projection_hash = value_projection_name +
      "_has_batch_f_use_out_source_f_do_axpby_f_align_M_t_align_N_t_align_K_t";
  value_projection_pipeline_ = device.get_kernel(
      value_projection_name, library, value_projection_hash, constants);
  scatter_value_pipeline_ = device.get_kernel(
      "glm53_native_scatter_projected_union_value_tile_bfloat16", library);
  map_pipeline_ = device.get_kernel(
      "glm53_native_build_virtual_bk16_map", library);
  av_pipeline_ = device.get_kernel(
      "glm53_native_virtual_bk16_av_bfloat16", library);
  initial_buffer_identities_ = buffer_identities();
}

void NativeProjectedQKUnionTileLoopPlan::validate_input(
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

mx::array NativeProjectedQKUnionTileLoopPlan::execute(
    const mx::array &selected_indices, const mx::array &selected_valid,
    const mx::array &latent, const mx::array &key_weight,
    const mx::array &attention_query, float attention_scale) {
  validate_input(
      latent, "latent", mx::bfloat16,
      static_cast<size_t>(physical_k_) * kLatentDim);
  validate_input(
      key_weight, "key_weight", mx::bfloat16,
      static_cast<size_t>(kHeads) * kLatentDim * kLatentDim);
  validate_input(
      attention_query, "attention_query", mx::bfloat16,
      static_cast<size_t>(kHeads) * attention_query_rows_ * kLatentDim);
  if (!std::isfinite(attention_scale) || !(attention_scale > 0.0f)) {
    throw std::invalid_argument("attention scale must be finite and positive");
  }

  const auto union_outputs = union_plan_.execute(
      selected_indices, selected_valid);
  const auto &union_indices = union_outputs[0];
  const auto &union_count = union_outputs[1];
  const auto &query_union_slots = union_outputs[2];

  constexpr int bm = 64;
  constexpr int bn = 64;
  constexpr int bk = 16;
  constexpr int wn = 2;
  const int tiles_m = tile_rows_ / bm;
  constexpr int tiles_n = kLatentDim / bn;
  mlx::steel::GEMMParams projection_params{
      tile_rows_, kLatentDim, kLatentDim,
      kLatentDim, kLatentDim, kLatentDim,
      tiles_n, tiles_m, 0,
      static_cast<int64_t>(kLatentDim) * kLatentDim,
      static_cast<int64_t>(tile_rows_) * kLatentDim,
      0, kLatentDim / bk, 1};
  const uint32_t score_elements = static_cast<uint32_t>(
      attention_query_rows_ * kHeads * kSelectedWidth);
  constexpr int qk_output_rows_per_group = 16;
  constexpr int qk_groups =
      (kSelectedWidth + qk_output_rows_per_group - 1) /
      qk_output_rows_per_group;

  auto &encoder = mx::metal::get_command_encoder(stream_);
  encoder.barrier();
  encoder.set_compute_pipeline_state(clear_scores_pipeline_);
  encoder.set_output_array(attention_scores_, 0);
  encoder.set_bytes(score_elements, 1);
  encoder.dispatch_threads(
      MTL::Size(score_elements, 1, 1), MTL::Size(256, 1, 1));

  encoder.set_compute_pipeline_state(query_scale_pipeline_);
  encoder.set_input_array(attention_query, 0);
  encoder.set_output_array(scaled_queries_, 1);
  encoder.set_bytes(attention_scale, 2);
  encoder.set_bytes(attention_query_rows_, 3);
  encoder.dispatch_threads(
      MTL::Size(
          attention_query_rows_ * kHeads * kLatentDim, 1, 1),
      MTL::Size(256, 1, 1));
  encoder.barrier();

  for (int tile = 0; tile < tile_count_; ++tile) {
    const int tile_offset = tile * tile_rows_;
    encoder.set_compute_pipeline_state(gather_tile_pipeline_);
    encoder.set_input_array(latent, 0);
    encoder.set_input_array(union_indices, 1);
    encoder.set_input_array(union_count, 2);
    encoder.set_output_array(union_latent_tile_, 3);
    encoder.set_bytes(physical_k_, 4);
    encoder.set_bytes(tile_offset, 5);
    encoder.set_bytes(tile_rows_, 6);
    encoder.dispatch_threadgroups(
        MTL::Size(tile_rows_ / 8, 1, 1), MTL::Size(256, 1, 1));
    encoder.barrier();

    encoder.set_compute_pipeline_state(projection_pipeline_);
    encoder.set_input_array(union_latent_tile_, 0);
    encoder.set_input_array(key_weight, 1);
    encoder.set_output_array(projected_union_key_tile_, 3);
    encoder.set_bytes(projection_params, 4);
    encoder.dispatch_threadgroups(
        MTL::Size(tiles_n, tiles_m, kHeads), MTL::Size(32, wn, 1));
    encoder.barrier();

    for (int row = 0; row < attention_query_rows_; ++row) {
      encoder.set_compute_pipeline_state(projected_qk_pipeline_);
      encoder.set_input_array(projected_union_key_tile_, 0);
      encoder.set_input_array(scaled_queries_, 1);
      encoder.set_input_array(query_union_slots, 2);
      encoder.set_input_array(selected_valid, 3);
      encoder.set_input_array(union_count, 4);
      encoder.set_output_array(attention_scores_, 5);
      encoder.set_bytes(row, 6);
      encoder.set_bytes(tile_offset, 7);
      encoder.set_bytes(tile_rows_, 8);
      encoder.dispatch_threadgroups(
          MTL::Size(qk_groups, 1, kHeads), MTL::Size(32, 4, 1));
    }
    encoder.barrier();
  }

  ++execution_count_;
  if (buffer_identities() != initial_buffer_identities_) {
    throw std::runtime_error("projected-QK tile-loop buffer identity changed");
  }
  return attention_scores_;
}

mx::array NativeProjectedQKUnionTileLoopPlan::execute_probabilities(
    const mx::array &selected_indices, const mx::array &selected_valid,
    const mx::array &latent, const mx::array &key_weight,
    const mx::array &attention_query, float attention_scale) {
  if (!softmax_enabled_) {
    throw std::runtime_error(
        "projected-QK probability arena was not enabled at construction");
  }
  execute(
      selected_indices, selected_valid, latent, key_weight, attention_query,
      attention_scale);

  constexpr int softmax_reads = 4;
  constexpr int softmax_simd = 32;
  constexpr int threads_needed =
      (kSelectedWidth + softmax_reads - 1) / softmax_reads;
  constexpr int groups_needed =
      (threads_needed + softmax_simd - 1) / softmax_simd;
  constexpr int group_size = softmax_simd * groups_needed;
  constexpr int64_t row_bytes =
      static_cast<int64_t>(kHeads) * kSelectedWidth * sizeof(uint16_t);

  auto &encoder = mx::metal::get_command_encoder(stream_);
  encoder.barrier();
  for (int row = 0; row < attention_query_rows_; ++row) {
    encoder.set_compute_pipeline_state(softmax_pipeline_);
    encoder.set_input_array(attention_scores_, 0, row * row_bytes);
    encoder.set_output_array(attention_probabilities_, 1, row * row_bytes);
    encoder.set_bytes(kSelectedWidth, 2);
    encoder.dispatch_threads(
        MTL::Size(kHeads * group_size, 1, 1),
        MTL::Size(group_size, 1, 1));
  }
  encoder.barrier();

  if (buffer_identities() != initial_buffer_identities_) {
    throw std::runtime_error(
        "projected-QK softmax buffer identity changed");
  }
  return attention_probabilities_;
}

mx::array NativeProjectedQKUnionTileLoopPlan::execute_attention(
    const mx::array &selected_indices, const mx::array &selected_valid,
    const mx::array &latent, const mx::array &key_weight,
    const mx::array &value_weight, const mx::array &attention_query,
    float attention_scale) {
  if (!value_enabled_) {
    throw std::runtime_error(
        "projected-QK value arena was not enabled at construction");
  }
  validate_input(
      value_weight, "value_weight", mx::bfloat16,
      static_cast<size_t>(kHeads) * kValueDim * kLatentDim);
  execute_probabilities(
      selected_indices, selected_valid, latent, key_weight, attention_query,
      attention_scale);

  constexpr int bm = 64;
  constexpr int bn = 64;
  constexpr int bk = 16;
  constexpr int wn = 2;
  const int tiles_m = tile_rows_ / bm;
  constexpr int tiles_n = kValueDim / bn;
  mlx::steel::GEMMParams projection_params{
      tile_rows_, kValueDim, kLatentDim,
      kLatentDim, kLatentDim, kValueDim,
      tiles_n, tiles_m, 0,
      static_cast<int64_t>(kValueDim) * kLatentDim,
      static_cast<int64_t>(tile_rows_) * kValueDim,
      0, kLatentDim / bk, 1};
  const uint32_t selected_edges = static_cast<uint32_t>(
      attention_query_rows_ * kSelectedWidth);
  constexpr int64_t probability_row_bytes =
      static_cast<int64_t>(kHeads) * kSelectedWidth * sizeof(uint16_t);
  constexpr int64_t selected_value_row_bytes =
      static_cast<int64_t>(kHeads) * kSelectedWidth * kValueDim *
      sizeof(uint16_t);
  constexpr int64_t index_row_bytes =
      static_cast<int64_t>(kSelectedWidth) * sizeof(int32_t);
  constexpr int64_t valid_row_bytes =
      static_cast<int64_t>(kSelectedWidth) * sizeof(bool);
  constexpr int64_t output_row_bytes =
      static_cast<int64_t>(kHeads) * kValueDim * sizeof(uint16_t);

  const auto &union_indices = union_plan_.union_indices();
  const auto &union_count = union_plan_.union_count();
  const auto &query_union_slots = union_plan_.query_union_slots();
  auto &encoder = mx::metal::get_command_encoder(stream_);
  encoder.barrier();
  for (int tile = 0; tile < tile_count_; ++tile) {
    const int tile_offset = tile * tile_rows_;
    encoder.set_compute_pipeline_state(gather_tile_pipeline_);
    encoder.set_input_array(latent, 0);
    encoder.set_input_array(union_indices, 1);
    encoder.set_input_array(union_count, 2);
    encoder.set_output_array(union_latent_tile_, 3);
    encoder.set_bytes(physical_k_, 4);
    encoder.set_bytes(tile_offset, 5);
    encoder.set_bytes(tile_rows_, 6);
    encoder.dispatch_threadgroups(
        MTL::Size(tile_rows_ / 8, 1, 1), MTL::Size(256, 1, 1));
    encoder.barrier();

    encoder.set_compute_pipeline_state(value_projection_pipeline_);
    encoder.set_input_array(union_latent_tile_, 0);
    encoder.set_input_array(value_weight, 1);
    encoder.set_output_array(projected_union_value_tile_, 3);
    encoder.set_bytes(projection_params, 4);
    encoder.dispatch_threadgroups(
        MTL::Size(tiles_n, tiles_m, kHeads), MTL::Size(32, wn, 1));
    encoder.barrier();

    encoder.set_compute_pipeline_state(scatter_value_pipeline_);
    encoder.set_input_array(projected_union_value_tile_, 0);
    encoder.set_input_array(query_union_slots, 1);
    encoder.set_input_array(selected_valid, 2);
    encoder.set_input_array(union_count, 3);
    encoder.set_output_array(selected_values_, 4);
    encoder.set_bytes(selected_edges, 5);
    encoder.set_bytes(tile_offset, 6);
    encoder.set_bytes(tile_rows_, 7);
    encoder.dispatch_threadgroups(
        MTL::Size(selected_edges, 1, 1), MTL::Size(128, 1, 1));
    encoder.barrier();
  }

  for (int row = 0; row < attention_query_rows_; ++row) {
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
        attention_probabilities_, 0, row * probability_row_bytes);
    encoder.set_input_array(
        selected_values_, 1, row * selected_value_row_bytes);
    encoder.set_input_array(lane_to_selected_, 2);
    encoder.set_output_array(
        attention_output_, 3, row * output_row_bytes);
    encoder.set_bytes(packed_k_, 4);
    encoder.dispatch_threadgroups(
        MTL::Size(1, 1, kHeads), MTL::Size(32, 4, 1));
    encoder.barrier();
  }

  if (buffer_identities() != initial_buffer_identities_) {
    throw std::runtime_error("projected-QK value buffer identity changed");
  }
  return attention_output_;
}

uint64_t NativeProjectedQKUnionTileLoopPlan::scratch_bytes() const {
  return union_plan_.scratch_bytes() + union_latent_tile_.nbytes() +
      projected_union_key_tile_.nbytes() + scaled_queries_.nbytes() +
      attention_scores_.nbytes() + attention_probabilities_.nbytes() +
      projected_union_value_tile_.nbytes() + selected_values_.nbytes() +
      lane_to_selected_.nbytes() + attention_output_.nbytes();
}

std::vector<uint64_t>
NativeProjectedQKUnionTileLoopPlan::buffer_identities() const {
  auto result = union_plan_.buffer_identities();
  result.push_back(buffer_identity(union_latent_tile_));
  result.push_back(buffer_identity(projected_union_key_tile_));
  result.push_back(buffer_identity(scaled_queries_));
  result.push_back(buffer_identity(attention_scores_));
  result.push_back(buffer_identity(attention_probabilities_));
  result.push_back(buffer_identity(projected_union_value_tile_));
  result.push_back(buffer_identity(selected_values_));
  result.push_back(buffer_identity(lane_to_selected_));
  result.push_back(buffer_identity(attention_output_));
  return result;
}

} // namespace glm53::native_execution
