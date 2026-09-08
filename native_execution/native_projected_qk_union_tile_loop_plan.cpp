#include "native_projected_qk_union_tile_loop_plan.h"

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
  if (value <= 4096 || value > (32 << 10)) {
    throw std::invalid_argument(
        "multi-tile projected-QK physical K must be in (4096, 32K]");
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
    int physical_k, int attention_query_rows, int tile_rows)
    : physical_k_(checked_physical_k(physical_k)),
      attention_query_rows_(
          checked_attention_query_rows(attention_query_rows)),
      tile_rows_(checked_tile_rows(tile_rows)),
      tile_count_((physical_k_ + tile_rows_ - 1) / tile_rows_),
      stream_(mx::default_stream(mx::Device(mx::Device::gpu))),
      union_plan_(physical_k_),
      union_latent_tile_(
          owned_array({tile_rows_, kLatentDim}, mx::bfloat16)),
      projected_union_key_tile_(owned_array(
          {kHeads, tile_rows_, kLatentDim}, mx::bfloat16)),
      scaled_queries_(owned_array(
          {attention_query_rows_, kHeads, kLatentDim}, mx::bfloat16)),
      attention_scores_(owned_array(
          {attention_query_rows_, kHeads, kSelectedWidth}, mx::bfloat16)) {
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

uint64_t NativeProjectedQKUnionTileLoopPlan::scratch_bytes() const {
  return union_plan_.scratch_bytes() + union_latent_tile_.nbytes() +
      projected_union_key_tile_.nbytes() + scaled_queries_.nbytes() +
      attention_scores_.nbytes();
}

std::vector<uint64_t>
NativeProjectedQKUnionTileLoopPlan::buffer_identities() const {
  auto result = union_plan_.buffer_identities();
  result.push_back(buffer_identity(union_latent_tile_));
  result.push_back(buffer_identity(projected_union_key_tile_));
  result.push_back(buffer_identity(scaled_queries_));
  result.push_back(buffer_identity(attention_scores_));
  return result;
}

} // namespace glm53::native_execution
