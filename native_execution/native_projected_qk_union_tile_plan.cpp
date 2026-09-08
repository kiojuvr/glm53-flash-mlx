#include "native_projected_qk_union_tile_plan.h"

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

} // namespace

NativeProjectedQKUnionTilePlan::NativeProjectedQKUnionTilePlan()
    : stream_(mx::default_stream(mx::Device(mx::Device::gpu))),
      projected_union_key_(owned_array(
          {kHeads, kUnionTileRows, kLatentDim}, mx::bfloat16)),
      scaled_query_(owned_array({kHeads, kLatentDim}, mx::bfloat16)),
      attention_scores_(owned_array(
          {kQueryRows, kHeads, kSelectedWidth}, mx::bfloat16)) {
  auto &device = mx::metal::device(stream_.device);
  if (device.get_architecture().back() != 'd' ||
      device.get_architecture_gen() >= 17) {
    throw std::runtime_error(
        "projected-QK tile is qualified only for pre-NAX apple-gpu-d");
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
  projection_pipeline_ = device.get_kernel(
      projection_name, library, projection_hash, constants);
  query_scale_pipeline_ = device.get_kernel(
      "glm53_native_prepare_prefill_attention_query_bfloat16", library);
  projected_qk_pipeline_ = device.get_kernel(
      "glm53_native_projected_union_qk_bfloat16", library);
  initial_buffer_identities_ = buffer_identities();
}

void NativeProjectedQKUnionTilePlan::validate_input(
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

mx::array NativeProjectedQKUnionTilePlan::execute(
    const mx::array &union_latent, const mx::array &key_weight,
    const mx::array &attention_query, const mx::array &query_union_slots,
    const mx::array &selected_valid, float attention_scale) {
  validate_input(
      union_latent, "union_latent", mx::bfloat16,
      static_cast<size_t>(kUnionTileRows) * kLatentDim);
  validate_input(
      key_weight, "key_weight", mx::bfloat16,
      static_cast<size_t>(kHeads) * kLatentDim * kLatentDim);
  validate_input(
      attention_query, "attention_query", mx::bfloat16,
      static_cast<size_t>(kHeads) * kQueryRows * kLatentDim);
  validate_input(
      query_union_slots, "query_union_slots", mx::int32,
      static_cast<size_t>(kQueryRows) * kSelectedWidth);
  validate_input(
      selected_valid, "selected_valid", mx::bool_,
      static_cast<size_t>(kQueryRows) * kSelectedWidth);
  if (!std::isfinite(attention_scale) || !(attention_scale > 0.0f)) {
    throw std::invalid_argument("attention scale must be finite and positive");
  }

  constexpr int bm = 64;
  constexpr int bn = 64;
  constexpr int bk = 16;
  constexpr int wn = 2;
  constexpr int tiles_m = kUnionTileRows / bm;
  constexpr int tiles_n = kLatentDim / bn;
  mlx::steel::GEMMParams projection_params{
      kUnionTileRows, kLatentDim, kLatentDim,
      kLatentDim, kLatentDim, kLatentDim,
      tiles_n, tiles_m, 0,
      static_cast<int64_t>(kLatentDim) * kLatentDim,
      static_cast<int64_t>(kUnionTileRows) * kLatentDim,
      0, kLatentDim / bk, 1};
  constexpr int qk_output_rows_per_group = 16;
  constexpr int qk_groups =
      (kSelectedWidth + qk_output_rows_per_group - 1) /
      qk_output_rows_per_group;

  auto &encoder = mx::metal::get_command_encoder(stream_);
  encoder.set_compute_pipeline_state(projection_pipeline_);
  encoder.set_input_array(union_latent, 0);
  encoder.set_input_array(key_weight, 1);
  encoder.set_output_array(projected_union_key_, 3);
  encoder.set_bytes(projection_params, 4);
  encoder.dispatch_threadgroups(
      MTL::Size(tiles_n, tiles_m, kHeads), MTL::Size(32, wn, 1));
  encoder.barrier();

  for (int row = 0; row < kQueryRows; ++row) {
    encoder.set_compute_pipeline_state(query_scale_pipeline_);
    encoder.set_input_array(attention_query, 0);
    encoder.set_output_array(scaled_query_, 1);
    encoder.set_bytes(row, 2);
    encoder.set_bytes(attention_scale, 3);
    encoder.dispatch_threads(
        MTL::Size(kHeads * kLatentDim, 1, 1), MTL::Size(256, 1, 1));
    encoder.barrier();

    encoder.set_compute_pipeline_state(projected_qk_pipeline_);
    encoder.set_input_array(projected_union_key_, 0);
    encoder.set_input_array(scaled_query_, 1);
    encoder.set_input_array(query_union_slots, 2);
    encoder.set_input_array(selected_valid, 3);
    encoder.set_output_array(attention_scores_, 4);
    encoder.set_bytes(row, 5);
    encoder.dispatch_threadgroups(
        MTL::Size(qk_groups, 1, kHeads), MTL::Size(32, 4, 1));
    encoder.barrier();
  }

  ++execution_count_;
  if (buffer_identities() != initial_buffer_identities_) {
    throw std::runtime_error("projected-QK tile buffer identity changed");
  }
  return attention_scores_;
}

uint64_t NativeProjectedQKUnionTilePlan::scratch_bytes() const {
  return projected_union_key_.nbytes() + scaled_query_.nbytes() +
      attention_scores_.nbytes();
}

std::vector<uint64_t>
NativeProjectedQKUnionTilePlan::buffer_identities() const {
  return {
      buffer_identity(projected_union_key_), buffer_identity(scaled_query_),
      buffer_identity(attention_scores_),
  };
}

} // namespace glm53::native_execution
