#include "native_shared_physical_value_tile_plan.h"

#include <algorithm>
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
  if (physical_k < 4096 || physical_k > (512 << 10) || physical_k % 16 != 0) {
    throw std::invalid_argument(
        "shared physical value pass K must be BK16-aligned in [4096, 512K]");
  }
  return physical_k;
}

int checked_tile_rows(int physical_k, int tile_rows) {
  if (tile_rows == 0) tile_rows = std::min(physical_k, 65536);
  if (tile_rows < 4096 || tile_rows > 65536 ||
      (tile_rows & (tile_rows - 1)) != 0 || physical_k % tile_rows != 0) {
    throw std::invalid_argument(
        "shared value tile rows must be a power of two in [4096, 65536] "
        "that evenly divides physical K");
  }
  return tile_rows;
}

int checked_query_rows(int query_rows) {
  if (query_rows != 64 && query_rows != 256) {
    throw std::invalid_argument(
        "shared physical value pass query rows must be 64 or 256");
  }
  return query_rows;
}

} // namespace

NativeSharedPhysicalValueTilePlan::NativeSharedPhysicalValueTilePlan(
    int physical_k, int tile_rows, int query_rows)
    : physical_k_(checked_physical_k(physical_k)),
      tile_rows_(checked_tile_rows(physical_k_, tile_rows)),
      tile_count_(physical_k_ / tile_rows_),
      query_rows_(checked_query_rows(query_rows)),
      stream_(mx::default_stream(mx::Device(mx::Device::gpu))),
      physical_probabilities_(owned_array(
          {kHeads, kQueryBlockRows, tile_rows_}, mx::bfloat16)),
      projected_values_(owned_array(
          {kHeads, tile_rows_, kValueDim}, mx::bfloat16)),
      output_(owned_array({kHeads, query_rows_, kValueDim}, mx::bfloat16)),
      fp32_accumulator_(owned_array(
          {kHeads, query_rows_, kValueDim}, mx::float32)) {
  auto &device = mx::metal::device(stream_.device);
  if (device.get_architecture().back() != 'd' ||
      device.get_architecture_gen() >= 17) {
    throw std::runtime_error(
        "shared physical value tile is qualified only for pre-NAX apple-gpu-d");
  }
  auto *library = device.get_library(
      "glm53_native_execution", current_binary_dir());
  clear_pipeline_ = device.get_kernel(
      "glm53_native_clear_bfloat16", library);
  scatter_pipeline_ = device.get_kernel(
      "glm53_native_scatter_bm64_probabilities_to_physical_bfloat16", library);

  // Steel advances B/D by the z grid coordinate using GEMMParams strides.
  // `has_batch` is reserved for the optional multidimensional batch-shape
  // buffers, which this fixed one-dimensional head batch deliberately omits.
  const bool projection_has_batch = false;
  const bool use_out_source = false;
  const bool do_axpby = false;
  const bool align_m = true;
  const bool align_n = true;
  const bool align_k = true;
  mx::metal::MTLFCList projection_constants = {
      {&projection_has_batch, MTL::DataType::DataTypeBool, 10},
      {&use_out_source, MTL::DataType::DataTypeBool, 100},
      {&do_axpby, MTL::DataType::DataTypeBool, 110},
      {&align_m, MTL::DataType::DataTypeBool, 200},
      {&align_n, MTL::DataType::DataTypeBool, 201},
      {&align_k, MTL::DataType::DataTypeBool, 202},
  };
  const std::string projection_name =
      "glm53_native_steel_gemm_nt_bfloat16_bfloat16_bm64_bn64_bk16_wm1_wn2";
  projection_pipeline_ = device.get_kernel(
      projection_name, library,
      projection_name +
          "_has_batch_f_use_out_source_f_do_axpby_f_align_M_t_align_N_t_align_K_t",
      projection_constants);

  av_pipeline_ = device.get_kernel(
      "glm53_native_bm64_physical_value_tile_continue_bfloat16", library);
  initial_buffer_identities_ = buffer_identities();
}

void NativeSharedPhysicalValueTilePlan::validate_input(
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
        std::string(name) + " must be scheduled before native submission");
  }
}

mx::array NativeSharedPhysicalValueTilePlan::execute(
    const mx::array &selected_probabilities,
    const mx::array &selected_indices, const mx::array &selected_valid,
    const mx::array &latent, const mx::array &value_weight) {
  validate_input(
      selected_probabilities, "selected_probabilities", mx::bfloat16,
      static_cast<size_t>(query_rows_) * kHeads * kSelectedWidth);
  validate_input(
      selected_indices, "selected_indices", mx::int32,
      static_cast<size_t>(query_rows_) * kSelectedWidth);
  validate_input(
      selected_valid, "selected_valid", mx::bool_,
      static_cast<size_t>(query_rows_) * kSelectedWidth);
  validate_input(
      latent, "latent", mx::bfloat16,
      static_cast<size_t>(physical_k_) * kLatentDim);
  validate_input(
      value_weight, "value_weight", mx::bfloat16,
      static_cast<size_t>(kHeads) * kValueDim * kLatentDim);

  constexpr int bm = 64;
  constexpr int bn = 64;
  constexpr int bk = 16;
  constexpr int projection_wn = 2;
  const int projection_tiles_m = tile_rows_ / bm;
  constexpr int projection_tiles_n = kValueDim / bn;
  mlx::steel::GEMMParams projection_params{
      tile_rows_, kValueDim, kLatentDim,
      kLatentDim, kLatentDim, kValueDim,
      projection_tiles_n, projection_tiles_m,
      0,
      static_cast<int64_t>(kValueDim) * kLatentDim,
      static_cast<int64_t>(tile_rows_) * kValueDim,
      0, kLatentDim / bk, 1};

  const uint32_t probability_elements = static_cast<uint32_t>(
      static_cast<uint64_t>(kHeads) * kQueryBlockRows * tile_rows_);
  auto &encoder = mx::metal::get_command_encoder(stream_);
  encoder.barrier();
  for (int tile = 0; tile < tile_count_; ++tile) {
    const int tile_offset = tile * tile_rows_;
    encoder.set_compute_pipeline_state(projection_pipeline_);
    encoder.set_input_array(
        latent, 0,
        static_cast<int64_t>(tile_offset) * kLatentDim * sizeof(uint16_t));
    encoder.set_input_array(value_weight, 1);
    encoder.set_output_array(projected_values_, 3);
    encoder.set_bytes(projection_params, 4);
    encoder.dispatch_threadgroups(
        MTL::Size(projection_tiles_n, projection_tiles_m, kHeads),
        MTL::Size(32, projection_wn, 1));
    encoder.barrier();

    for (int query_offset = 0; query_offset < query_rows_;
         query_offset += kQueryBlockRows) {
      encoder.set_compute_pipeline_state(clear_pipeline_);
      encoder.set_output_array(physical_probabilities_, 0);
      encoder.set_bytes(probability_elements, 1);
      encoder.dispatch_threads(
          MTL::Size(probability_elements, 1, 1), MTL::Size(256, 1, 1));
      encoder.barrier();

      encoder.set_compute_pipeline_state(scatter_pipeline_);
      encoder.set_input_array(selected_probabilities, 0);
      encoder.set_input_array(selected_indices, 1);
      encoder.set_input_array(selected_valid, 2);
      encoder.set_output_array(physical_probabilities_, 3);
      encoder.set_bytes(physical_k_, 4);
      encoder.set_bytes(tile_offset, 5);
      encoder.set_bytes(tile_rows_, 6);
      encoder.set_bytes(query_offset, 7);
      encoder.set_bytes(query_rows_, 8);
      encoder.dispatch_threads(
          MTL::Size(kSelectedWidth, kQueryBlockRows, kHeads),
          MTL::Size(32, 1, 1));
      encoder.barrier();

      const bool first_tile = tile == 0;
      const bool final_tile = tile + 1 == tile_count_;
      encoder.set_compute_pipeline_state(av_pipeline_);
      encoder.set_input_array(physical_probabilities_, 0);
      encoder.set_input_array(projected_values_, 1);
      encoder.set_output_array(fp32_accumulator_, 2);
      encoder.set_output_array(output_, 3);
      encoder.set_bytes(tile_rows_, 4);
      encoder.set_bytes(first_tile, 5);
      encoder.set_bytes(final_tile, 6);
      encoder.set_bytes(query_offset, 7);
      encoder.set_bytes(query_rows_, 8);
      encoder.dispatch_threadgroups(
          MTL::Size(kValueDim / bn, 1, kHeads), MTL::Size(32, 2, 2));
      encoder.barrier();
    }
  }

  ++execution_count_;
  if (buffer_identities() != initial_buffer_identities_) {
    throw std::runtime_error(
        "shared physical value tile buffer identity changed");
  }
  return output_;
}

uint64_t NativeSharedPhysicalValueTilePlan::scratch_bytes() const {
  return physical_probabilities_.nbytes() + projected_values_.nbytes() +
      output_.nbytes() + fp32_accumulator_.nbytes();
}

std::vector<uint64_t>
NativeSharedPhysicalValueTilePlan::buffer_identities() const {
  return {
      buffer_identity(physical_probabilities_),
      buffer_identity(projected_values_),
      buffer_identity(output_),
      buffer_identity(fp32_accumulator_),
  };
}

} // namespace glm53::native_execution
