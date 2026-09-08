#include "native_selected_v_av_plan.h"

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
  if (physical_k <= 0 || physical_k > (512 << 10)) {
    throw std::invalid_argument("physical K must be in [1, 512K]");
  }
  return physical_k;
}

} // namespace

NativeSelectedVProjectionAVPlan::NativeSelectedVProjectionAVPlan(int physical_k)
    : physical_k_(checked_physical_k(physical_k)),
      packed_k_(std::min(
          ((physical_k_ + kBK - 1) / kBK) * kBK, kMaximumPackedK)),
      stream_(mx::default_stream(mx::Device(mx::Device::gpu))),
      projected_value_(owned_array(
          {kHeads, kSelectedWidth, kValueDim}, mx::bfloat16)),
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
  projection_pipeline_ = device.get_kernel(
      projection_name, library, projection_hash, constants);
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

uint64_t NativeSelectedVProjectionAVPlan::scratch_bytes() const {
  return projected_value_.nbytes() + lane_to_selected_.nbytes();
}

std::vector<uint64_t> NativeSelectedVProjectionAVPlan::buffer_identities() const {
  return {
      buffer_identity(projected_value_),
      buffer_identity(lane_to_selected_),
      buffer_identity(output_),
  };
}

} // namespace glm53::native_execution
