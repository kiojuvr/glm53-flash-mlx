#include "native_dsa_score_plan.h"

#include <cmath>
#include <dlfcn.h>
#include <filesystem>
#include <stdexcept>
#include <utility>

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels/steel/gemm/params.h"

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

} // namespace

NativeDSAScoreSelectionPlan::NativeDSAScoreSelectionPlan(
    std::string mode,
    int query_rows,
    int physical_pool_rows,
    float softmax_scale)
    : mode_(std::move(mode)),
      query_rows_(checked_query_rows(mode_, query_rows)),
      physical_pool_rows_(checked_pool_rows(physical_pool_rows)),
      softmax_scale_(softmax_scale),
      stream_(mx::default_stream(mx::Device(mx::Device::gpu))),
      head_scores_(owned_array(
          {query_rows_ * kHeads, physical_pool_rows_}, mx::bfloat16)),
      index_scores_(owned_array(
          {1, query_rows_, physical_pool_rows_}, mx::bfloat16)),
      selected_pool_scratch_(owned_array(
          {1, query_rows_, kSelectedPools}, mx::uint32)),
      selected_token_indices_(owned_array(
          {1, query_rows_, kSelectedWidth}, mx::int32)),
      selected_token_valid_(owned_array(
          {1, query_rows_, kSelectedWidth}, mx::bool_)) {
  if (!std::isfinite(softmax_scale_) || !(softmax_scale_ > 0.0f)) {
    throw std::invalid_argument("softmax scale must be finite and positive");
  }
  auto& device = mx::metal::device(stream_.device);
  if (device.get_architecture().back() != 'd' ||
      device.get_architecture_gen() >= 17) {
    throw std::runtime_error(
        "Tier-1 Steel geometry is qualified only for pre-NAX apple-gpu-d");
  }
  initial_buffer_identities_ = buffer_identities();
}

void NativeDSAScoreSelectionPlan::validate_input(
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

std::vector<mx::array> NativeDSAScoreSelectionPlan::execute(
    const mx::array& query,
    const mx::array& mixture_weights,
    const mx::array& pool_keys,
    const mx::array& pool_indices,
    const mx::array& pool_valid,
    const mx::array& raw_positions,
    const mx::array& raw_valid,
    const mx::array& current_valid,
    int logical_pool_rows,
    int kv_len,
    int active_tail_count) {
  validate_input(
      query,
      "query",
      mx::bfloat16,
      static_cast<size_t>(query_rows_) * kHeads * kHeadDim);
  validate_input(
      mixture_weights,
      "mixture_weights",
      mx::bfloat16,
      static_cast<size_t>(query_rows_) * kHeads);
  validate_input(
      pool_keys,
      "pool_keys",
      mx::bfloat16,
      static_cast<size_t>(physical_pool_rows_) * kHeadDim);
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
  validate_input(
      current_valid,
      "current_valid",
      mx::bool_,
      static_cast<size_t>(query_rows_));
  if (raw_positions.dtype() != mx::int64 ||
      raw_positions.shape(-1) < kIndexKPool) {
    throw std::invalid_argument("raw_positions must contain the active pool");
  }
  if (raw_valid.dtype() != mx::bool_ ||
      raw_valid.size() != raw_positions.size()) {
    throw std::invalid_argument("raw_valid must match raw_positions");
  }
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
  constexpr int bm = 64;
  constexpr int bn = 64;
  constexpr int bk = 16;
  constexpr int wm = 1;
  constexpr int wn = 2;
  const int matrix_rows = query_rows_ * kHeads;
  const int tiles_n = (physical_pool_rows_ + bn - 1) / bn;
  const int tiles_m = (matrix_rows + bm - 1) / bm;
  const bool has_batch = false;
  const bool use_out_source = false;
  const bool do_axpby = false;
  const bool align_m = matrix_rows % bm == 0;
  const bool align_n = physical_pool_rows_ % bn == 0;
  const bool align_k = kHeadDim % bk == 0;
  mx::metal::MTLFCList gemm_constants = {
      {&has_batch, MTL::DataType::DataTypeBool, 10},
      {&use_out_source, MTL::DataType::DataTypeBool, 100},
      {&do_axpby, MTL::DataType::DataTypeBool, 110},
      {&align_m, MTL::DataType::DataTypeBool, 200},
      {&align_n, MTL::DataType::DataTypeBool, 201},
      {&align_k, MTL::DataType::DataTypeBool, 202},
  };
  const std::string gemm_name =
      "glm53_native_steel_gemm_nt_bfloat16_bfloat16_bm64_bn64_bk16_wm1_wn2";
  const std::string gemm_hash = gemm_name +
      "_has_batch_f_use_out_source_f_do_axpby_f_align_M_" +
      (align_m ? "t" : "f") + "_align_N_t_align_K_t";
  auto* gemm = device.get_kernel(
      gemm_name, library, gemm_hash, gemm_constants);
  auto* score = device.get_kernel(
      "glm53_native_finish_pooled_score_bfloat16", library);
  auto* topk = device.get_kernel(
      "glm53_native_exact_partial_topk_512_bfloat16", library);
  auto* expand = device.get_kernel(
      "glm53_native_expand_selected_pools", library);
  auto& encoder = mx::metal::get_command_encoder(stream_);

  mlx::steel::GEMMParams gemm_params{
      matrix_rows,
      physical_pool_rows_,
      kHeadDim,
      kHeadDim,
      kHeadDim,
      physical_pool_rows_,
      tiles_n,
      tiles_m,
      0,
      0,
      static_cast<int64_t>(matrix_rows) * physical_pool_rows_,
      0,
      kHeadDim / bk,
      1};
  encoder.set_compute_pipeline_state(gemm);
  encoder.set_input_array(query, 0);
  // Steel's B operand is buffer(1); pool keys are logically transposed by the
  // instantiated nt loader, without creating a view or a copy.
  encoder.set_input_array(pool_keys, 1);
  encoder.set_output_array(head_scores_, 3);
  encoder.set_bytes(gemm_params, 4);
  encoder.dispatch_threadgroups(
      MTL::Size(tiles_n, tiles_m, 1), MTL::Size(32, wn, wm));
  encoder.barrier();

  encoder.set_compute_pipeline_state(score);
  encoder.set_input_array(head_scores_, 0);
  encoder.set_input_array(mixture_weights, 1);
  encoder.set_input_array(pool_valid, 2);
  encoder.set_output_array(index_scores_, 3);
  encoder.set_bytes(logical_pool_rows, 4);
  encoder.set_bytes(physical_pool_rows_, 5);
  encoder.set_bytes(softmax_scale_, 6);
  encoder.dispatch_threadgroups(
      MTL::Size(physical_pool_rows_, query_rows_, 1), MTL::Size(kHeads, 1, 1));
  encoder.barrier();

  encoder.set_compute_pipeline_state(topk);
  encoder.set_input_array(index_scores_, 0);
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
    throw std::runtime_error("native DSA plan buffer identity changed during execute");
  }
  return {selected_token_indices_, selected_token_valid_};
}

uint64_t NativeDSAScoreSelectionPlan::scratch_bytes() const {
  return head_scores_.nbytes() + index_scores_.nbytes() +
      selected_pool_scratch_.nbytes();
}

std::vector<uint64_t> NativeDSAScoreSelectionPlan::buffer_identities() const {
  return {
      buffer_identity(head_scores_),
      buffer_identity(index_scores_),
      buffer_identity(selected_pool_scratch_),
      buffer_identity(selected_token_indices_),
      buffer_identity(selected_token_valid_),
  };
}

} // namespace glm53::native_execution
