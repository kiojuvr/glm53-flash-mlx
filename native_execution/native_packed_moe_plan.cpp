#include "native_packed_moe_plan.h"

#include <dlfcn.h>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/primitives.h"

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

int checked_dimension(int value, const char *name) {
  if (value <= 0 || value % 128 != 0) {
    throw std::invalid_argument(std::string(name) +
                                " must be positive and block-128 aligned");
  }
  return value;
}

} // namespace

class NativePackedMoELazyPrimitive : public mx::Primitive {
public:
  NativePackedMoELazyPrimitive(mx::Stream stream,
                               NativePackedMoEDecodePlan *plan)
      : mx::Primitive(stream), plan_(plan) {}

  void eval_cpu(const std::vector<mx::array> &,
                std::vector<mx::array> &) override {
    throw std::runtime_error("native packed MoE lazy primitive is GPU-only");
  }

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override {
    if (inputs.size() != 3 || outputs.size() != 1 || !plan_->weights_bound()) {
      throw std::runtime_error("native packed MoE lazy graph is invalid");
    }
    outputs[0].set_data(mx::allocator::malloc(outputs[0].nbytes()));
    NativePackedMoEDecodePlan::WeightRefs weights{
        plan_->bound_weights_[0], plan_->bound_weights_[1],
        plan_->bound_weights_[2], plan_->bound_weights_[3],
        plan_->bound_weights_[4], plan_->bound_weights_[5],
        plan_->bound_weights_[6], plan_->bound_weights_[7],
        plan_->bound_weights_[8], plan_->bound_weights_[9],
    };
    plan_->encode(inputs[0], inputs[1], inputs[2], weights,
                  plan_->bound_pipelines_, outputs[0]);
  }

  std::vector<mx::Shape>
  output_shapes(const std::vector<mx::array> &) override {
    return {{1, 1, plan_->hidden_size()}};
  }

  const char *name() const override { return "NativePackedMoELazyPrimitive"; }

private:
  NativePackedMoEDecodePlan *plan_;
};

NativePackedMoEDecodePlan::NativePackedMoEDecodePlan(
    int hidden_size, int intermediate_size, int shared_intermediate_size,
    int expert_count, int swiglu_limit)
    : hidden_size_(checked_dimension(hidden_size, "hidden_size")),
      intermediate_size_(
          checked_dimension(intermediate_size, "intermediate_size")),
      shared_intermediate_size_(checked_dimension(shared_intermediate_size,
                                                   "shared_intermediate_size")),
      expert_count_(expert_count), swiglu_limit_(swiglu_limit),
      hidden_scale_rows_(hidden_size_ / kBlockSize),
      intermediate_scale_rows_(intermediate_size_ / kBlockSize),
      shared_scale_rows_(shared_intermediate_size_ / kBlockSize),
      stream_(mx::default_stream(mx::Device(mx::Device::gpu))),
      routed_hidden_(
          owned_array({kTopK, intermediate_size_}, mx::bfloat16)),
      routed_down_(owned_array({kTopK, hidden_size_}, mx::bfloat16)),
      routed_output_(owned_array({hidden_size_}, mx::bfloat16)),
      shared_hidden_(
          owned_array({shared_intermediate_size_}, mx::bfloat16)),
      shared_down_(owned_array({hidden_size_}, mx::bfloat16)),
      output_(owned_array({1, 1, hidden_size_}, mx::bfloat16)) {
  if (expert_count_ < kTopK) {
    throw std::invalid_argument("expert_count must cover selected top-8");
  }
  if (swiglu_limit_ <= 0) {
    throw std::invalid_argument("SwiGLU limit must be a positive integer");
  }
  initial_buffer_identities_ = buffer_identities();
}

void NativePackedMoEDecodePlan::validate_input(
    const mx::array &array, const char *name, mx::Dtype dtype,
    size_t elements) const {
  validate_input_descriptor(array, name, dtype, elements);
  if (array.status() == mx::array::Status::unscheduled) {
    throw std::invalid_argument(std::string(name) +
                                " must be scheduled before native submission");
  }
}

void NativePackedMoEDecodePlan::validate_input_descriptor(
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
}

mx::array NativePackedMoEDecodePlan::execute(
    const mx::array &x, const mx::array &expert_ids, const mx::array &scores,
    const mx::array &gate_up_weight, const mx::array &gate_up_scale_inv,
    const mx::array &down_weight, const mx::array &down_scale_inv,
    const mx::array &shared_gate_weight,
    const mx::array &shared_gate_scale_inv,
    const mx::array &shared_up_weight,
    const mx::array &shared_up_scale_inv,
    const mx::array &shared_down_weight,
    const mx::array &shared_down_scale_inv) {
  dynamic_input_validation_count_ += 3;
  validate_input(x, "x", mx::bfloat16, hidden_size_);
  validate_input(expert_ids, "expert_ids", mx::uint32, kTopK);
  validate_input(scores, "scores", mx::float32, kTopK);
  WeightRefs weights{
      gate_up_weight,          gate_up_scale_inv,    down_weight,
      down_scale_inv,          shared_gate_weight,   shared_gate_scale_inv,
      shared_up_weight,        shared_up_scale_inv,  shared_down_weight,
      shared_down_scale_inv,
  };
  validate_static_weights(weights);
  return encode(x, expert_ids, scores, weights, resolve_pipelines(), output_);
}

void NativePackedMoEDecodePlan::validate_static_weights(
    const WeightRefs &weights) {
  static_input_validation_count_ += 10;
  validate_input(weights.gate_up_weight, "gate_up_weight", mx::uint8,
                 static_cast<size_t>(expert_count_) * 2 * intermediate_size_ *
                     hidden_size_);
  validate_input(weights.gate_up_scale_inv, "gate_up_scale_inv", mx::float32,
                 static_cast<size_t>(expert_count_) * 2 *
                     intermediate_scale_rows_ * hidden_scale_rows_);
  validate_input(weights.down_weight, "down_weight", mx::uint8,
                 static_cast<size_t>(expert_count_) * hidden_size_ *
                     intermediate_size_);
  validate_input(weights.down_scale_inv, "down_scale_inv", mx::float32,
                 static_cast<size_t>(expert_count_) * hidden_scale_rows_ *
                     intermediate_scale_rows_);
  validate_input(weights.shared_gate_weight, "shared_gate_weight", mx::uint8,
                 static_cast<size_t>(shared_intermediate_size_) * hidden_size_);
  validate_input(weights.shared_gate_scale_inv, "shared_gate_scale_inv",
                 mx::float32,
                 static_cast<size_t>(shared_scale_rows_) * hidden_scale_rows_);
  validate_input(weights.shared_up_weight, "shared_up_weight", mx::uint8,
                 static_cast<size_t>(shared_intermediate_size_) * hidden_size_);
  validate_input(weights.shared_up_scale_inv, "shared_up_scale_inv",
                 mx::float32,
                 static_cast<size_t>(shared_scale_rows_) * hidden_scale_rows_);
  validate_input(weights.shared_down_weight, "shared_down_weight", mx::uint8,
                 static_cast<size_t>(hidden_size_) * shared_intermediate_size_);
  validate_input(weights.shared_down_scale_inv, "shared_down_scale_inv",
                 mx::float32,
                 static_cast<size_t>(hidden_scale_rows_) * shared_scale_rows_);
}

std::array<MTL::ComputePipelineState *, 6>
NativePackedMoEDecodePlan::resolve_pipelines() {
  pipeline_lookup_count_ += 6;

  auto &device = mx::metal::device(stream_.device);
  auto *library =
      device.get_library("glm53_native_execution", current_binary_dir());
  const char *routed_gate_up_name = uses_shape_specialized_routed_gate_up()
      ? "glm53_native_glm53_packed_selected8_gate_up_swiglu"
      : "glm53_native_packed_selected8_gate_up_swiglu";
  auto *routed_gate_up = device.get_kernel(routed_gate_up_name, library);
  auto *routed_down =
      device.get_kernel("glm53_native_packed_selected8_down", library);
  auto *aggregate = device.get_kernel(
      "glm53_native_packed_selected8_weighted_reduction", library);
  auto *shared_gate_up = device.get_kernel(
      "glm53_native_shared_gate_up_swiglu", library);
  auto *shared_down =
      device.get_kernel("glm53_native_shared_down", library);
  auto *add = device.get_kernel("glm53_native_add_routed_shared", library);
  return {routed_gate_up, routed_down, aggregate,
          shared_gate_up, shared_down, add};
}

void NativePackedMoEDecodePlan::bind_weights(
    const mx::array &gate_up_weight, const mx::array &gate_up_scale_inv,
    const mx::array &down_weight, const mx::array &down_scale_inv,
    const mx::array &shared_gate_weight,
    const mx::array &shared_gate_scale_inv,
    const mx::array &shared_up_weight,
    const mx::array &shared_up_scale_inv,
    const mx::array &shared_down_weight,
    const mx::array &shared_down_scale_inv) {
  if (weights_bound()) {
    throw std::logic_error("native packed MoE weights are already bound");
  }
  WeightRefs weights{
      gate_up_weight,          gate_up_scale_inv,    down_weight,
      down_scale_inv,          shared_gate_weight,   shared_gate_scale_inv,
      shared_up_weight,        shared_up_scale_inv,  shared_down_weight,
      shared_down_scale_inv,
  };
  validate_static_weights(weights);
  bound_weights_.reserve(10);
  bound_weights_.insert(
      bound_weights_.end(),
      {gate_up_weight, gate_up_scale_inv, down_weight, down_scale_inv,
       shared_gate_weight, shared_gate_scale_inv, shared_up_weight,
       shared_up_scale_inv, shared_down_weight, shared_down_scale_inv});
  initial_bound_weight_identities_ = bound_weight_identities();
  bound_pipelines_ = resolve_pipelines();
}

mx::array NativePackedMoEDecodePlan::execute_bound(
    const mx::array &x, const mx::array &expert_ids,
    const mx::array &scores) {
  if (!weights_bound()) {
    throw std::logic_error("native packed MoE weights are not bound");
  }
  dynamic_input_validation_count_ += 3;
  validate_input(x, "x", mx::bfloat16, hidden_size_);
  validate_input(expert_ids, "expert_ids", mx::uint32, kTopK);
  validate_input(scores, "scores", mx::float32, kTopK);
  if (!bound_weight_identities_stable()) {
    throw std::runtime_error("native packed MoE bound weight identity changed");
  }
  WeightRefs weights{
      bound_weights_[0], bound_weights_[1], bound_weights_[2],
      bound_weights_[3], bound_weights_[4], bound_weights_[5],
      bound_weights_[6], bound_weights_[7], bound_weights_[8],
      bound_weights_[9],
  };
  return encode(x, expert_ids, scores, weights, bound_pipelines_, output_);
}

mx::array NativePackedMoEDecodePlan::execute_lazy(
    const mx::array &x, const mx::array &expert_ids,
    const mx::array &scores) {
  if (!weights_bound()) {
    throw std::logic_error("native packed MoE weights are not bound");
  }
  dynamic_input_validation_count_ += 3;
  validate_input_descriptor(x, "x", mx::bfloat16, hidden_size_);
  validate_input_descriptor(expert_ids, "expert_ids", mx::uint32, kTopK);
  validate_input_descriptor(scores, "scores", mx::float32, kTopK);
  if (!bound_weight_identities_stable()) {
    throw std::runtime_error("native packed MoE bound weight identity changed");
  }
  lazy_graph_count_++;
  auto primitive =
      std::make_shared<NativePackedMoELazyPrimitive>(stream_, this);
  return mx::array({1, 1, hidden_size_}, mx::bfloat16, std::move(primitive),
                   {x, expert_ids, scores});
}

mx::array NativePackedMoEDecodePlan::encode(
    const mx::array &x, const mx::array &expert_ids,
    const mx::array &scores, const WeightRefs &weights,
    const std::array<MTL::ComputePipelineState *, 6> &pipelines,
    mx::array &final_output) {
  auto &encoder = mx::metal::get_command_encoder(stream_);

  encoder.set_compute_pipeline_state(pipelines[0]);
  encoder.set_input_array(x, 0);
  encoder.set_input_array(expert_ids, 1);
  encoder.set_input_array(weights.gate_up_weight, 2);
  encoder.set_input_array(weights.gate_up_scale_inv, 3);
  encoder.set_output_array(routed_hidden_, 4);
  encoder.set_bytes(hidden_size_, 5);
  encoder.set_bytes(intermediate_size_, 6);
  encoder.set_bytes(intermediate_scale_rows_, 7);
  encoder.set_bytes(hidden_scale_rows_, 8);
  encoder.set_bytes(swiglu_limit_, 9);
  encoder.dispatch_threadgroups(
      MTL::Size(kTopK * intermediate_size_, 1, 1),
      MTL::Size(kThreads, 1, 1));
  encoder.barrier();

  encoder.set_compute_pipeline_state(pipelines[1]);
  encoder.set_input_array(routed_hidden_, 0);
  encoder.set_input_array(expert_ids, 1);
  encoder.set_input_array(weights.down_weight, 2);
  encoder.set_input_array(weights.down_scale_inv, 3);
  encoder.set_output_array(routed_down_, 4);
  encoder.set_bytes(intermediate_size_, 5);
  encoder.set_bytes(hidden_size_, 6);
  encoder.set_bytes(intermediate_scale_rows_, 7);
  encoder.dispatch_threadgroups(MTL::Size(kTopK * hidden_size_, 1, 1),
                                MTL::Size(kThreads, 1, 1));
  encoder.barrier();

  encoder.set_compute_pipeline_state(pipelines[2]);
  encoder.set_input_array(routed_down_, 0);
  encoder.set_input_array(scores, 1);
  encoder.set_output_array(routed_output_, 2);
  encoder.set_bytes(hidden_size_, 3);
  encoder.dispatch_threads(MTL::Size(hidden_size_, 1, 1),
                           MTL::Size(kThreads, 1, 1));
  encoder.barrier();

  encoder.set_compute_pipeline_state(pipelines[3]);
  encoder.set_input_array(x, 0);
  encoder.set_input_array(weights.shared_gate_weight, 1);
  encoder.set_input_array(weights.shared_gate_scale_inv, 2);
  encoder.set_input_array(weights.shared_up_weight, 3);
  encoder.set_input_array(weights.shared_up_scale_inv, 4);
  encoder.set_output_array(shared_hidden_, 5);
  encoder.set_bytes(hidden_size_, 6);
  encoder.set_bytes(shared_intermediate_size_, 7);
  encoder.set_bytes(hidden_scale_rows_, 8);
  encoder.set_bytes(swiglu_limit_, 9);
  encoder.dispatch_threadgroups(MTL::Size(shared_intermediate_size_, 1, 1),
                                MTL::Size(kThreads, 1, 1));
  encoder.barrier();

  encoder.set_compute_pipeline_state(pipelines[4]);
  encoder.set_input_array(shared_hidden_, 0);
  encoder.set_input_array(weights.shared_down_weight, 1);
  encoder.set_input_array(weights.shared_down_scale_inv, 2);
  encoder.set_output_array(shared_down_, 3);
  encoder.set_bytes(shared_intermediate_size_, 4);
  encoder.set_bytes(hidden_size_, 5);
  encoder.set_bytes(shared_scale_rows_, 6);
  encoder.dispatch_threadgroups(MTL::Size(hidden_size_, 1, 1),
                                MTL::Size(kThreads, 1, 1));
  encoder.barrier();

  encoder.set_compute_pipeline_state(pipelines[5]);
  encoder.set_input_array(routed_output_, 0);
  encoder.set_input_array(shared_down_, 1);
  encoder.set_output_array(final_output, 2);
  encoder.set_bytes(hidden_size_, 3);
  encoder.dispatch_threads(MTL::Size(hidden_size_, 1, 1),
                           MTL::Size(kThreads, 1, 1));

  execution_count_++;
  if (!scratch_buffer_identities_stable()) {
    throw std::runtime_error(
        "native packed MoE plan buffer identity changed during execute");
  }
  return final_output;
}

bool NativePackedMoEDecodePlan::scratch_buffer_identities_stable() const {
  return initial_buffer_identities_.size() == 6 &&
      buffer_identity(routed_hidden_) == initial_buffer_identities_[0] &&
      buffer_identity(routed_down_) == initial_buffer_identities_[1] &&
      buffer_identity(routed_output_) == initial_buffer_identities_[2] &&
      buffer_identity(shared_hidden_) == initial_buffer_identities_[3] &&
      buffer_identity(shared_down_) == initial_buffer_identities_[4] &&
      buffer_identity(output_) == initial_buffer_identities_[5];
}

bool NativePackedMoEDecodePlan::bound_weight_identities_stable() const {
  if (!weights_bound() || initial_bound_weight_identities_.size() != 10) {
    return false;
  }
  for (size_t index = 0; index < bound_weights_.size(); ++index) {
    if (buffer_identity(bound_weights_[index]) !=
        initial_bound_weight_identities_[index]) {
      return false;
    }
  }
  return true;
}

uint64_t NativePackedMoEDecodePlan::scratch_bytes() const {
  return routed_hidden_.nbytes() + routed_down_.nbytes() +
      routed_output_.nbytes() + shared_hidden_.nbytes() +
      shared_down_.nbytes() + output_.nbytes();
}

std::vector<uint64_t> NativePackedMoEDecodePlan::buffer_identities() const {
  return {buffer_identity(routed_hidden_), buffer_identity(routed_down_),
          buffer_identity(routed_output_), buffer_identity(shared_hidden_),
          buffer_identity(shared_down_), buffer_identity(output_)};
}

std::vector<uint64_t>
NativePackedMoEDecodePlan::bound_weight_identities() const {
  std::vector<uint64_t> identities;
  identities.reserve(bound_weights_.size());
  for (const auto &weight : bound_weights_) {
    identities.push_back(buffer_identity(weight));
  }
  return identities;
}

NativePackedMoERoutedDiagnostic::NativePackedMoERoutedDiagnostic(
    int expert_count)
    : expert_count_(expert_count),
      stream_(mx::default_stream(mx::Device(mx::Device::gpu))),
      gate_(owned_array({kTopK, kIntermediateSize}, mx::bfloat16)),
      up_(owned_array({kTopK, kIntermediateSize}, mx::bfloat16)),
      sigmoid_(owned_array({kTopK, kIntermediateSize}, mx::bfloat16)),
      silu_(owned_array({kTopK, kIntermediateSize}, mx::bfloat16)),
      hidden_(owned_array({kTopK, kIntermediateSize}, mx::bfloat16)) {
  if (expert_count_ < kTopK) {
    throw std::invalid_argument("expert_count must cover selected top-8");
  }
  initial_buffer_identities_ = buffer_identities();
}

mx::array NativePackedMoERoutedDiagnostic::execute(
    const mx::array &x, const mx::array &expert_ids,
    const mx::array &gate_up_weight,
    const mx::array &gate_up_scale_inv) {
  auto require = [](const mx::array &array, const char *name, mx::Dtype dtype,
                    size_t elements) {
    if (array.dtype() != dtype || array.size() != elements ||
        !array.flags().row_contiguous ||
        array.status() == mx::array::Status::unscheduled) {
      throw std::invalid_argument(std::string(name) +
                                  " violates routed diagnostic ABI");
    }
  };
  require(x, "x", mx::bfloat16, kHiddenSize);
  require(expert_ids, "expert_ids", mx::uint32, kTopK);
  require(gate_up_weight, "gate_up_weight", mx::uint8,
          static_cast<size_t>(expert_count_) * 2 * kIntermediateSize *
              kHiddenSize);
  require(gate_up_scale_inv, "gate_up_scale_inv", mx::float32,
          static_cast<size_t>(expert_count_) * 2 *
              (kIntermediateSize / kBlockSize) *
              (kHiddenSize / kBlockSize));

  auto &device = mx::metal::device(stream_.device);
  auto *library =
      device.get_library("glm53_native_execution", current_binary_dir());
  auto *kernel = device.get_kernel(
      "glm53_native_glm53_packed_selected8_gate_up_swiglu_diagnostic",
      library);
  auto &encoder = mx::metal::get_command_encoder(stream_);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(x, 0);
  encoder.set_input_array(expert_ids, 1);
  encoder.set_input_array(gate_up_weight, 2);
  encoder.set_input_array(gate_up_scale_inv, 3);
  encoder.set_output_array(gate_, 4);
  encoder.set_output_array(up_, 5);
  encoder.set_output_array(sigmoid_, 6);
  encoder.set_output_array(silu_, 7);
  encoder.set_output_array(hidden_, 8);
  encoder.dispatch_threadgroups(
      MTL::Size(kTopK * kIntermediateSize, 1, 1), MTL::Size(256, 1, 1));
  execution_count_++;
  if (buffer_identities() != initial_buffer_identities_) {
    throw std::runtime_error(
        "routed diagnostic buffer identity changed during execute");
  }
  return hidden_;
}

uint64_t NativePackedMoERoutedDiagnostic::scratch_bytes() const {
  return gate_.nbytes() + up_.nbytes() + sigmoid_.nbytes() + silu_.nbytes() +
      hidden_.nbytes();
}

std::vector<uint64_t>
NativePackedMoERoutedDiagnostic::buffer_identities() const {
  return {buffer_identity(gate_), buffer_identity(up_),
          buffer_identity(sigmoid_), buffer_identity(silu_),
          buffer_identity(hidden_)};
}

NativeRoutedSigmoidFormulaSweep::NativeRoutedSigmoidFormulaSweep(int elements)
    : elements_(elements),
      stream_(mx::default_stream(mx::Device(mx::Device::gpu))),
      standard_bf16_(owned_array({elements_}, mx::bfloat16)),
      precise_bf16_(owned_array({elements_}, mx::bfloat16)),
      standard_f32_(owned_array({elements_}, mx::bfloat16)),
      precise_f32_(owned_array({elements_}, mx::bfloat16)),
      fast_bf16_(owned_array({elements_}, mx::bfloat16)),
      fast_f32_(owned_array({elements_}, mx::bfloat16)) {
  if (elements_ <= 0) {
    throw std::invalid_argument("sigmoid formula sweep requires elements > 0");
  }
}

mx::array
NativeRoutedSigmoidFormulaSweep::execute(const mx::array &gate) {
  if (gate.dtype() != mx::bfloat16 || gate.size() != elements_ ||
      !gate.flags().row_contiguous ||
      gate.status() == mx::array::Status::unscheduled) {
    throw std::invalid_argument("gate violates sigmoid formula sweep ABI");
  }
  auto &device = mx::metal::device(stream_.device);
  auto *library =
      device.get_library("glm53_native_execution", current_binary_dir());
  auto *kernel =
      device.get_kernel("glm53_native_routed_sigmoid_formula_sweep", library);
  auto &encoder = mx::metal::get_command_encoder(stream_);
  encoder.set_compute_pipeline_state(kernel);
  encoder.set_input_array(gate, 0);
  encoder.set_output_array(standard_bf16_, 1);
  encoder.set_output_array(precise_bf16_, 2);
  encoder.set_output_array(standard_f32_, 3);
  encoder.set_output_array(precise_f32_, 4);
  encoder.set_output_array(fast_bf16_, 5);
  encoder.set_output_array(fast_f32_, 6);
  encoder.set_bytes(elements_, 7);
  encoder.dispatch_threads(MTL::Size(elements_, 1, 1), MTL::Size(256, 1, 1));
  return precise_f32_;
}

} // namespace glm53::native_execution
