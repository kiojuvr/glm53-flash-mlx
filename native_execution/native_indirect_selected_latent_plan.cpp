#include "native_indirect_selected_latent_plan.h"

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

// MLX 0.32.2 does not expose indirect dispatch on CommandEncoder.  Access the
// pinned private raw encoder without changing the MLX header or its ABI.  This
// friend-injection is deliberately confined to the probe-only native bridge.
template <typename Tag, typename Tag::type Member>
struct PrivateMemberAccess {
  friend typename Tag::type get_private_member(Tag) { return Member; }
};

struct RawEncoderTag {
  using type = MTL::ComputeCommandEncoder *(
      mx::metal::CommandEncoder::*)();
  friend type get_private_member(RawEncoderTag);
};

template struct PrivateMemberAccess<
    RawEncoderTag, &mx::metal::CommandEncoder::get_command_encoder>;

MTL::ComputeCommandEncoder *raw_encoder(mx::metal::CommandEncoder &encoder) {
  return (encoder.*get_private_member(RawEncoderTag{}))();
}

MTL::Buffer *metal_buffer(mx::array &array) {
  return reinterpret_cast<MTL::Buffer *>(array.buffer().ptr());
}

} // namespace

NativeIndirectSelectedLatentPlan::NativeIndirectSelectedLatentPlan(
    int physical_k)
    : physical_k_(physical_k),
      stream_(mx::default_stream(mx::Device(mx::Device::gpu))),
      union_plan_(physical_k),
      indirect_arguments_(owned_array({3}, mx::uint32)),
      union_latent_(owned_array({physical_k, kLatentDim}, mx::bfloat16)) {
  auto &device = mx::metal::device(stream_.device);
  auto *library = device.get_library(
      "glm53_native_execution", current_binary_dir());
  arguments_pipeline_ = device.get_kernel(
      "glm53_native_build_selected_union_gather_arguments", library);
  gather_pipeline_ = device.get_kernel(
      "glm53_native_gather_selected_union_latent_bfloat16", library);
  initial_buffer_identities_ = buffer_identities();
}

void NativeIndirectSelectedLatentPlan::validate_latent(
    const mx::array &latent) const {
  if (latent.dtype() != mx::bfloat16 ||
      latent.size() != static_cast<size_t>(physical_k_) * kLatentDim) {
    throw std::invalid_argument("latent shape/dtype mismatch");
  }
  if (!latent.flags().row_contiguous) {
    throw std::invalid_argument("latent must be row-contiguous");
  }
  if (latent.status() == mx::array::Status::unscheduled) {
    throw std::invalid_argument("latent must be scheduled before submission");
  }
}

std::vector<mx::array> NativeIndirectSelectedLatentPlan::execute(
    const mx::array &selected_indices, const mx::array &selected_valid,
    const mx::array &latent) {
  validate_latent(latent);
  const auto union_outputs = union_plan_.execute(
      selected_indices, selected_valid);
  const auto &union_indices = union_outputs[0];
  const auto &union_count = union_outputs[1];

  auto &encoder = mx::metal::get_command_encoder(stream_);
  encoder.barrier();
  encoder.set_compute_pipeline_state(arguments_pipeline_);
  encoder.set_input_array(union_count, 0);
  encoder.set_output_array(indirect_arguments_, 1);
  encoder.dispatch_threads(MTL::Size(1, 1, 1), MTL::Size(1, 1, 1));
  encoder.barrier();

  encoder.set_compute_pipeline_state(gather_pipeline_);
  encoder.set_input_array(latent, 0);
  encoder.set_input_array(union_indices, 1);
  encoder.set_input_array(union_count, 2);
  encoder.set_output_array(union_latent_, 3);
  encoder.set_bytes(physical_k_, 4);
  raw_encoder(encoder)->dispatchThreadgroups(
      metal_buffer(indirect_arguments_), 0,
      MTL::Size(kThreadsPerGroup, 1, 1));

  ++execution_count_;
  if (buffer_identities() != initial_buffer_identities_) {
    throw std::runtime_error("indirect union-latent buffer identity changed");
  }
  return {union_indices, union_count, union_latent_};
}

uint64_t NativeIndirectSelectedLatentPlan::scratch_bytes() const {
  return union_plan_.scratch_bytes() + indirect_arguments_.nbytes() +
      union_latent_.nbytes();
}

std::vector<uint64_t>
NativeIndirectSelectedLatentPlan::buffer_identities() const {
  auto result = union_plan_.buffer_identities();
  result.push_back(buffer_identity(indirect_arguments_));
  result.push_back(buffer_identity(union_latent_));
  return result;
}

} // namespace glm53::native_execution
