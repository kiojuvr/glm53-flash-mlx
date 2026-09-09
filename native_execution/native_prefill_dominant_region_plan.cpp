#include "native_prefill_dominant_region_plan.h"

#include <stdexcept>

#include "mlx/backend/metal/device.h"

namespace glm53::native_execution {

NativePrefillDominantRegionPlan::NativePrefillDominantRegionPlan(
    int physical_k, int tile_rows, int expert_count)
    : stream_(mx::default_stream(mx::Device(mx::Device::gpu))),
      dsa_body_(physical_k, tile_rows, 256), dsa_output_(),
      moe_(expert_count) {
  initial_buffer_identities_ = buffer_identities();
}

void NativePrefillDominantRegionPlan::barrier() {
  mx::metal::get_command_encoder(stream_).barrier();
}

mx::array NativePrefillDominantRegionPlan::encode_dsa(
    const mx::array &selected_indices, const mx::array &selected_valid,
    const mx::array &latent, const mx::array &key_weight,
    const mx::array &value_weight, const mx::array &attention_query,
    float attention_scale, const mx::array &output_weight,
    const mx::array &output_scale_inv) {
  auto head_major = dsa_body_.execute(
      selected_indices, selected_valid, latent, key_weight, value_weight,
      attention_query, attention_scale);
  barrier();
  auto output = dsa_output_.execute(
      head_major, output_weight, output_scale_inv);
  // The same owned DSA arenas are reused by the next logical layer.
  barrier();
  return output;
}

mx::array NativePrefillDominantRegionPlan::encode_moe(
    const mx::array &hidden, const mx::array &expert_ids,
    const mx::array &scores, const mx::array &gate_up_weight,
    const mx::array &gate_up_scale_inv, const mx::array &down_weight,
    const mx::array &down_scale_inv, const mx::array &shared_gate_weight,
    const mx::array &shared_gate_scale_inv,
    const mx::array &shared_up_weight,
    const mx::array &shared_up_scale_inv,
    const mx::array &shared_down_weight,
    const mx::array &shared_down_scale_inv) {
  auto output = moe_.execute_indirect(
      hidden, expert_ids, scores, gate_up_weight, gate_up_scale_inv,
      down_weight, down_scale_inv, shared_gate_weight, shared_gate_scale_inv,
      shared_up_weight, shared_up_scale_inv, shared_down_weight,
      shared_down_scale_inv);
  // finish() reads routed/shared scratch which the next layer reuses.
  barrier();
  return output;
}

mx::array NativePrefillDominantRegionPlan::execute_dsa(
    const mx::array &selected_indices, const mx::array &selected_valid,
    const mx::array &latent, const mx::array &key_weight,
    const mx::array &value_weight, const mx::array &attention_query,
    float attention_scale, const mx::array &output_weight,
    const mx::array &output_scale_inv) {
  auto output = encode_dsa(
      selected_indices, selected_valid, latent, key_weight, value_weight,
      attention_query, attention_scale, output_weight, output_scale_inv);
  ++layerwise_execution_count_;
  return output;
}

mx::array NativePrefillDominantRegionPlan::execute_moe(
    const mx::array &hidden, const mx::array &expert_ids,
    const mx::array &scores, const mx::array &gate_up_weight,
    const mx::array &gate_up_scale_inv, const mx::array &down_weight,
    const mx::array &down_scale_inv, const mx::array &shared_gate_weight,
    const mx::array &shared_gate_scale_inv,
    const mx::array &shared_up_weight,
    const mx::array &shared_up_scale_inv,
    const mx::array &shared_down_weight,
    const mx::array &shared_down_scale_inv) {
  auto output = encode_moe(
      hidden, expert_ids, scores, gate_up_weight, gate_up_scale_inv,
      down_weight, down_scale_inv, shared_gate_weight, shared_gate_scale_inv,
      shared_up_weight, shared_up_scale_inv, shared_down_weight,
      shared_down_scale_inv);
  ++layerwise_execution_count_;
  return output;
}

std::vector<mx::array> NativePrefillDominantRegionPlan::execute_all(
    const mx::array &selected_indices, const mx::array &selected_valid,
    const mx::array &latent, const mx::array &key_weight,
    const mx::array &value_weight, const mx::array &attention_query,
    float attention_scale, const mx::array &output_weight,
    const mx::array &output_scale_inv, const mx::array &hidden,
    const mx::array &expert_ids, const mx::array &scores,
    const mx::array &gate_up_weight,
    const mx::array &gate_up_scale_inv, const mx::array &down_weight,
    const mx::array &down_scale_inv, const mx::array &shared_gate_weight,
    const mx::array &shared_gate_scale_inv,
    const mx::array &shared_up_weight,
    const mx::array &shared_up_scale_inv,
    const mx::array &shared_down_weight,
    const mx::array &shared_down_scale_inv) {
  mx::array dsa_terminal = output_weight;
  for (int layer = 0; layer < kDSALayerCount; ++layer) {
    dsa_terminal = encode_dsa(
        selected_indices, selected_valid, latent, key_weight, value_weight,
        attention_query, attention_scale, output_weight, output_scale_inv);
  }
  mx::array moe_terminal = hidden;
  for (int layer = 0; layer < kMoELayerCount; ++layer) {
    moe_terminal = encode_moe(
        hidden, expert_ids, scores, gate_up_weight, gate_up_scale_inv,
        down_weight, down_scale_inv, shared_gate_weight,
        shared_gate_scale_inv, shared_up_weight, shared_up_scale_inv,
        shared_down_weight, shared_down_scale_inv);
  }
  ++composed_execution_count_;
  if (!buffer_identities_stable()) {
    throw std::runtime_error(
        "dominant-region native prefill arena identity changed");
  }
  return {dsa_terminal, moe_terminal};
}

uint64_t
NativePrefillDominantRegionPlan::returned_diagnostic_anchor_bytes() const {
  return static_cast<uint64_t>(256) * 4096 * 2 * 2;
}

uint64_t NativePrefillDominantRegionPlan::scratch_bytes() const {
  return dsa_body_.scratch_bytes() + dsa_output_.scratch_bytes() +
      moe_.scratch_bytes();
}

std::vector<uint64_t>
NativePrefillDominantRegionPlan::buffer_identities() const {
  auto result = dsa_body_.buffer_identities();
  auto output = dsa_output_.buffer_identities();
  auto moe = moe_.buffer_identities();
  result.insert(result.end(), output.begin(), output.end());
  result.insert(result.end(), moe.begin(), moe.end());
  return result;
}

bool NativePrefillDominantRegionPlan::buffer_identities_stable() const {
  return buffer_identities() == initial_buffer_identities_;
}

} // namespace glm53::native_execution
