#include "native_q256_dsa_prefill_plan.h"

#include <stdexcept>

namespace glm53::native_execution {

NativeQ256DSAPrefillPlan::NativeQ256DSAPrefillPlan(
    int physical_k, int tile_rows)
    : qk_plan_(physical_k, 256, tile_rows, true, false),
      value_plan_(physical_k, tile_rows, 256) {
  initial_buffer_identities_ = buffer_identities();
}

mx::array NativeQ256DSAPrefillPlan::execute(
    const mx::array &selected_indices, const mx::array &selected_valid,
    const mx::array &latent, const mx::array &key_weight,
    const mx::array &value_weight, const mx::array &attention_query,
    float attention_scale) {
  const auto probabilities = qk_plan_.execute_probabilities(
      selected_indices, selected_valid, latent, key_weight, attention_query,
      attention_scale);
  auto output = value_plan_.execute(
      probabilities, selected_indices, selected_valid, latent, value_weight);
  ++execution_count_;
  if (buffer_identities() != initial_buffer_identities_) {
    throw std::runtime_error("Q256 DSA prefill plan buffer identity changed");
  }
  return output;
}

std::vector<uint64_t> NativeQ256DSAPrefillPlan::buffer_identities() const {
  auto result = qk_plan_.buffer_identities();
  const auto value = value_plan_.buffer_identities();
  result.insert(result.end(), value.begin(), value.end());
  return result;
}

} // namespace glm53::native_execution
