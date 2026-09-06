#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/stream.h"

namespace glm53::native_execution {

namespace mx = mlx::core;

class NativeIndexSelectionPlan {
 public:
  NativeIndexSelectionPlan(
      std::string mode,
      int query_rows,
      int physical_pool_rows,
      std::string score_dtype);

  std::vector<mx::array> execute(
      const mx::array& scores,
      const mx::array& pool_indices,
      const mx::array& pool_valid,
      const mx::array& raw_positions,
      const mx::array& raw_valid,
      const mx::array& current_valid,
      int logical_pool_rows,
      int kv_len,
      int active_tail_count);

  const std::string& mode() const { return mode_; }
  int query_rows() const { return query_rows_; }
  int physical_pool_rows() const { return physical_pool_rows_; }
  int selected_width() const { return kSelectedWidth; }
  uint64_t execution_count() const { return execution_count_; }
  uint64_t dynamic_allocation_count() const { return 0; }
  uint64_t graph_node_count() const { return 0; }
  uint64_t shape_discovery_count() const { return 0; }
  uint64_t host_synchronization_count() const { return 0; }
  std::vector<uint64_t> buffer_identities() const;

 private:
  static constexpr int kSelectedPools = 512;
  static constexpr int kIndexKPool = 4;
  static constexpr int kSelectedWidth = 2051;
  static constexpr int kTailWidth = 3;

  std::string mode_;
  int query_rows_;
  int physical_pool_rows_;
  mx::Dtype score_dtype_;
  mx::Stream stream_;
  mx::array selected_pool_scratch_;
  mx::array selected_token_indices_;
  mx::array selected_token_valid_;
  std::vector<uint64_t> initial_buffer_identities_;
  uint64_t execution_count_{0};

  void validate_input(
      const mx::array& array,
      const char* name,
      mx::Dtype dtype,
      size_t elements) const;
};

} // namespace glm53::native_execution
