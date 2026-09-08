#pragma once

#include <cstdint>
#include <vector>

#include <Metal/Metal.hpp>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/stream.h"

namespace glm53::native_execution {

namespace mx = mlx::core;

// Device-resident physical-order union of Q256 selected token lists.  The
// count remains a GPU buffer so a later projection plan can consume it via an
// indirect dispatch without synchronizing Python or the host executor.
class NativeSelectedUnionPlan {
public:
  explicit NativeSelectedUnionPlan(int physical_k);

  std::vector<mx::array> execute(
      const mx::array &selected_indices, const mx::array &selected_valid);

  int physical_k() const { return physical_k_; }
  int query_rows() const { return kQueryRows; }
  int selected_width() const { return kSelectedWidth; }
  uint64_t execution_count() const { return execution_count_; }
  uint64_t dynamic_allocation_count() const { return 0; }
  uint64_t graph_node_count() const { return 0; }
  uint64_t shape_discovery_count() const { return 0; }
  uint64_t host_synchronization_count() const { return 0; }
  uint64_t scratch_bytes() const;
  std::vector<uint64_t> buffer_identities() const;
  mx::array union_indices() const { return union_indices_; }
  mx::array union_count() const { return union_count_; }
  mx::array query_union_slots() const { return query_union_slots_; }
  mx::array debug_membership_words() const { return membership_words_; }
  mx::array debug_block_counts() const { return block_counts_; }
  mx::array debug_block_prefix() const { return block_prefix_; }

private:
  static constexpr int kQueryRows = 256;
  static constexpr int kSelectedWidth = 2051;
  static constexpr int kTokensPerBlock = 256;

  int physical_k_;
  uint32_t word_count_;
  uint32_t block_count_;
  mx::Stream stream_;
  mx::array membership_words_;
  mx::array block_counts_;
  mx::array block_prefix_;
  mx::array union_indices_;
  mx::array physical_to_union_;
  mx::array query_union_slots_;
  mx::array union_count_;
  MTL::ComputePipelineState *mark_pipeline_{nullptr};
  MTL::ComputePipelineState *count_pipeline_{nullptr};
  MTL::ComputePipelineState *prefix_pipeline_{nullptr};
  MTL::ComputePipelineState *scatter_pipeline_{nullptr};
  MTL::ComputePipelineState *map_pipeline_{nullptr};
  std::vector<uint64_t> initial_buffer_identities_;
  uint64_t execution_count_{0};

  void validate_input(const mx::array &array, const char *name,
                      mx::Dtype dtype) const;
};

} // namespace glm53::native_execution
