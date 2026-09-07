#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "native_dsa_attention_plan.h"
#include "native_dsa_score_plan.h"
#include "native_indexpool_update_plan.h"
#include "native_indexer_plan.h"
#include "native_packed_moe_plan.h"

namespace nb = nanobind;
using namespace nb::literals;
using glm53::native_execution::NativeDSAScoreSelectionPlan;
using glm53::native_execution::NativeDSASparseAttentionPlan;
using glm53::native_execution::NativeIndexSelectionPlan;
using glm53::native_execution::NativeIndexPoolUpdateSelectionPlan;
using glm53::native_execution::NativePackedMoEDecodePlan;

NB_MODULE(_ext, module) {
  module.doc() = "Probe-only persistent native execution bridge for GLM-5.3";
  nb::class_<NativeIndexSelectionPlan>(module, "NativeIndexSelectionPlan")
      .def(nb::init<std::string, int, int, std::string>(), "mode"_a,
           "query_rows"_a, "physical_pool_rows"_a, "score_dtype"_a)
      .def("execute", &NativeIndexSelectionPlan::execute, "scores"_a,
           "pool_indices"_a, "pool_valid"_a, "raw_positions"_a, "raw_valid"_a,
           "current_valid"_a, "logical_pool_rows"_a, "kv_len"_a,
           "active_tail_count"_a)
      .def_prop_ro("mode", &NativeIndexSelectionPlan::mode)
      .def_prop_ro("query_rows", &NativeIndexSelectionPlan::query_rows)
      .def_prop_ro("physical_pool_rows",
                   &NativeIndexSelectionPlan::physical_pool_rows)
      .def_prop_ro("selected_width", &NativeIndexSelectionPlan::selected_width)
      .def_prop_ro("execution_count",
                   &NativeIndexSelectionPlan::execution_count)
      .def_prop_ro("dynamic_allocation_count",
                   &NativeIndexSelectionPlan::dynamic_allocation_count)
      .def_prop_ro("graph_node_count",
                   &NativeIndexSelectionPlan::graph_node_count)
      .def_prop_ro("shape_discovery_count",
                   &NativeIndexSelectionPlan::shape_discovery_count)
      .def_prop_ro("host_synchronization_count",
                   &NativeIndexSelectionPlan::host_synchronization_count)
      .def_prop_ro("buffer_identities",
                   &NativeIndexSelectionPlan::buffer_identities);

  nb::class_<NativeDSAScoreSelectionPlan>(module, "NativeDSAScoreSelectionPlan")
      .def(nb::init<std::string, int, int, float>(), "mode"_a, "query_rows"_a,
           "physical_pool_rows"_a, "softmax_scale"_a)
      .def("execute", &NativeDSAScoreSelectionPlan::execute, "query"_a,
           "mixture_weights"_a, "pool_keys"_a, "pool_indices"_a, "pool_valid"_a,
           "raw_positions"_a, "raw_valid"_a, "current_valid"_a,
           "logical_pool_rows"_a, "kv_len"_a, "active_tail_count"_a)
      .def_prop_ro("mode", &NativeDSAScoreSelectionPlan::mode)
      .def_prop_ro("query_rows", &NativeDSAScoreSelectionPlan::query_rows)
      .def_prop_ro("physical_pool_rows",
                   &NativeDSAScoreSelectionPlan::physical_pool_rows)
      .def_prop_ro("selected_width",
                   &NativeDSAScoreSelectionPlan::selected_width)
      .def_prop_ro("execution_count",
                   &NativeDSAScoreSelectionPlan::execution_count)
      .def_prop_ro("dynamic_allocation_count",
                   &NativeDSAScoreSelectionPlan::dynamic_allocation_count)
      .def_prop_ro("graph_node_count",
                   &NativeDSAScoreSelectionPlan::graph_node_count)
      .def_prop_ro("shape_discovery_count",
                   &NativeDSAScoreSelectionPlan::shape_discovery_count)
      .def_prop_ro("host_synchronization_count",
                   &NativeDSAScoreSelectionPlan::host_synchronization_count)
      .def_prop_ro("returned_score_tensor_bytes",
                   &NativeDSAScoreSelectionPlan::returned_score_tensor_bytes)
      .def_prop_ro("scratch_bytes", &NativeDSAScoreSelectionPlan::scratch_bytes)
      .def_prop_ro("debug_head_scores",
                   &NativeDSAScoreSelectionPlan::debug_head_scores,
                   "Diagnostic-only Steel GEMM output view")
      .def_prop_ro(
          "debug_index_scores",
          &NativeDSAScoreSelectionPlan::debug_index_scores,
          "Diagnostic-only view; execute() never returns score scratch")
      .def_prop_ro("buffer_identities",
                   &NativeDSAScoreSelectionPlan::buffer_identities);

  nb::class_<NativeIndexPoolUpdateSelectionPlan>(
      module, "NativeIndexPoolUpdateSelectionPlan")
      .def(nb::init<int, float>(), "physical_pool_rows"_a,
           "softmax_scale"_a)
      .def("execute", &NativeIndexPoolUpdateSelectionPlan::execute,
           "key"_a, "gate"_a, "current_valid"_a, "query"_a,
           "mixture_weights"_a, "pool_keys"_a, "pool_indices"_a,
           "pool_valid"_a, "raw_keys"_a, "raw_gates"_a, "raw_valid"_a,
           "raw_positions"_a, "compress_ape"_a,
           "previous_total_tokens"_a)
      .def_prop_ro("physical_pool_rows",
                   &NativeIndexPoolUpdateSelectionPlan::physical_pool_rows)
      .def_prop_ro("selected_width",
                   &NativeIndexPoolUpdateSelectionPlan::selected_width)
      .def_prop_ro("execution_count",
                   &NativeIndexPoolUpdateSelectionPlan::execution_count)
      .def_prop_ro("dynamic_allocation_count",
                   &NativeIndexPoolUpdateSelectionPlan::dynamic_allocation_count)
      .def_prop_ro("graph_node_count",
                   &NativeIndexPoolUpdateSelectionPlan::graph_node_count)
      .def_prop_ro("shape_discovery_count",
                   &NativeIndexPoolUpdateSelectionPlan::shape_discovery_count)
      .def_prop_ro("host_synchronization_count",
                   &NativeIndexPoolUpdateSelectionPlan::host_synchronization_count)
      .def_prop_ro("returned_intermediate_tensor_bytes",
                   &NativeIndexPoolUpdateSelectionPlan::returned_intermediate_tensor_bytes)
      .def_prop_ro("scratch_bytes",
                   &NativeIndexPoolUpdateSelectionPlan::scratch_bytes)
      .def_prop_ro("debug_pool_logits",
                   &NativeIndexPoolUpdateSelectionPlan::debug_pool_logits)
      .def_prop_ro("debug_pool_probabilities",
                   &NativeIndexPoolUpdateSelectionPlan::debug_pool_probabilities)
      .def_prop_ro("current_raw_keys",
                   &NativeIndexPoolUpdateSelectionPlan::current_raw_keys)
      .def_prop_ro("current_raw_gates",
                   &NativeIndexPoolUpdateSelectionPlan::current_raw_gates)
      .def_prop_ro("current_raw_valid",
                   &NativeIndexPoolUpdateSelectionPlan::current_raw_valid)
      .def_prop_ro("current_raw_positions",
                   &NativeIndexPoolUpdateSelectionPlan::current_raw_positions)
      .def_prop_ro("buffer_identities",
                   &NativeIndexPoolUpdateSelectionPlan::buffer_identities);

  nb::class_<NativeDSASparseAttentionPlan>(module,
                                           "NativeDSASparseAttentionPlan")
      .def(nb::init<int, int, float, float>(), "physical_pool_rows"_a,
           "physical_kv_rows"_a, "indexer_softmax_scale"_a, "attention_scale"_a)
      .def("execute", &NativeDSASparseAttentionPlan::execute, "index_query"_a,
           "mixture_weights"_a, "pool_keys"_a, "pool_indices"_a, "pool_valid"_a,
           "raw_positions"_a, "raw_valid"_a, "current_valid"_a,
           "attention_query"_a, "latent"_a, "logical_pool_rows"_a, "kv_len"_a,
           "active_tail_count"_a)
      .def("debug_prepare_inputs",
           &NativeDSASparseAttentionPlan::debug_prepare_inputs,
           "selected_indices"_a, "selected_valid"_a, "attention_query"_a,
           "latent"_a, "kv_len"_a)
      .def("debug_attention_math",
           &NativeDSASparseAttentionPlan::debug_attention_math,
           "scaled_query"_a, "gathered_latent"_a, "selected_valid"_a)
      .def_prop_ro("physical_pool_rows",
                   &NativeDSASparseAttentionPlan::physical_pool_rows)
      .def_prop_ro("physical_kv_rows",
                   &NativeDSASparseAttentionPlan::physical_kv_rows)
      .def_prop_ro("selected_width",
                   &NativeDSASparseAttentionPlan::selected_width)
      .def_prop_ro("execution_count",
                   &NativeDSASparseAttentionPlan::execution_count)
      .def_prop_ro("dynamic_allocation_count",
                   &NativeDSASparseAttentionPlan::dynamic_allocation_count)
      .def_prop_ro("graph_node_count",
                   &NativeDSASparseAttentionPlan::graph_node_count)
      .def_prop_ro("shape_discovery_count",
                   &NativeDSASparseAttentionPlan::shape_discovery_count)
      .def_prop_ro("host_synchronization_count",
                   &NativeDSASparseAttentionPlan::host_synchronization_count)
      .def_prop_ro(
          "returned_intermediate_tensor_bytes",
          &NativeDSASparseAttentionPlan::returned_intermediate_tensor_bytes)
      .def_prop_ro("scratch_bytes",
                   &NativeDSASparseAttentionPlan::scratch_bytes)
      .def_prop_ro("debug_selected_indices",
                   &NativeDSASparseAttentionPlan::debug_selected_indices)
      .def_prop_ro("debug_selected_valid",
                   &NativeDSASparseAttentionPlan::debug_selected_valid)
      .def_prop_ro("debug_gathered_latent",
                   &NativeDSASparseAttentionPlan::debug_gathered_latent)
      .def_prop_ro("debug_attention_scores",
                   &NativeDSASparseAttentionPlan::debug_attention_scores)
      .def_prop_ro("buffer_identities",
                   &NativeDSASparseAttentionPlan::buffer_identities);

  nb::class_<NativePackedMoEDecodePlan>(module,
                                        "NativePackedMoEDecodePlan")
      .def(nb::init<int, int, int, int, int>(), "hidden_size"_a,
           "intermediate_size"_a, "shared_intermediate_size"_a,
           "expert_count"_a, "swiglu_limit"_a)
      .def("execute", &NativePackedMoEDecodePlan::execute, "x"_a,
           "expert_ids"_a, "scores"_a, "gate_up_weight"_a,
           "gate_up_scale_inv"_a, "down_weight"_a,
           "down_scale_inv"_a, "shared_gate_weight"_a,
           "shared_gate_scale_inv"_a, "shared_up_weight"_a,
           "shared_up_scale_inv"_a, "shared_down_weight"_a,
           "shared_down_scale_inv"_a)
      .def_prop_ro("hidden_size", &NativePackedMoEDecodePlan::hidden_size)
      .def_prop_ro("intermediate_size",
                   &NativePackedMoEDecodePlan::intermediate_size)
      .def_prop_ro("shared_intermediate_size",
                   &NativePackedMoEDecodePlan::shared_intermediate_size)
      .def_prop_ro("expert_count", &NativePackedMoEDecodePlan::expert_count)
      .def_prop_ro("execution_count",
                   &NativePackedMoEDecodePlan::execution_count)
      .def_prop_ro("dynamic_allocation_count",
                   &NativePackedMoEDecodePlan::dynamic_allocation_count)
      .def_prop_ro("graph_node_count",
                   &NativePackedMoEDecodePlan::graph_node_count)
      .def_prop_ro("shape_discovery_count",
                   &NativePackedMoEDecodePlan::shape_discovery_count)
      .def_prop_ro("host_synchronization_count",
                   &NativePackedMoEDecodePlan::host_synchronization_count)
      .def_prop_ro("returned_intermediate_tensor_bytes",
                   &NativePackedMoEDecodePlan::returned_intermediate_tensor_bytes)
      .def_prop_ro("scratch_bytes", &NativePackedMoEDecodePlan::scratch_bytes)
      .def_prop_ro(
          "uses_shape_specialized_routed_gate_up",
          &NativePackedMoEDecodePlan::uses_shape_specialized_routed_gate_up)
      .def_prop_ro("debug_routed_hidden",
                   &NativePackedMoEDecodePlan::debug_routed_hidden)
      .def_prop_ro("debug_routed_down",
                   &NativePackedMoEDecodePlan::debug_routed_down)
      .def_prop_ro("debug_routed_output",
                   &NativePackedMoEDecodePlan::debug_routed_output)
      .def_prop_ro("debug_shared_hidden",
                   &NativePackedMoEDecodePlan::debug_shared_hidden)
      .def_prop_ro("debug_shared_down",
                   &NativePackedMoEDecodePlan::debug_shared_down)
      .def_prop_ro("buffer_identities",
                   &NativePackedMoEDecodePlan::buffer_identities);
}
