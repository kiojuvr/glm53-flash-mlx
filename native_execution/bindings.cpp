#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "native_dsa_attention_plan.h"
#include "native_dsa_score_plan.h"
#include "native_indexer_plan.h"

namespace nb = nanobind;
using namespace nb::literals;
using glm53::native_execution::NativeDSAScoreSelectionPlan;
using glm53::native_execution::NativeDSASparseAttentionPlan;
using glm53::native_execution::NativeIndexSelectionPlan;

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

  nb::class_<NativeDSASparseAttentionPlan>(module,
                                           "NativeDSASparseAttentionPlan")
      .def(nb::init<int, int, float, float>(), "physical_pool_rows"_a,
           "physical_kv_rows"_a, "indexer_softmax_scale"_a, "attention_scale"_a)
      .def("execute", &NativeDSASparseAttentionPlan::execute, "index_query"_a,
           "mixture_weights"_a, "pool_keys"_a, "pool_indices"_a, "pool_valid"_a,
           "raw_positions"_a, "raw_valid"_a, "current_valid"_a,
           "attention_query"_a, "latent"_a, "logical_pool_rows"_a, "kv_len"_a,
           "active_tail_count"_a)
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
}
