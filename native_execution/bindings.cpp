#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "native_dsa_attention_plan.h"
#include "native_dsa_score_plan.h"
#include "native_indexpool_update_plan.h"
#include "native_indexer_plan.h"
#include "native_packed_moe_plan.h"
#include "native_prefill_av_plan.h"
#include "native_prefill_layer_plan.h"
#include "native_prefill_moe_route_plan.h"
#include "native_prefill_moe_gate_up_plan.h"
#include "native_prefill_shared_expert_plan.h"
#include "native_prefill_moe_plan.h"
#include "native_selected_v_av_plan.h"
#include "native_selected_kv_attention_selection_plan.h"
#include "native_selected_union_plan.h"
#include "native_indirect_selected_latent_plan.h"
#include "native_projected_qk_union_tile_plan.h"
#include "native_projected_qk_union_tile_loop_plan.h"
#include "native_shared_physical_value_tile_plan.h"
#include "native_q256_dsa_prefill_plan.h"
#include "native_prefill_dsa_output_plan.h"

namespace nb = nanobind;
using namespace nb::literals;
using glm53::native_execution::NativeDSAScoreSelectionPlan;
using glm53::native_execution::NativeDSASparseAttentionPlan;
using glm53::native_execution::NativeIndexSelectionPlan;
using glm53::native_execution::NativeIndexPoolUpdateSelectionPlan;
using glm53::native_execution::NativePackedMoEDecodePlan;
using glm53::native_execution::NativePackedMoERoutedDiagnostic;
using glm53::native_execution::NativeSparsePrefillAVPlan;
using glm53::native_execution::NativeSelectedVProjectionAVPlan;
using glm53::native_execution::NativeSelectedKVAttentionSelectionPlan;
using glm53::native_execution::NativeSelectedUnionPlan;
using glm53::native_execution::NativeIndirectSelectedLatentPlan;
using glm53::native_execution::NativeProjectedQKUnionTilePlan;
using glm53::native_execution::NativeProjectedQKUnionTileLoopPlan;
using glm53::native_execution::NativeSharedPhysicalValueTilePlan;
using glm53::native_execution::NativeQ256DSAPrefillPlan;
using glm53::native_execution::NativePrefillDSAOutputPlan;
using glm53::native_execution::NativePrefillLayerSubstrate;
using glm53::native_execution::NativePrefillMoERoutePlan;
using glm53::native_execution::NativePrefillMoEGateUpPlan;
using glm53::native_execution::NativePrefillSharedExpertPlan;
using glm53::native_execution::NativePrefillMoEPlan;
using glm53::native_execution::NativeRoutedSigmoidFormulaSweep;

NB_MODULE(_ext, module) {
  module.doc() = "Probe-only persistent native execution bridge for GLM-5.3";
  nb::class_<NativePrefillDSAOutputPlan>(module,
                                         "NativePrefillDSAOutputPlan")
      .def(nb::init<>())
      .def("execute", &NativePrefillDSAOutputPlan::execute,
           "head_major"_a, "weight"_a, "scale_inv"_a)
      .def_prop_ro("query_rows", &NativePrefillDSAOutputPlan::query_rows)
      .def_prop_ro("heads", &NativePrefillDSAOutputPlan::heads)
      .def_prop_ro("value_dim", &NativePrefillDSAOutputPlan::value_dim)
      .def_prop_ro("hidden_size", &NativePrefillDSAOutputPlan::hidden_size)
      .def_prop_ro("execution_count", &NativePrefillDSAOutputPlan::execution_count)
      .def_prop_ro("dynamic_allocation_count",
                   &NativePrefillDSAOutputPlan::dynamic_allocation_count)
      .def_prop_ro("graph_node_count",
                   &NativePrefillDSAOutputPlan::graph_node_count)
      .def_prop_ro("shape_discovery_count",
                   &NativePrefillDSAOutputPlan::shape_discovery_count)
      .def_prop_ro("host_synchronization_count",
                   &NativePrefillDSAOutputPlan::host_synchronization_count)
      .def_prop_ro("returned_intermediate_tensor_bytes",
                   &NativePrefillDSAOutputPlan::returned_intermediate_tensor_bytes)
      .def_prop_ro("scratch_bytes", &NativePrefillDSAOutputPlan::scratch_bytes)
      .def_prop_ro("buffer_identities",
                   &NativePrefillDSAOutputPlan::buffer_identities)
      .def_prop_ro("debug_row_major",
                   &NativePrefillDSAOutputPlan::debug_row_major);
  nb::class_<NativePrefillLayerSubstrate>(module,
                                          "NativePrefillLayerSubstrate")
      .def(nb::init<>())
      .def("execute", &NativePrefillLayerSubstrate::execute, "hidden"_a)
      .def_prop_ro("query_rows", &NativePrefillLayerSubstrate::query_rows)
      .def_prop_ro("query_block_rows",
                   &NativePrefillLayerSubstrate::query_block_rows)
      .def_prop_ro("query_block_count",
                   &NativePrefillLayerSubstrate::query_block_count)
      .def_prop_ro("hidden_size", &NativePrefillLayerSubstrate::hidden_size)
      .def_prop_ro("physical_pool_rows",
                   &NativePrefillLayerSubstrate::physical_pool_rows)
      .def_prop_ro("selected_width",
                   &NativePrefillLayerSubstrate::selected_width)
      .def_prop_ro("route_rows", &NativePrefillLayerSubstrate::route_rows)
      .def_prop_ro("execution_count",
                   &NativePrefillLayerSubstrate::execution_count)
      .def_prop_ro("dynamic_allocation_count",
                   &NativePrefillLayerSubstrate::dynamic_allocation_count)
      .def_prop_ro("graph_node_count",
                   &NativePrefillLayerSubstrate::graph_node_count)
      .def_prop_ro("shape_discovery_count",
                   &NativePrefillLayerSubstrate::shape_discovery_count)
      .def_prop_ro("host_synchronization_count",
                   &NativePrefillLayerSubstrate::host_synchronization_count)
      .def_prop_ro("native_encoder_scopes_per_execute",
                   &NativePrefillLayerSubstrate::native_encoder_scopes_per_execute)
      .def_prop_ro("startup_pipeline_lookup_count",
                   &NativePrefillLayerSubstrate::startup_pipeline_lookup_count)
      .def_prop_ro("pipeline_lookup_count_per_execute",
                   &NativePrefillLayerSubstrate::pipeline_lookup_count_per_execute)
      .def_prop_ro("returned_intermediate_tensor_bytes",
                   &NativePrefillLayerSubstrate::returned_intermediate_tensor_bytes)
      .def_prop_ro("dsa_score_scratch_bytes",
                   &NativePrefillLayerSubstrate::dsa_score_scratch_bytes)
      .def_prop_ro("scratch_bytes",
                   &NativePrefillLayerSubstrate::scratch_bytes)
      .def_prop_ro("arena_bytes", &NativePrefillLayerSubstrate::arena_bytes)
      .def_prop_ro("output", &NativePrefillLayerSubstrate::output)
      .def_prop_ro("buffer_identities_stable",
                   &NativePrefillLayerSubstrate::buffer_identities_stable)
      .def_prop_ro("buffer_identities",
                   &NativePrefillLayerSubstrate::buffer_identities)
      .def_prop_ro("fixed_topology",
                   &NativePrefillLayerSubstrate::fixed_topology);

  nb::class_<NativePrefillMoERoutePlan>(module,
                                        "NativePrefillMoERoutePlan")
      .def(nb::init<int>(), "expert_count"_a = 288)
      .def("execute", &NativePrefillMoERoutePlan::execute,
           "expert_ids"_a, "scores"_a)
      .def_prop_ro("query_rows", &NativePrefillMoERoutePlan::query_rows)
      .def_prop_ro("top_k", &NativePrefillMoERoutePlan::top_k)
      .def_prop_ro("route_rows", &NativePrefillMoERoutePlan::route_rows)
      .def_prop_ro("expert_count", &NativePrefillMoERoutePlan::expert_count)
      .def_prop_ro("tile_rows", &NativePrefillMoERoutePlan::tile_rows)
      .def_prop_ro("descriptor_capacity",
                   &NativePrefillMoERoutePlan::descriptor_capacity)
      .def_prop_ro("execution_count",
                   &NativePrefillMoERoutePlan::execution_count)
      .def_prop_ro("dynamic_allocation_count",
                   &NativePrefillMoERoutePlan::dynamic_allocation_count)
      .def_prop_ro("graph_node_count",
                   &NativePrefillMoERoutePlan::graph_node_count)
      .def_prop_ro("shape_discovery_count",
                   &NativePrefillMoERoutePlan::shape_discovery_count)
      .def_prop_ro("host_synchronization_count",
                   &NativePrefillMoERoutePlan::host_synchronization_count)
      .def_prop_ro("materialized_sorted_hidden_bytes",
                   &NativePrefillMoERoutePlan::materialized_sorted_hidden_bytes)
      .def_prop_ro("scratch_bytes", &NativePrefillMoERoutePlan::scratch_bytes)
      .def_prop_ro("buffer_identities",
                   &NativePrefillMoERoutePlan::buffer_identities)
      .def_prop_ro("sorted_route_order",
                   &NativePrefillMoERoutePlan::sorted_route_order)
      .def_prop_ro("inverse_route_order",
                   &NativePrefillMoERoutePlan::inverse_route_order)
      .def_prop_ro("sorted_experts",
                   &NativePrefillMoERoutePlan::sorted_experts)
      .def_prop_ro("sorted_scores", &NativePrefillMoERoutePlan::sorted_scores)
      .def_prop_ro("expert_offsets",
                   &NativePrefillMoERoutePlan::expert_offsets)
      .def_prop_ro("tile_experts", &NativePrefillMoERoutePlan::tile_experts)
      .def_prop_ro("tile_starts", &NativePrefillMoERoutePlan::tile_starts)
      .def_prop_ro("tile_lengths", &NativePrefillMoERoutePlan::tile_lengths)
      .def_prop_ro("descriptor_count",
                   &NativePrefillMoERoutePlan::descriptor_count)
      .def_prop_ro("invalid_route_count",
                   &NativePrefillMoERoutePlan::invalid_route_count);

  nb::class_<NativePrefillMoEGateUpPlan>(module,
                                         "NativePrefillMoEGateUpPlan")
      .def(nb::init<int>(), "expert_count"_a = 288)
      .def("execute", &NativePrefillMoEGateUpPlan::execute,
           "hidden"_a, "expert_ids"_a, "scores"_a,
           "gate_up_weight"_a, "gate_up_scale_inv"_a)
      .def("execute_routed", &NativePrefillMoEGateUpPlan::execute_routed,
           "hidden"_a, "expert_ids"_a, "scores"_a,
           "gate_up_weight"_a, "gate_up_scale_inv"_a,
           "down_weight"_a, "down_scale_inv"_a)
      .def("execute_routed_fused",
           &NativePrefillMoEGateUpPlan::execute_routed_fused,
           "hidden"_a, "expert_ids"_a, "scores"_a,
           "gate_up_weight"_a, "gate_up_scale_inv"_a,
           "down_weight"_a, "down_scale_inv"_a)
      .def_prop_ro("query_rows", &NativePrefillMoEGateUpPlan::query_rows)
      .def_prop_ro("route_rows", &NativePrefillMoEGateUpPlan::route_rows)
      .def_prop_ro("hidden_size", &NativePrefillMoEGateUpPlan::hidden_size)
      .def_prop_ro("intermediate_size",
                   &NativePrefillMoEGateUpPlan::intermediate_size)
      .def_prop_ro("expert_count", &NativePrefillMoEGateUpPlan::expert_count)
      .def_prop_ro("execution_count",
                   &NativePrefillMoEGateUpPlan::execution_count)
      .def_prop_ro("dynamic_allocation_count",
                   &NativePrefillMoEGateUpPlan::dynamic_allocation_count)
      .def_prop_ro("graph_node_count",
                   &NativePrefillMoEGateUpPlan::graph_node_count)
      .def_prop_ro("shape_discovery_count",
                   &NativePrefillMoEGateUpPlan::shape_discovery_count)
      .def_prop_ro("host_synchronization_count",
                   &NativePrefillMoEGateUpPlan::host_synchronization_count)
      .def_prop_ro("returned_route_metadata_bytes",
                   &NativePrefillMoEGateUpPlan::returned_route_metadata_bytes)
      .def_prop_ro("materialized_sorted_hidden_bytes",
                   &NativePrefillMoEGateUpPlan::materialized_sorted_hidden_bytes)
      .def_prop_ro("fused_materialized_routed_down_bytes",
                   &NativePrefillMoEGateUpPlan::fused_materialized_routed_down_bytes)
      .def_prop_ro("scratch_bytes", &NativePrefillMoEGateUpPlan::scratch_bytes)
      .def_prop_ro("buffer_identities",
                   &NativePrefillMoEGateUpPlan::buffer_identities)
      .def_prop_ro("debug_route_order",
                   &NativePrefillMoEGateUpPlan::debug_route_order)
      .def_prop_ro("debug_sorted_experts",
                   &NativePrefillMoEGateUpPlan::debug_sorted_experts);

  nb::class_<NativePrefillSharedExpertPlan>(module,
                                            "NativePrefillSharedExpertPlan")
      .def(nb::init<>())
      .def("execute", &NativePrefillSharedExpertPlan::execute,
           "hidden"_a, "gate_weight"_a, "gate_scale_inv"_a,
           "up_weight"_a, "up_scale_inv"_a, "down_weight"_a,
           "down_scale_inv"_a)
      .def_prop_ro("query_rows", &NativePrefillSharedExpertPlan::query_rows)
      .def_prop_ro("hidden_size", &NativePrefillSharedExpertPlan::hidden_size)
      .def_prop_ro("intermediate_size",
                   &NativePrefillSharedExpertPlan::intermediate_size)
      .def_prop_ro("tile_rows", &NativePrefillSharedExpertPlan::tile_rows)
      .def_prop_ro("execution_count",
                   &NativePrefillSharedExpertPlan::execution_count)
      .def_prop_ro("dynamic_allocation_count",
                   &NativePrefillSharedExpertPlan::dynamic_allocation_count)
      .def_prop_ro("graph_node_count",
                   &NativePrefillSharedExpertPlan::graph_node_count)
      .def_prop_ro("shape_discovery_count",
                   &NativePrefillSharedExpertPlan::shape_discovery_count)
      .def_prop_ro("host_synchronization_count",
                   &NativePrefillSharedExpertPlan::host_synchronization_count)
      .def_prop_ro("returned_intermediate_tensor_bytes",
                   &NativePrefillSharedExpertPlan::returned_intermediate_tensor_bytes)
      .def_prop_ro("scratch_bytes",
                   &NativePrefillSharedExpertPlan::scratch_bytes)
      .def_prop_ro("buffer_identities",
                   &NativePrefillSharedExpertPlan::buffer_identities);

  nb::class_<NativePrefillMoEPlan>(module, "NativePrefillMoEPlan")
      .def(nb::init<int>(), "expert_count"_a = 288)
      .def("execute", &NativePrefillMoEPlan::execute,
           "hidden"_a, "expert_ids"_a, "scores"_a,
           "gate_up_weight"_a, "gate_up_scale_inv"_a,
           "down_weight"_a, "down_scale_inv"_a,
           "shared_gate_weight"_a, "shared_gate_scale_inv"_a,
           "shared_up_weight"_a, "shared_up_scale_inv"_a,
           "shared_down_weight"_a, "shared_down_scale_inv"_a)
      .def("execute_fused_down_reduce",
           &NativePrefillMoEPlan::execute_fused_down_reduce,
           "hidden"_a, "expert_ids"_a, "scores"_a,
           "gate_up_weight"_a, "gate_up_scale_inv"_a,
           "down_weight"_a, "down_scale_inv"_a,
           "shared_gate_weight"_a, "shared_gate_scale_inv"_a,
           "shared_up_weight"_a, "shared_up_scale_inv"_a,
           "shared_down_weight"_a, "shared_down_scale_inv"_a)
      .def_prop_ro("query_rows", &NativePrefillMoEPlan::query_rows)
      .def_prop_ro("hidden_size", &NativePrefillMoEPlan::hidden_size)
      .def_prop_ro("execution_count", &NativePrefillMoEPlan::execution_count)
      .def_prop_ro("dynamic_allocation_count",
                   &NativePrefillMoEPlan::dynamic_allocation_count)
      .def_prop_ro("graph_node_count", &NativePrefillMoEPlan::graph_node_count)
      .def_prop_ro("shape_discovery_count",
                   &NativePrefillMoEPlan::shape_discovery_count)
      .def_prop_ro("host_synchronization_count",
                   &NativePrefillMoEPlan::host_synchronization_count)
      .def_prop_ro("returned_intermediate_tensor_bytes",
                   &NativePrefillMoEPlan::returned_intermediate_tensor_bytes)
      .def_prop_ro("fused_materialized_routed_down_bytes",
                   &NativePrefillMoEPlan::fused_materialized_routed_down_bytes)
      .def_prop_ro("scratch_bytes", &NativePrefillMoEPlan::scratch_bytes)
      .def_prop_ro("buffer_identities",
                   &NativePrefillMoEPlan::buffer_identities);

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
      .def("bind_weights", &NativePackedMoEDecodePlan::bind_weights,
           "gate_up_weight"_a, "gate_up_scale_inv"_a, "down_weight"_a,
           "down_scale_inv"_a, "shared_gate_weight"_a,
           "shared_gate_scale_inv"_a, "shared_up_weight"_a,
           "shared_up_scale_inv"_a, "shared_down_weight"_a,
           "shared_down_scale_inv"_a)
      .def("execute_bound", &NativePackedMoEDecodePlan::execute_bound, "x"_a,
           "expert_ids"_a, "scores"_a)
      .def("execute_lazy", &NativePackedMoEDecodePlan::execute_lazy, "x"_a,
           "expert_ids"_a, "scores"_a)
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
      .def_prop_ro("static_input_validation_count",
                   &NativePackedMoEDecodePlan::static_input_validation_count)
      .def_prop_ro("dynamic_input_validation_count",
                   &NativePackedMoEDecodePlan::dynamic_input_validation_count)
      .def_prop_ro("pipeline_lookup_count",
                   &NativePackedMoEDecodePlan::pipeline_lookup_count)
      .def_prop_ro("lazy_graph_count",
                   &NativePackedMoEDecodePlan::lazy_graph_count)
      .def_prop_ro("weights_bound", &NativePackedMoEDecodePlan::weights_bound)
      .def_prop_ro("bound_weight_identities_stable",
                   &NativePackedMoEDecodePlan::bound_weight_identities_stable)
      .def_prop_ro("scratch_bytes", &NativePackedMoEDecodePlan::scratch_bytes)
      .def_prop_ro(
          "uses_shape_specialized_routed_gate_up",
          &NativePackedMoEDecodePlan::uses_shape_specialized_routed_gate_up)
      .def_prop_ro("uses_fast_bf16_routed_sigmoid",
                   &NativePackedMoEDecodePlan::uses_fast_bf16_routed_sigmoid)
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
                   &NativePackedMoEDecodePlan::buffer_identities)
      .def_prop_ro("bound_weight_identities",
                   &NativePackedMoEDecodePlan::bound_weight_identities);

  nb::class_<NativePackedMoERoutedDiagnostic>(
      module, "NativePackedMoERoutedDiagnostic")
      .def(nb::init<int>(), "expert_count"_a)
      .def("execute", &NativePackedMoERoutedDiagnostic::execute, "x"_a,
           "expert_ids"_a, "gate_up_weight"_a, "gate_up_scale_inv"_a)
      .def_prop_ro("gate", &NativePackedMoERoutedDiagnostic::gate)
      .def_prop_ro("up", &NativePackedMoERoutedDiagnostic::up)
      .def_prop_ro("sigmoid", &NativePackedMoERoutedDiagnostic::sigmoid)
      .def_prop_ro("silu", &NativePackedMoERoutedDiagnostic::silu)
      .def_prop_ro("hidden", &NativePackedMoERoutedDiagnostic::hidden)
      .def_prop_ro("execution_count",
                   &NativePackedMoERoutedDiagnostic::execution_count)
      .def_prop_ro("scratch_bytes",
                   &NativePackedMoERoutedDiagnostic::scratch_bytes)
      .def_prop_ro("buffer_identities",
                   &NativePackedMoERoutedDiagnostic::buffer_identities);

  nb::class_<NativeRoutedSigmoidFormulaSweep>(
      module, "NativeRoutedSigmoidFormulaSweep")
      .def(nb::init<int>(), "elements"_a)
      .def("execute", &NativeRoutedSigmoidFormulaSweep::execute, "gate"_a)
      .def_prop_ro("standard_bf16",
                   &NativeRoutedSigmoidFormulaSweep::standard_bf16)
      .def_prop_ro("precise_bf16",
                   &NativeRoutedSigmoidFormulaSweep::precise_bf16)
      .def_prop_ro("standard_f32",
                   &NativeRoutedSigmoidFormulaSweep::standard_f32)
      .def_prop_ro("precise_f32",
                   &NativeRoutedSigmoidFormulaSweep::precise_f32)
      .def_prop_ro("fast_bf16",
                   &NativeRoutedSigmoidFormulaSweep::fast_bf16)
      .def_prop_ro("fast_f32",
                   &NativeRoutedSigmoidFormulaSweep::fast_f32)
      .def_prop_ro("elements", &NativeRoutedSigmoidFormulaSweep::elements);

  nb::class_<NativeSparsePrefillAVPlan>(module,
                                        "NativeSparsePrefillAVPlan")
      .def(nb::init<int>(), "physical_k"_a)
      .def("execute", &NativeSparsePrefillAVPlan::execute,
           "selected_probabilities"_a, "selected_values"_a,
           "selected_indices"_a, "selected_valid"_a)
      .def("execute_prepared", &NativeSparsePrefillAVPlan::execute_prepared,
           "selected_probabilities"_a, "selected_values"_a)
      .def_prop_ro("physical_k", &NativeSparsePrefillAVPlan::physical_k)
      .def_prop_ro("packed_k", &NativeSparsePrefillAVPlan::packed_k)
      .def_prop_ro("selected_width",
                   &NativeSparsePrefillAVPlan::selected_width)
      .def_prop_ro("bk", &NativeSparsePrefillAVPlan::bk)
      .def_prop_ro("head_tile", &NativeSparsePrefillAVPlan::head_tile)
      .def_prop_ro("execution_count",
                   &NativeSparsePrefillAVPlan::execution_count)
      .def_prop_ro("dynamic_allocation_count",
                   &NativeSparsePrefillAVPlan::dynamic_allocation_count)
      .def_prop_ro("graph_node_count",
                   &NativeSparsePrefillAVPlan::graph_node_count)
      .def_prop_ro("shape_discovery_count",
                   &NativeSparsePrefillAVPlan::shape_discovery_count)
      .def_prop_ro("host_synchronization_count",
                   &NativeSparsePrefillAVPlan::host_synchronization_count)
      .def_prop_ro("returned_intermediate_tensor_bytes",
                   &NativeSparsePrefillAVPlan::returned_intermediate_tensor_bytes)
      .def_prop_ro("scratch_bytes", &NativeSparsePrefillAVPlan::scratch_bytes)
      .def_prop_ro("buffer_identities",
                   &NativeSparsePrefillAVPlan::buffer_identities)
      .def_prop_ro("debug_lane_to_selected",
                   &NativeSparsePrefillAVPlan::debug_lane_to_selected);

  nb::class_<NativeSelectedVProjectionAVPlan>(
      module, "NativeSelectedVProjectionAVPlan")
      .def(nb::init<int, bool>(), "physical_k"_a,
           "attention_enabled"_a = false)
      .def("execute", &NativeSelectedVProjectionAVPlan::execute,
           "selected_probabilities"_a, "selected_latent"_a,
           "value_weight"_a, "selected_indices"_a, "selected_valid"_a)
      .def("execute_attention",
           &NativeSelectedVProjectionAVPlan::execute_attention,
           "selected_latent"_a, "key_weight"_a, "value_weight"_a,
           "attention_query"_a, "selected_indices"_a,
           "selected_valid"_a, "attention_scale"_a)
      .def_prop_ro("physical_k", &NativeSelectedVProjectionAVPlan::physical_k)
      .def_prop_ro("attention_enabled",
                   &NativeSelectedVProjectionAVPlan::attention_enabled)
      .def_prop_ro("packed_k", &NativeSelectedVProjectionAVPlan::packed_k)
      .def_prop_ro("query_rows", &NativeSelectedVProjectionAVPlan::query_rows)
      .def_prop_ro("selected_width",
                   &NativeSelectedVProjectionAVPlan::selected_width)
      .def_prop_ro("execution_count",
                   &NativeSelectedVProjectionAVPlan::execution_count)
      .def_prop_ro("dynamic_allocation_count",
                   &NativeSelectedVProjectionAVPlan::dynamic_allocation_count)
      .def_prop_ro("graph_node_count",
                   &NativeSelectedVProjectionAVPlan::graph_node_count)
      .def_prop_ro("shape_discovery_count",
                   &NativeSelectedVProjectionAVPlan::shape_discovery_count)
      .def_prop_ro("host_synchronization_count",
                   &NativeSelectedVProjectionAVPlan::host_synchronization_count)
      .def_prop_ro("returned_intermediate_tensor_bytes",
                   &NativeSelectedVProjectionAVPlan::returned_intermediate_tensor_bytes)
      .def_prop_ro("scratch_bytes", &NativeSelectedVProjectionAVPlan::scratch_bytes)
      .def_prop_ro("buffer_identities",
                   &NativeSelectedVProjectionAVPlan::buffer_identities)
      .def_prop_ro("debug_projected_value",
                   &NativeSelectedVProjectionAVPlan::debug_projected_value)
      .def_prop_ro("debug_projected_key",
                   &NativeSelectedVProjectionAVPlan::debug_projected_key)
      .def_prop_ro("debug_scaled_query",
                   &NativeSelectedVProjectionAVPlan::debug_scaled_query)
      .def_prop_ro("debug_attention_scores",
                   &NativeSelectedVProjectionAVPlan::debug_attention_scores)
      .def_prop_ro(
          "debug_attention_probabilities",
          &NativeSelectedVProjectionAVPlan::debug_attention_probabilities)
      .def_prop_ro("debug_lane_to_selected",
                   &NativeSelectedVProjectionAVPlan::debug_lane_to_selected);

  nb::class_<NativeSelectedKVAttentionSelectionPlan>(
      module, "NativeSelectedKVAttentionSelectionPlan")
      .def(nb::init<int, int, float, float>(), "physical_pool_rows"_a,
           "physical_kv_rows"_a, "indexer_softmax_scale"_a,
           "attention_scale"_a)
      .def("execute", &NativeSelectedKVAttentionSelectionPlan::execute,
           "index_query"_a, "mixture_weights"_a, "pool_keys"_a,
           "pool_indices"_a, "pool_valid"_a, "raw_positions"_a,
           "raw_valid"_a, "current_valid"_a, "latent"_a,
           "key_weight"_a, "value_weight"_a, "attention_query"_a,
           "logical_pool_rows"_a, "kv_len"_a, "active_tail_count"_a)
      .def_prop_ro("physical_pool_rows",
                   &NativeSelectedKVAttentionSelectionPlan::physical_pool_rows)
      .def_prop_ro("physical_kv_rows",
                   &NativeSelectedKVAttentionSelectionPlan::physical_kv_rows)
      .def_prop_ro("query_rows",
                   &NativeSelectedKVAttentionSelectionPlan::query_rows)
      .def_prop_ro("selected_width",
                   &NativeSelectedKVAttentionSelectionPlan::selected_width)
      .def_prop_ro("execution_count",
                   &NativeSelectedKVAttentionSelectionPlan::execution_count)
      .def_prop_ro("dynamic_allocation_count",
                   &NativeSelectedKVAttentionSelectionPlan::dynamic_allocation_count)
      .def_prop_ro("graph_node_count",
                   &NativeSelectedKVAttentionSelectionPlan::graph_node_count)
      .def_prop_ro("shape_discovery_count",
                   &NativeSelectedKVAttentionSelectionPlan::shape_discovery_count)
      .def_prop_ro("host_synchronization_count",
                   &NativeSelectedKVAttentionSelectionPlan::host_synchronization_count)
      .def_prop_ro("returned_intermediate_tensor_bytes",
                   &NativeSelectedKVAttentionSelectionPlan::returned_intermediate_tensor_bytes)
      .def_prop_ro("scratch_bytes",
                   &NativeSelectedKVAttentionSelectionPlan::scratch_bytes)
      .def_prop_ro("buffer_identities",
                   &NativeSelectedKVAttentionSelectionPlan::buffer_identities)
      .def_prop_ro("debug_selected_indices",
                   &NativeSelectedKVAttentionSelectionPlan::debug_selected_indices)
      .def_prop_ro("debug_selected_valid",
                   &NativeSelectedKVAttentionSelectionPlan::debug_selected_valid)
      .def_prop_ro("debug_score_order_indices",
                   &NativeSelectedKVAttentionSelectionPlan::debug_score_order_indices)
      .def_prop_ro("debug_selected_latent",
                   &NativeSelectedKVAttentionSelectionPlan::debug_selected_latent);

  nb::class_<NativeSelectedUnionPlan>(module, "NativeSelectedUnionPlan")
      .def(nb::init<int>(), "physical_k"_a)
      .def("execute", &NativeSelectedUnionPlan::execute,
           "selected_indices"_a, "selected_valid"_a)
      .def_prop_ro("physical_k", &NativeSelectedUnionPlan::physical_k)
      .def_prop_ro("query_rows", &NativeSelectedUnionPlan::query_rows)
      .def_prop_ro("selected_width", &NativeSelectedUnionPlan::selected_width)
      .def_prop_ro("execution_count", &NativeSelectedUnionPlan::execution_count)
      .def_prop_ro("dynamic_allocation_count",
                   &NativeSelectedUnionPlan::dynamic_allocation_count)
      .def_prop_ro("graph_node_count", &NativeSelectedUnionPlan::graph_node_count)
      .def_prop_ro("shape_discovery_count",
                   &NativeSelectedUnionPlan::shape_discovery_count)
      .def_prop_ro("host_synchronization_count",
                   &NativeSelectedUnionPlan::host_synchronization_count)
      .def_prop_ro("scratch_bytes", &NativeSelectedUnionPlan::scratch_bytes)
      .def_prop_ro("buffer_identities",
                   &NativeSelectedUnionPlan::buffer_identities)
      .def_prop_ro("union_indices", &NativeSelectedUnionPlan::union_indices)
      .def_prop_ro("union_count", &NativeSelectedUnionPlan::union_count)
      .def_prop_ro("query_union_slots",
                   &NativeSelectedUnionPlan::query_union_slots)
      .def_prop_ro("debug_membership_words",
                   &NativeSelectedUnionPlan::debug_membership_words)
      .def_prop_ro("debug_block_counts",
                   &NativeSelectedUnionPlan::debug_block_counts)
      .def_prop_ro("debug_block_prefix",
                   &NativeSelectedUnionPlan::debug_block_prefix);

  nb::class_<NativeIndirectSelectedLatentPlan>(
      module, "NativeIndirectSelectedLatentPlan")
      .def(nb::init<int>(), "physical_k"_a)
      .def("execute", &NativeIndirectSelectedLatentPlan::execute,
           "selected_indices"_a, "selected_valid"_a, "latent"_a)
      .def_prop_ro("physical_k",
                   &NativeIndirectSelectedLatentPlan::physical_k)
      .def_prop_ro("execution_count",
                   &NativeIndirectSelectedLatentPlan::execution_count)
      .def_prop_ro("dynamic_allocation_count",
                   &NativeIndirectSelectedLatentPlan::dynamic_allocation_count)
      .def_prop_ro("graph_node_count",
                   &NativeIndirectSelectedLatentPlan::graph_node_count)
      .def_prop_ro("shape_discovery_count",
                   &NativeIndirectSelectedLatentPlan::shape_discovery_count)
      .def_prop_ro("host_synchronization_count",
                   &NativeIndirectSelectedLatentPlan::host_synchronization_count)
      .def_prop_ro("returned_intermediate_tensor_bytes",
                   &NativeIndirectSelectedLatentPlan::returned_intermediate_tensor_bytes)
      .def_prop_ro("scratch_bytes",
                   &NativeIndirectSelectedLatentPlan::scratch_bytes)
      .def_prop_ro("buffer_identities",
                   &NativeIndirectSelectedLatentPlan::buffer_identities)
      .def_prop_ro("union_count",
                   &NativeIndirectSelectedLatentPlan::union_count)
      .def_prop_ro("union_indices",
                   &NativeIndirectSelectedLatentPlan::union_indices)
      .def_prop_ro("union_latent",
                   &NativeIndirectSelectedLatentPlan::union_latent)
      .def_prop_ro("debug_indirect_arguments",
                   &NativeIndirectSelectedLatentPlan::debug_indirect_arguments);

  nb::class_<NativeProjectedQKUnionTilePlan>(
      module, "NativeProjectedQKUnionTilePlan")
      .def(nb::init<>())
      .def("execute", &NativeProjectedQKUnionTilePlan::execute,
           "union_latent"_a, "key_weight"_a, "attention_query"_a,
           "query_union_slots"_a, "selected_valid"_a,
           "attention_scale"_a)
      .def_prop_ro("union_tile_rows",
                   &NativeProjectedQKUnionTilePlan::union_tile_rows)
      .def_prop_ro("query_rows",
                   &NativeProjectedQKUnionTilePlan::query_rows)
      .def_prop_ro("selected_width",
                   &NativeProjectedQKUnionTilePlan::selected_width)
      .def_prop_ro("execution_count",
                   &NativeProjectedQKUnionTilePlan::execution_count)
      .def_prop_ro("dynamic_allocation_count",
                   &NativeProjectedQKUnionTilePlan::dynamic_allocation_count)
      .def_prop_ro("graph_node_count",
                   &NativeProjectedQKUnionTilePlan::graph_node_count)
      .def_prop_ro("shape_discovery_count",
                   &NativeProjectedQKUnionTilePlan::shape_discovery_count)
      .def_prop_ro("host_synchronization_count",
                   &NativeProjectedQKUnionTilePlan::host_synchronization_count)
      .def_prop_ro("returned_intermediate_tensor_bytes",
                   &NativeProjectedQKUnionTilePlan::returned_intermediate_tensor_bytes)
      .def_prop_ro("scratch_bytes",
                   &NativeProjectedQKUnionTilePlan::scratch_bytes)
      .def_prop_ro("buffer_identities",
                   &NativeProjectedQKUnionTilePlan::buffer_identities)
      .def_prop_ro("debug_projected_union_key",
                   &NativeProjectedQKUnionTilePlan::debug_projected_union_key)
      .def_prop_ro("debug_scaled_query",
                   &NativeProjectedQKUnionTilePlan::debug_scaled_query);

  nb::class_<NativeProjectedQKUnionTileLoopPlan>(
      module, "NativeProjectedQKUnionTileLoopPlan")
      .def(nb::init<int, int, int, bool, bool>(), "physical_k"_a,
           "attention_query_rows"_a = 4, "tile_rows"_a = 4096,
           "softmax_enabled"_a = false, "value_enabled"_a = false)
      .def("execute", &NativeProjectedQKUnionTileLoopPlan::execute,
           "selected_indices"_a, "selected_valid"_a, "latent"_a,
           "key_weight"_a, "attention_query"_a, "attention_scale"_a)
      .def("execute_probabilities",
           &NativeProjectedQKUnionTileLoopPlan::execute_probabilities,
           "selected_indices"_a, "selected_valid"_a, "latent"_a,
           "key_weight"_a, "attention_query"_a, "attention_scale"_a)
      .def("execute_attention",
           &NativeProjectedQKUnionTileLoopPlan::execute_attention,
           "selected_indices"_a, "selected_valid"_a, "latent"_a,
           "key_weight"_a, "value_weight"_a, "attention_query"_a,
           "attention_scale"_a)
      .def_prop_ro("physical_k",
                   &NativeProjectedQKUnionTileLoopPlan::physical_k)
      .def_prop_ro("tile_rows",
                   &NativeProjectedQKUnionTileLoopPlan::tile_rows)
      .def_prop_ro("tile_count",
                   &NativeProjectedQKUnionTileLoopPlan::tile_count)
      .def_prop_ro("query_rows",
                   &NativeProjectedQKUnionTileLoopPlan::query_rows)
      .def_prop_ro("selected_width",
                   &NativeProjectedQKUnionTileLoopPlan::selected_width)
      .def_prop_ro("softmax_enabled",
                   &NativeProjectedQKUnionTileLoopPlan::softmax_enabled)
      .def_prop_ro("value_enabled",
                   &NativeProjectedQKUnionTileLoopPlan::value_enabled)
      .def_prop_ro("execution_count",
                   &NativeProjectedQKUnionTileLoopPlan::execution_count)
      .def_prop_ro("dynamic_allocation_count",
                   &NativeProjectedQKUnionTileLoopPlan::dynamic_allocation_count)
      .def_prop_ro("graph_node_count",
                   &NativeProjectedQKUnionTileLoopPlan::graph_node_count)
      .def_prop_ro("shape_discovery_count",
                   &NativeProjectedQKUnionTileLoopPlan::shape_discovery_count)
      .def_prop_ro("host_synchronization_count",
                   &NativeProjectedQKUnionTileLoopPlan::host_synchronization_count)
      .def_prop_ro("returned_intermediate_tensor_bytes",
                   &NativeProjectedQKUnionTileLoopPlan::returned_intermediate_tensor_bytes)
      .def_prop_ro("materialized_selected_key_bytes",
                   &NativeProjectedQKUnionTileLoopPlan::materialized_selected_key_bytes)
      .def_prop_ro("scratch_bytes",
                   &NativeProjectedQKUnionTileLoopPlan::scratch_bytes)
      .def_prop_ro("buffer_identities",
                   &NativeProjectedQKUnionTileLoopPlan::buffer_identities)
      .def_prop_ro("union_indices",
                   &NativeProjectedQKUnionTileLoopPlan::union_indices)
      .def_prop_ro("union_count",
                   &NativeProjectedQKUnionTileLoopPlan::union_count)
      .def_prop_ro("query_union_slots",
                   &NativeProjectedQKUnionTileLoopPlan::query_union_slots)
      .def_prop_ro("debug_union_latent_tile",
                   &NativeProjectedQKUnionTileLoopPlan::debug_union_latent_tile)
      .def_prop_ro("debug_projected_union_key_tile",
                   &NativeProjectedQKUnionTileLoopPlan::debug_projected_union_key_tile)
      .def_prop_ro("debug_scaled_queries",
                   &NativeProjectedQKUnionTileLoopPlan::debug_scaled_queries)
      .def_prop_ro("debug_attention_scores",
                   &NativeProjectedQKUnionTileLoopPlan::debug_attention_scores)
      .def_prop_ro("debug_attention_probabilities",
                   &NativeProjectedQKUnionTileLoopPlan::debug_attention_probabilities)
      .def_prop_ro("debug_projected_union_value_tile",
                   &NativeProjectedQKUnionTileLoopPlan::debug_projected_union_value_tile)
      .def_prop_ro("debug_selected_values",
                   &NativeProjectedQKUnionTileLoopPlan::debug_selected_values);

  nb::class_<NativeSharedPhysicalValueTilePlan>(
      module, "NativeSharedPhysicalValueTilePlan")
      .def(nb::init<int, int, int>(), "physical_k"_a, "tile_rows"_a = 0,
           "query_rows"_a = 64)
      .def("execute", &NativeSharedPhysicalValueTilePlan::execute,
           "selected_probabilities"_a, "selected_indices"_a,
           "selected_valid"_a, "latent"_a, "value_weight"_a)
      .def_prop_ro("physical_k",
                   &NativeSharedPhysicalValueTilePlan::physical_k)
      .def_prop_ro("tile_rows",
                   &NativeSharedPhysicalValueTilePlan::tile_rows)
      .def_prop_ro("tile_count",
                   &NativeSharedPhysicalValueTilePlan::tile_count)
      .def_prop_ro("query_rows",
                   &NativeSharedPhysicalValueTilePlan::query_rows)
      .def_prop_ro("query_blocks",
                   &NativeSharedPhysicalValueTilePlan::query_blocks)
      .def_prop_ro("selected_width",
                   &NativeSharedPhysicalValueTilePlan::selected_width)
      .def_prop_ro("execution_count",
                   &NativeSharedPhysicalValueTilePlan::execution_count)
      .def_prop_ro("dynamic_allocation_count",
                   &NativeSharedPhysicalValueTilePlan::dynamic_allocation_count)
      .def_prop_ro("graph_node_count",
                   &NativeSharedPhysicalValueTilePlan::graph_node_count)
      .def_prop_ro("shape_discovery_count",
                   &NativeSharedPhysicalValueTilePlan::shape_discovery_count)
      .def_prop_ro("host_synchronization_count",
                   &NativeSharedPhysicalValueTilePlan::host_synchronization_count)
      .def_prop_ro("returned_intermediate_tensor_bytes",
                   &NativeSharedPhysicalValueTilePlan::returned_intermediate_tensor_bytes)
      .def_prop_ro("materialized_query_local_selected_value_bytes",
                   &NativeSharedPhysicalValueTilePlan::materialized_query_local_selected_value_bytes)
      .def_prop_ro("scratch_bytes",
                   &NativeSharedPhysicalValueTilePlan::scratch_bytes)
      .def_prop_ro("buffer_identities",
                   &NativeSharedPhysicalValueTilePlan::buffer_identities)
      .def_prop_ro("debug_physical_probabilities",
                   &NativeSharedPhysicalValueTilePlan::debug_physical_probabilities)
      .def_prop_ro("debug_projected_values",
                   &NativeSharedPhysicalValueTilePlan::debug_projected_values);

  nb::class_<NativeQ256DSAPrefillPlan>(module, "NativeQ256DSAPrefillPlan")
      .def(nb::init<int, int>(), "physical_k"_a, "tile_rows"_a = 65536)
      .def("execute", &NativeQ256DSAPrefillPlan::execute,
           "selected_indices"_a, "selected_valid"_a, "latent"_a,
           "key_weight"_a, "value_weight"_a, "attention_query"_a,
           "attention_scale"_a)
      .def_prop_ro("physical_k", &NativeQ256DSAPrefillPlan::physical_k)
      .def_prop_ro("tile_rows", &NativeQ256DSAPrefillPlan::tile_rows)
      .def_prop_ro("tile_count", &NativeQ256DSAPrefillPlan::tile_count)
      .def_prop_ro("query_rows", &NativeQ256DSAPrefillPlan::query_rows)
      .def_prop_ro("execution_count", &NativeQ256DSAPrefillPlan::execution_count)
      .def_prop_ro("dynamic_allocation_count",
                   &NativeQ256DSAPrefillPlan::dynamic_allocation_count)
      .def_prop_ro("graph_node_count", &NativeQ256DSAPrefillPlan::graph_node_count)
      .def_prop_ro("shape_discovery_count",
                   &NativeQ256DSAPrefillPlan::shape_discovery_count)
      .def_prop_ro("host_synchronization_count",
                   &NativeQ256DSAPrefillPlan::host_synchronization_count)
      .def_prop_ro("returned_intermediate_tensor_bytes",
                   &NativeQ256DSAPrefillPlan::returned_intermediate_tensor_bytes)
      .def_prop_ro("materialized_selected_key_bytes",
                   &NativeQ256DSAPrefillPlan::materialized_selected_key_bytes)
      .def_prop_ro("materialized_query_local_selected_value_bytes",
                   &NativeQ256DSAPrefillPlan::materialized_query_local_selected_value_bytes)
      .def_prop_ro("scratch_bytes", &NativeQ256DSAPrefillPlan::scratch_bytes)
      .def_prop_ro("buffer_identities",
                   &NativeQ256DSAPrefillPlan::buffer_identities)
      .def_prop_ro("debug_probabilities",
                   &NativeQ256DSAPrefillPlan::debug_probabilities)
      .def_prop_ro("debug_scores", &NativeQ256DSAPrefillPlan::debug_scores)
      .def_prop_ro("debug_union_count",
                   &NativeQ256DSAPrefillPlan::debug_union_count);
}
