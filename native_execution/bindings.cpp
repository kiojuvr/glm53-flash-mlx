#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "native_indexer_plan.h"

namespace nb = nanobind;
using namespace nb::literals;
using glm53::native_execution::NativeIndexSelectionPlan;

NB_MODULE(_ext, module) {
  module.doc() = "Probe-only persistent native execution bridge for GLM-5.3";
  nb::class_<NativeIndexSelectionPlan>(module, "NativeIndexSelectionPlan")
      .def(
          nb::init<std::string, int, int, std::string>(),
          "mode"_a,
          "query_rows"_a,
          "physical_pool_rows"_a,
          "score_dtype"_a)
      .def(
          "execute",
          &NativeIndexSelectionPlan::execute,
          "scores"_a,
          "pool_indices"_a,
          "pool_valid"_a,
          "raw_positions"_a,
          "raw_valid"_a,
          "current_valid"_a,
          "logical_pool_rows"_a,
          "kv_len"_a,
          "active_tail_count"_a)
      .def_prop_ro("mode", &NativeIndexSelectionPlan::mode)
      .def_prop_ro("query_rows", &NativeIndexSelectionPlan::query_rows)
      .def_prop_ro(
          "physical_pool_rows", &NativeIndexSelectionPlan::physical_pool_rows)
      .def_prop_ro("selected_width", &NativeIndexSelectionPlan::selected_width)
      .def_prop_ro(
          "execution_count", &NativeIndexSelectionPlan::execution_count)
      .def_prop_ro(
          "dynamic_allocation_count",
          &NativeIndexSelectionPlan::dynamic_allocation_count)
      .def_prop_ro("graph_node_count", &NativeIndexSelectionPlan::graph_node_count)
      .def_prop_ro(
          "shape_discovery_count",
          &NativeIndexSelectionPlan::shape_discovery_count)
      .def_prop_ro(
          "host_synchronization_count",
          &NativeIndexSelectionPlan::host_synchronization_count)
      .def_prop_ro(
          "buffer_identities", &NativeIndexSelectionPlan::buffer_identities);
}
