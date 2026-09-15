"""Test NPU topology parsing without importing torch or vLLM."""

import ast
import re
import unittest
from pathlib import Path
from typing import Dict, List, Optional

SOURCE = Path(__file__).resolve().parents[1] / "ucm/integration/vllm/device.py"
FUNCTIONS = {
    "_expand_cpu_list",
    "_parse_npu_topo_affinity",
    "_parse_cpu_numa_map",
    "_resolve_npu_numa_node",
}
tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
body = [
    node
    for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    and node.name in FUNCTIONS
]
namespace = {
    "re": re,
    "Dict": Dict,
    "List": List,
    "Optional": Optional,
}
exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), namespace)
parse_affinity = namespace["_parse_npu_topo_affinity"]
resolve_node = namespace["_resolve_npu_numa_node"]


class NpuNumaTopologyTest(unittest.TestCase):
    A2_TOPO = """
               NPU0       NPU1       NPU2       NPU3       CPU Affinity
    NPU0       X          HCCS       HCCS       HCCS       0-3
    NPU1       HCCS       X          HCCS       HCCS       4-7
    NPU2       HCCS       HCCS       X          HCCS       8-11
    NPU3       HCCS       HCCS       HCCS       X          12-15
    """
    CPU_NUMA = """
    CPU NODE
      0    0
      1    0
      2    0
      3    0
      4    0
      5    0
      6    0
      7    0
      8    1
      9    1
     10    1
     11    1
     12    1
     13    1
     14    1
     15    1
    """

    def test_two_devices_resolve_to_each_numa_node(self):
        self.assertEqual(resolve_node(self.A2_TOPO, self.CPU_NUMA, 0), 0)
        self.assertEqual(resolve_node(self.A2_TOPO, self.CPU_NUMA, 1), 0)
        self.assertEqual(resolve_node(self.A2_TOPO, self.CPU_NUMA, 2), 1)
        self.assertEqual(resolve_node(self.A2_TOPO, self.CPU_NUMA, 3), 1)

    def test_topology_without_cpu_affinity_falls_back(self):
        a3_topo = """
                   NPU0       NPU1       NIC0
        NPU0       X          UB         NA
        NPU1       UB         X          NA
        """
        self.assertEqual(parse_affinity(a3_topo), {})
        self.assertIsNone(resolve_node(a3_topo, self.CPU_NUMA, 0))

    def test_ambiguous_or_incomplete_cpu_mapping_falls_back(self):
        self.assertIsNone(resolve_node("NPU0 X 0-16", self.CPU_NUMA, 0))
        mixed = "CPU NODE\n0 0\n1 1"
        self.assertIsNone(resolve_node("NPU0 X 0-1", mixed, 0))


if __name__ == "__main__":
    unittest.main()
