"""Test CUDA PCI-to-NUMA resolution without importing torch or vLLM."""

import ast
import os
import re
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import mock_open, patch

SOURCE = Path(__file__).resolve().parents[1] / "ucm/integration/vllm/device.py"
FUNCTIONS = {"_normalize_cuda_pci_bus_id", "_resolve_cuda_numa_node"}
tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
body = [
    node
    for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    and node.name in FUNCTIONS
]
namespace = {"os": os, "re": re, "Optional": Optional}
exec(compile(ast.Module(body=body, type_ignores=[]), str(SOURCE), "exec"), namespace)
normalize_bdf = namespace["_normalize_cuda_pci_bus_id"]
resolve_node = namespace["_resolve_cuda_numa_node"]


class CudaNumaTopologyTest(unittest.TestCase):
    def test_normalizes_torch_and_nvml_pci_ids(self):
        self.assertEqual(normalize_bdf("0000:1A:00.0"), "0000:1a:00.0")
        self.assertEqual(normalize_bdf("00000000:3D:00.0"), "0000:3d:00.0")
        self.assertIsNone(normalize_bdf("not-a-pci-id"))

    def test_reads_nonnegative_sysfs_numa_node(self):
        reader = mock_open(read_data="3\n")
        with patch("builtins.open", reader):
            self.assertEqual(resolve_node("00000000:1A:00.0", "/sys/pci"), 3)
        reader.assert_called_once_with(
            os.path.join("/sys/pci", "0000:1a:00.0", "numa_node")
        )

    def test_missing_or_unknown_sysfs_numa_node_falls_back(self):
        with patch("builtins.open", mock_open(read_data="-1\n")):
            self.assertIsNone(resolve_node("0000:1a:00.0", "/sys/pci"))
        with patch("builtins.open", side_effect=FileNotFoundError):
            self.assertIsNone(resolve_node("0000:3d:00.0", "/sys/pci"))


if __name__ == "__main__":
    unittest.main()
