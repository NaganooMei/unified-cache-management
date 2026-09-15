"""Run partitioned Buffer topology configuration without importing vLLM or torch."""

import ast
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

SOURCE = Path(__file__).resolve().parents[1] / "ucm/integration/vllm/ucm_connector.py"
ROLE = types.SimpleNamespace(WORKER="worker", SCHEDULER="scheduler")
tree = ast.parse(SOURCE.read_text(encoding="utf-8-sig"))
methods = {
    node.name: node
    for node in ast.walk(tree)
    if isinstance(node, ast.FunctionDef)
    and node.name in {"_configure_partitioned_store", "_configure_numa_placement"}
}
current_platform = types.SimpleNamespace(device_type="npu")
namespace = {
    "Any": object,
    "KVConnectorRole": ROLE,
    "current_platform": current_platform,
}
exec(
    compile(ast.Module(body=list(methods.values()), type_ignores=[]), str(SOURCE), "exec"),
    namespace,
)
configure = namespace["_configure_partitioned_store"]
configure_numa = namespace["_configure_numa_placement"]


class PartitionedBufferTopologyTest(unittest.TestCase):
    def worker(self, dp=0, rank=0, pp=1):
        parallel = types.SimpleNamespace(
            tensor_parallel_size=8,
            pipeline_parallel_size=pp,
            data_parallel_rank=dp,
            rank=rank,
        )
        return types.SimpleNamespace(
            is_mla=True,
            tp_size=8,
            _role=ROLE.WORKER,
            _vllm_config=types.SimpleNamespace(parallel_config=parallel),
        )

    def config(self):
        return dict(
            share_buffer_enable=True,
            unique_id="instance",
            device_id=15,
        )

    def group(self, flags, rank=7, world_size=8):
        module = types.ModuleType("vllm.distributed.parallel_state")
        module.get_tp_group = Mock(
            return_value=types.SimpleNamespace(
                cpu_group=object(), world_size=world_size, rank_in_group=rank
            )
        )
        module.in_the_same_node_as = Mock(return_value=flags)
        return module

    def test_dp_groups_share_namespace_and_use_group_rank(self):
        module = self.group([True] * 8)
        with patch.dict("sys.modules", {module.__name__: module}):
            first, second = self.config(), self.config()
            configure(self.worker(dp=0, rank=7), first)
            configure(self.worker(dp=1, rank=15), second)
        self.assertEqual(first["unique_id"], "instance")
        self.assertEqual(second["unique_id"], first["unique_id"])
        self.assertEqual(second["share_buffer_rank"], 7)
        self.assertEqual(second["share_buffer_segment_count"], 8)
        self.assertEqual(second["device_id"], 15)

    def test_cross_node_fails_before_store_creation(self):
        module = self.group([True] * 4 + [False] * 4)
        with patch.dict("sys.modules", {module.__name__: module}):
            with self.assertRaisesRegex(ValueError, "cross-node TP is unsupported"):
                configure(self.worker(), self.config())

    def test_group_size_mismatch_fails(self):
        module = self.group([True] * 8, world_size=4)
        with patch.dict("sys.modules", {module.__name__: module}):
            with self.assertRaisesRegex(
                ValueError, "does not match configured TP size"
            ):
                configure(self.worker(), self.config())

    def test_probe_once_for_multiple_stores(self):
        module = self.group([True] * 8)
        worker = self.worker()
        with patch.dict("sys.modules", {module.__name__: module}):
            configure(worker, self.config())
            configure(worker, self.config())
        module.in_the_same_node_as.assert_called_once()

    def test_scheduler_sets_segment_count_without_collective(self):
        scheduler = self.worker(dp=1)
        scheduler._role = ROLE.SCHEDULER
        config = self.config()
        configure(scheduler, config)
        self.assertEqual(config["unique_id"], "instance")
        self.assertEqual(config["share_buffer_segment_count"], 8)
        self.assertNotIn("share_buffer_rank", config)

    def test_pipeline_groups_do_not_share_metadata(self):
        worker = self.worker(rank=8, pp=2)
        worker._partitioned_buffer_topology = (0, 8)
        config = self.config()
        configure(worker, config)
        self.assertEqual(config["unique_id"], "instance_pp1")

    def test_mla_defaults_to_partitioned_shared_buffer(self):
        config = {"unique_id": "instance", "device_id": 15}
        worker = self.worker()
        worker._partitioned_buffer_topology = (3, 8)
        configure(worker, config)
        self.assertTrue(config["share_buffer_enable"])
        self.assertEqual(config["share_buffer_segment_count"], 8)
        self.assertEqual(config["share_buffer_rank"], 3)

    def test_mla_honors_explicitly_disabled_shared_buffer(self):
        config = {"share_buffer_enable": False, "unique_id": "instance"}
        configure(self.worker(), config)
        self.assertFalse(config["share_buffer_enable"])
        self.assertNotIn("share_buffer_segment_count", config)
        self.assertNotIn("share_buffer_rank", config)

    def test_gqa_uses_process_local_buffer_by_default(self):
        worker = self.worker()
        worker.is_mla = False
        config = {"unique_id": "instance"}
        configure(worker, config)
        self.assertFalse(config["share_buffer_enable"])
        self.assertNotIn("share_buffer_rank", config)

    def test_gqa_shared_buffer_is_also_partitioned(self):
        worker = self.worker()
        worker.is_mla = False
        worker._partitioned_buffer_topology = (5, 8)
        config = self.config()
        configure(worker, config)
        self.assertTrue(config["share_buffer_enable"])
        self.assertEqual(config["share_buffer_segment_count"], 8)
        self.assertEqual(config["share_buffer_rank"], 5)
        self.assertEqual(config["local_rank_size"], 1)

    def test_gqa_without_topology_falls_back_to_tp_rank(self):
        worker = self.worker(rank=13)
        worker.is_mla = False
        worker.tp_rank = 13
        worker.device_id = 5
        worker.device = Mock()
        worker.device.get_numa_node.return_value = None
        config = {}
        configure_numa(worker, config)
        self.assertEqual(config["cache_fallback_numa_rank"], 5)

    def test_detected_topology_takes_priority_for_gqa_and_mla(self):
        for is_mla in (False, True):
            worker = self.worker()
            worker.is_mla = is_mla
            worker.tp_rank = 7
            worker.device_id = 7
            worker.device = Mock()
            worker.device.get_numa_node.return_value = 3
            config = {}
            configure_numa(worker, config)
            self.assertEqual(config["cache_detected_numa_node"], 3)
            self.assertNotIn("cache_fallback_numa_rank", config)

    def test_scheduler_does_not_probe_numa_topology(self):
        scheduler = self.worker()
        scheduler._role = ROLE.SCHEDULER
        scheduler.device = Mock()
        config = {}
        configure_numa(scheduler, config)
        scheduler.device.get_numa_node.assert_not_called()
        self.assertEqual(config, {})


if __name__ == "__main__":
    unittest.main()
