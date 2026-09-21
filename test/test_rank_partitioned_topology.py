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
    and node.name
    in {
        "_configure_partitioned_store",
        "_configure_numa_placement",
        "_cache_unique_id",
    }
}
current_platform = types.SimpleNamespace(device_type="npu")
namespace = {
    "Any": object,
    "KVConnectorRole": ROLE,
    "current_platform": current_platform,
}
exec(
    compile(
        ast.Module(body=list(methods.values()), type_ignores=[]), str(SOURCE), "exec"
    ),
    namespace,
)
configure = namespace["_configure_partitioned_store"]
configure_numa = namespace["_configure_numa_placement"]
cache_unique_id = namespace["_cache_unique_id"]


class PartitionedBufferTopologyTest(unittest.TestCase):
    def worker(self, dp=0, rank=0, pp=1, tp=8, local_dp_rank=None):
        parallel = types.SimpleNamespace(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            data_parallel_rank=dp,
            data_parallel_rank_local=(dp if local_dp_rank is None else local_dp_rank),
            data_parallel_index=dp,
            rank=rank,
        )
        return types.SimpleNamespace(
            is_mla=True,
            tp_size=tp,
            _role=ROLE.WORKER,
            _vllm_config=types.SimpleNamespace(parallel_config=parallel),
        )

    def config(self):
        return dict(
            unique_id="instance",
            device_id=15,
        )

    def test_gqa_cache_domain_is_scoped_by_data_parallel_index(self):
        for dp_rank in (0, 3):
            connector = self.worker(dp=dp_rank, local_dp_rank=0)
            connector.is_mla = False
            connector.unique_id = "instance"
            self.assertEqual(cache_unique_id(connector), f"instance_dp{dp_rank}")
            self.assertEqual(
                cache_unique_id(connector, "_fawa_fa"),
                f"instance_dp{dp_rank}_fawa_fa",
            )

    def test_mla_cache_domain_is_shared_across_data_parallel_ranks(self):
        connector = self.worker(dp=3)
        connector.unique_id = "instance"
        self.assertEqual(cache_unique_id(connector), "instance")
        self.assertEqual(cache_unique_id(connector, "_fawa_fa"), "instance_fawa_fa")

    def test_gqa_dp2tp4_has_one_control_domain_and_four_segments_per_dp(self):
        domains = {}
        for dp_rank in range(2):
            for tp_rank in range(4):
                worker = self.worker(dp=dp_rank, rank=tp_rank, tp=4)
                worker.is_mla = False
                worker.unique_id = "instance"
                worker._partitioned_buffer_topology = (tp_rank, 4)
                config = {"unique_id": cache_unique_id(worker)}
                configure(worker, config)
                domains.setdefault(config["unique_id"], set()).add(
                    config["share_buffer_rank"]
                )
                self.assertEqual(config["share_buffer_segment_count"], 4)
        self.assertEqual(
            domains,
            {"instance_dp0": set(range(4)), "instance_dp1": set(range(4))},
        )

    def partitioned_domains(self, is_mla, dp_size=4, tp_size=4):
        scheduler_domains = set()
        worker_segments = set()
        workers_per_segment = {}
        for dp_rank in range(dp_size):
            scheduler = self.worker(dp=dp_rank, rank=0, tp=tp_size)
            scheduler.is_mla = is_mla
            scheduler.unique_id = "instance"
            scheduler._role = ROLE.SCHEDULER
            scheduler_config = {"unique_id": cache_unique_id(scheduler)}
            configure(scheduler, scheduler_config)
            self.assertEqual(scheduler_config["share_buffer_segment_count"], tp_size)
            self.assertNotIn("share_buffer_rank", scheduler_config)
            scheduler_domains.add(scheduler_config["unique_id"])

            for tp_rank in range(tp_size):
                worker = self.worker(
                    dp=dp_rank,
                    rank=dp_rank * tp_size + tp_rank,
                    tp=tp_size,
                )
                worker.is_mla = is_mla
                worker.unique_id = "instance"
                worker._partitioned_buffer_topology = (tp_rank, tp_size)
                worker_config = {"unique_id": cache_unique_id(worker)}
                configure(worker, worker_config)
                self.assertEqual(worker_config["share_buffer_segment_count"], tp_size)
                self.assertEqual(worker_config["share_buffer_rank"], tp_rank)
                segment = (worker_config["unique_id"], tp_rank)
                worker_segments.add(segment)
                workers_per_segment[segment] = workers_per_segment.get(segment, 0) + 1
        return scheduler_domains, worker_segments, workers_per_segment

    def test_mla_dp4tp4_has_one_control_and_four_data_segments(self):
        domains, segments, workers_per_segment = self.partitioned_domains(True)
        self.assertEqual(domains, {"instance"})
        self.assertEqual(segments, {("instance", rank) for rank in range(4)})
        self.assertEqual(set(workers_per_segment.values()), {4})

    def test_gqa_dp4tp4_has_four_controls_and_sixteen_data_segments(self):
        domains, segments, workers_per_segment = self.partitioned_domains(False)
        self.assertEqual(domains, {f"instance_dp{rank}" for rank in range(4)})
        self.assertEqual(
            segments,
            {
                (f"instance_dp{dp_rank}", tp_rank)
                for dp_rank in range(4)
                for tp_rank in range(4)
            },
        )
        self.assertEqual(set(workers_per_segment.values()), {1})

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
        self.assertNotIn("share_buffer_enable", config)
        self.assertEqual(config["share_buffer_segment_count"], 8)
        self.assertEqual(config["share_buffer_rank"], 3)

    def test_legacy_share_buffer_setting_is_ignored(self):
        config = {"share_buffer_enable": False, "unique_id": "instance"}
        worker = self.worker()
        worker._partitioned_buffer_topology = (3, 8)
        configure(worker, config)
        self.assertNotIn("share_buffer_enable", config)
        self.assertEqual(config["share_buffer_segment_count"], 8)
        self.assertEqual(config["share_buffer_rank"], 3)

    def test_gqa_defaults_to_dp_scoped_tp_partitioned_buffer(self):
        worker = self.worker()
        worker.is_mla = False
        worker._partitioned_buffer_topology = (3, 8)
        config = {"unique_id": "instance"}
        configure(worker, config)
        self.assertNotIn("share_buffer_enable", config)
        self.assertEqual(config["share_buffer_segment_count"], 8)
        self.assertEqual(config["local_rank_size"], 8)
        self.assertEqual(config["share_buffer_rank"], 3)

    def test_gqa_shared_buffer_is_also_partitioned(self):
        worker = self.worker()
        worker.is_mla = False
        worker._partitioned_buffer_topology = (5, 8)
        config = self.config()
        configure(worker, config)
        self.assertNotIn("share_buffer_enable", config)
        self.assertEqual(config["share_buffer_segment_count"], 8)
        self.assertEqual(config["share_buffer_rank"], 5)
        self.assertEqual(config["local_rank_size"], 8)

    def test_a3_gqa_dp8tp1_spreads_by_local_worker_rank(self):
        for dp_rank in range(8):
            worker = self.worker(dp=dp_rank, rank=0, tp=1)
            worker.is_mla = False
            worker.device_id = dp_rank
            worker.device = Mock()
            worker.device.get_numa_node.return_value = None
            config = {}
            configure_numa(worker, config)
            self.assertEqual(config["cache_fallback_numa_rank"], dp_rank)

    def test_a3_gqa_dp2tp8_uses_full_worker_position(self):
        worker = self.worker(dp=1, rank=7, tp=8)
        worker.is_mla = False
        worker.device_id = 15
        worker.device = Mock()
        worker.device.get_numa_node.return_value = None
        config = {}
        configure_numa(worker, config)
        self.assertEqual(config["cache_fallback_numa_rank"], 15)

    def test_a3_mla_dp8tp1_keeps_one_shared_segment(self):
        for dp_rank in range(8):
            worker = self.worker(dp=dp_rank, rank=0, tp=1)
            worker._partitioned_buffer_topology = (0, 1)
            worker.device_id = dp_rank
            worker.device = Mock()
            worker.device.get_numa_node.return_value = None
            config = {"unique_id": "instance"}
            configure(worker, config)
            configure_numa(worker, config)
            self.assertEqual(config["unique_id"], "instance")
            self.assertEqual(config["share_buffer_segment_count"], 1)
            self.assertEqual(config["share_buffer_rank"], 0)
            self.assertNotIn("cache_fallback_numa_rank", config)

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
