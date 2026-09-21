"""Pure placement-matrix tests for cache_numa_topology_live.py."""

import unittest
from test.cache_numa_topology_live import (
    _parse_cuda_gpu_bus_ids,
    _parse_cuda_topo_affinity,
    _worker_placements,
)


class CacheNumaTopologyLiveMatrixTest(unittest.TestCase):
    devices = list(range(8))
    nodes = list(range(8))

    def targets(self, placements):
        return [item.owner_target for item in placements]

    def test_a3_gqa_dp1tp8_spreads_by_local_worker_rank(self):
        placements = _worker_placements(
            "GQA", 1, 8, self.devices, dict.fromkeys(self.devices), self.nodes
        )
        self.assertEqual(self.targets(placements), [(node,) for node in self.nodes])

    def test_a3_gqa_dp8tp1_spreads_by_local_worker_rank(self):
        placements = _worker_placements(
            "GQA", 8, 1, self.devices, dict.fromkeys(self.devices), self.nodes
        )
        self.assertEqual(self.targets(placements), [(node,) for node in self.nodes])

    def test_a3_mla_dp1tp8_stripes_segments(self):
        placements = _worker_placements(
            "MLA", 1, 8, self.devices, dict.fromkeys(self.devices), self.nodes
        )
        self.assertEqual(self.targets(placements), [(node,) for node in self.nodes])

    def test_a3_mla_dp8tp1_keeps_one_shared_segment_on_one_node(self):
        placements = _worker_placements(
            "MLA", 8, 1, self.devices, dict.fromkeys(self.devices), self.nodes
        )
        self.assertEqual(self.targets(placements), [(0,)] * 8)

    def test_a2_topology_overrides_fallback_for_both_models(self):
        detected = {device: device // 2 for device in self.devices}
        for model in ("GQA", "MLA"):
            with self.subTest(model=model):
                placements = _worker_placements(
                    model, 1, 8, self.devices, detected, self.nodes
                )
                self.assertEqual(
                    self.targets(placements),
                    [(device // 2,) for device in self.devices],
                )

    def test_cuda_gqa_uses_detected_gpu_numa_nodes(self):
        detected = {device: device // 4 for device in self.devices}
        placements = _worker_placements(
            "GQA",
            1,
            8,
            self.devices,
            detected,
            [0, 1],
            platform_type="cuda",
        )
        self.assertEqual(self.targets(placements), [(0,)] * 4 + [(1,)] * 4)

    def test_cuda_gqa_without_topology_uses_segment_placement(self):
        placements = _worker_placements(
            "GQA",
            1,
            8,
            self.devices,
            dict.fromkeys(self.devices),
            [0, 1],
            platform_type="cuda",
        )
        self.assertEqual(self.targets(placements), [(0,), (1,)] * 4)

    def test_parse_h100_topology_and_pci_bus_ids(self):
        topo = """
                GPU0    GPU1    CPU Affinity    NUMA Affinity    GPU NUMA ID
        GPU0     X       NV18    0-47,96-143     0                N/A
        GPU1     NV18    X       48-95,144-191   1                N/A
        """
        self.assertEqual(
            _parse_cuda_topo_affinity(topo),
            {
                0: list(range(0, 48)) + list(range(96, 144)),
                1: list(range(48, 96)) + list(range(144, 192)),
            },
        )
        self.assertEqual(
            _parse_cuda_gpu_bus_ids("0, 00000000:1A:00.0\n1, 00000000:3D:00.0\n"),
            {0: "0000:1a:00.0", 1: "0000:3d:00.0"},
        )


if __name__ == "__main__":
    unittest.main()
