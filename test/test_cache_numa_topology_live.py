"""Pure placement-matrix tests for cache_numa_topology_live.py."""

import unittest
from test.cache_numa_topology_live import _worker_placements


class CacheNumaTopologyLiveMatrixTest(unittest.TestCase):
    devices = list(range(8))
    nodes = list(range(8))

    def targets(self, placements):
        return [item.owner_target for item in placements]

    def test_a3_gqa_dp1tp8_spreads_by_tp_rank(self):
        placements = _worker_placements(
            "GQA", 1, 8, self.devices, dict.fromkeys(self.devices), self.nodes
        )
        self.assertEqual(self.targets(placements), [(node,) for node in self.nodes])

    def test_a3_gqa_dp8tp1_exposes_tp_rank_concentration(self):
        placements = _worker_placements(
            "GQA", 8, 1, self.devices, dict.fromkeys(self.devices), self.nodes
        )
        self.assertEqual(self.targets(placements), [(0,)] * 8)

    def test_a3_mla_dp1tp8_stripes_segments(self):
        placements = _worker_placements(
            "MLA", 1, 8, self.devices, dict.fromkeys(self.devices), self.nodes
        )
        self.assertEqual(self.targets(placements), [(node,) for node in self.nodes])

    def test_a3_mla_dp8tp1_uses_first_touch(self):
        placements = _worker_placements(
            "MLA", 8, 1, self.devices, dict.fromkeys(self.devices), self.nodes
        )
        self.assertEqual(self.targets(placements), [()] * 8)

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


if __name__ == "__main__":
    unittest.main()
