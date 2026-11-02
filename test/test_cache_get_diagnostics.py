"""Diagnostic deltas must survive destructive snapshots and late dispatch completion."""

import contextlib
import importlib.util
import io
import multiprocessing
import runpy
import sys
import unittest
from pathlib import Path
from unittest import mock

from scripts import collect_ucm_metrics as collector

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "cache_bw_diagnostics_test", ROOT / "ucm/store/test/e2e/cache_bw.py"
)
bw = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bw
SPEC.loader.exec_module(bw)
BOUNDS = (0.001, 0.01, 0.1, 1.0, 10.0, float("inf"))
MARKER = "cache_load_diagnostics_tasks_total"


def histogram(count=1, total=0.002):
    return ([0, count, 0, 0, 0, 0], total)


class Snapshots:
    def __init__(self, snapshots):
        self.snapshots = iter(snapshots)
        self.reads = 0

    def get_all_stats_and_clear(self):
        self.reads += 1
        return next(self.snapshots)


class CacheGetDiagnosticsTest(unittest.TestCase):
    def collect(self, source, **kwargs):
        return bw.collect_load_metrics(
            source,
            BOUNDS,
            1,
            4096,
            1000,
            diagnostics_bounds=BOUNDS,
            expected_gets=1,
            **kwargs,
        )

    def test_waits_for_dispatch_flush_after_load_completion_and_merges_deltas(self):
        source = Snapshots(
            [
                ({bw.LOAD_BYTES: 4096}, {}, {bw.LOAD_DURATION: histogram()}),
                (
                    {"cache_load_get_peer_unclaimed_total": 42},
                    {},
                    {bw.DIAGNOSTIC_HISTOGRAMS[1]: histogram(total=0.008)},
                ),
                ({MARKER: 1, "cache_load_prealloc_created_total": 3}, {}, {}),
            ]
        )
        sample = self.collect(source)
        self.assertEqual(source.reads, 3)
        self.assertEqual(sample.diagnostics.counters[MARKER], 1)
        self.assertEqual(
            sample.diagnostics.counters["cache_load_get_peer_unclaimed_total"], 42
        )
        self.assertEqual(
            sample.diagnostics.histograms[bw.DIAGNOSTIC_HISTOGRAMS[1]][1], 0.008
        )

    def test_missing_native_diagnostics_fail_instead_of_reporting_zero(self):
        source = Snapshots(
            [
                ({bw.LOAD_BYTES: 4096}, {}, {bw.LOAD_DURATION: histogram()}),
            ]
        )
        with mock.patch.object(bw.time, "monotonic", side_effect=[0, 2]):
            with self.assertRaisesRegex(RuntimeError, "native Cache2 rebuild"):
                self.collect(source)

    def test_completion_marker_cannot_hide_missing_get_samples(self):
        source = Snapshots(
            [
                ({bw.LOAD_BYTES: 4096, MARKER: 1}, {}, {bw.LOAD_DURATION: histogram()}),
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "Get samples=0/1"):
            self.collect(source)

    def test_wrong_histogram_layout_and_invalid_counters_are_rejected(self):
        for counters, histograms in [
            ({MARKER: -1}, {}),
            ({MARKER: 0.5}, {}),
            ({}, {bw.DIAGNOSTIC_HISTOGRAMS[0]: ([1], 0.1)}),
            ({}, {bw.DIAGNOSTIC_HISTOGRAMS[0]: histogram(total=float("nan"))}),
        ]:
            with self.subTest(counters=counters, histograms=histograms):
                with self.assertRaises(RuntimeError):
                    bw.DiagnosticMetrics().merge(counters, histograms, BOUNDS)

    def test_diagnostic_roles_and_counters_survive_shared_result_storage(self):
        args, workload = bw.parse_args(
            [
                "--mode",
                "cache-hit",
                "--model",
                "custom",
                "--workers",
                "1",
                "--epochs",
                "1",
                "--blocks",
                "1",
                "--get-diagnostics",
            ]
        )
        with (
            mock.patch.object(bw, "load_duration_buckets", return_value=BOUNDS),
            mock.patch.object(bw, "diagnostic_buckets", return_value=BOUNDS),
        ):
            results = bw.create_results(
                multiprocessing.get_context("spawn"), args, workload
            )
        diagnostics = bw.DiagnosticMetrics()
        diagnostics.merge(
            {MARKER: 1, "cache_load_get_retries_total": 7},
            {
                bw.DIAGNOSTIC_HISTOGRAMS[0]: histogram(),
                bw.DIAGNOSTIC_HISTOGRAMS[1]: histogram(3, 0.03),
            },
            BOUNDS,
        )
        bw.save_result(
            results, 0, bw.LoadMetrics(1.0, 4096, histogram()[0], diagnostics), [0.001]
        )
        restored = bw.DiagnosticMetrics.unpack(list(results.diagnostic_values), BOUNDS)
        self.assertEqual(restored.counters["cache_load_get_retries_total"], 7)
        self.assertEqual(sum(restored.histograms[bw.DIAGNOSTIC_HISTOGRAMS[0]][0]), 1)
        self.assertEqual(sum(restored.histograms[bw.DIAGNOSTIC_HISTOGRAMS[1]][0]), 3)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            bw.print_diagnostics("summary", "0", "ALL", restored, BOUNDS)
        self.assertIn("sum_ms=0.030000", output.getvalue())
        self.assertIn('"cache_load_get_retries_total": 7.0', output.getvalue())

    def test_legacy_rejected_and_seed_has_diagnostics_disabled(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            bw.parse_args(
                ["--mode", "cache-hit", "--cache-core", "legacy", "--get-diagnostics"]
            )
        args, workload = bw.parse_args(["--mode", "posix-hit", "--get-diagnostics"])
        self.assertTrue(
            bw.store_config(args, workload, "run", 0)["cache_load_diagnostics"]
        )
        self.assertFalse(
            bw.store_config(args, workload, "run", 0, seed=True)[
                "cache_load_diagnostics"
            ]
        )

    def test_collector_reports_counter_deltas_per_rank_and_aggregates(self):
        name = "cache_load_get_peer_unclaimed_total"
        before = collector.parse_samples(
            f'ucm:{name}{{worker_rank="0"}} 10\nucm:{name}{{worker_rank="1"}} 100\n'
        )
        after = collector.parse_samples(
            f'ucm:{name}{{worker_rank="0"}} 15\nucm:{name}{{worker_rank="1"}} 120\n'
        )
        rows = collector.build_page_rows("cachestore", (), before, after, 1, False)
        self.assertEqual(
            {row.group.worker_rank: row.count for row in rows}, {"0": 5, "1": 20}
        )
        combined = collector.build_page_rows("cachestore", (), before, after, 1, True)
        self.assertEqual(combined[0].count, 25)
        self.assertEqual(combined[0].unit, "events")

    def test_default_definitions_cover_all_consumers_and_microsecond_buckets(self):
        config = runpy.run_path(str(ROOT / "ucm/default_metrics_config.py"))[
            "get_default_metrics_config"
        ]()
        histograms = {item["name"]: item for item in config["histogram"]}
        counters = {item["name"] for item in config["counter"]}
        self.assertTrue(set(bw.DIAGNOSTIC_COUNTERS) <= counters)
        self.assertEqual(
            set(bw.DIAGNOSTIC_COUNTERS), set(collector.CACHESTORE_DIAGNOSTIC_COUNTERS)
        )
        collector_histograms = dict(collector.CACHESTORE_HISTOGRAMS)
        for name in bw.DIAGNOSTIC_HISTOGRAMS:
            self.assertIn(name, collector_histograms)
            self.assertLessEqual(histograms[name]["buckets"][0], 0.001)


if __name__ == "__main__":
    unittest.main()
