# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import argparse
import collections
import re
import sys

TRACE_MARKER = "[S2H_TRACE]"
FIELD_PATTERN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")
ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
READY_WAIT_THRESHOLD_NS = 10_000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize cache_posix_bw.py S2H trace logs."
    )
    parser.add_argument(
        "log_file",
        help="Benchmark log path, or '-' to read from standard input.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        help="Only summarize this measured load epoch.",
    )
    parser.add_argument(
        "--show-threads",
        action="store_true",
        help="Print task counts for every Posix PID/TID pair.",
    )
    parser.add_argument(
        "--show-xfer-worker",
        type=int,
        help="Print per-shard backend-wait/H2D timing for this worker.",
    )
    return parser.parse_args()


def open_log(path):
    if path == "-":
        return sys.stdin
    return open(path, "r", encoding="utf-8", errors="replace")


def parse_int(fields, name):
    value = fields.get(name)
    if value is None:
        raise ValueError(f"missing field {name}")
    return int(value)


def parse_optional_int(fields, name, default=-1):
    value = fields.get(name)
    return default if value is None else int(value)


def parse_log(stream):
    epochs = collections.defaultdict(
        lambda: {
            "assignments": [],
            "preads": [],
            "xfers": [],
            "load_begin": [],
            "load_end": [],
            "epoch_begin": [],
            "epoch_end": [],
        }
    )
    malformed = []
    for line_number, raw_line in enumerate(stream, start=1):
        line = ANSI_PATTERN.sub("", raw_line)
        marker_pos = line.find(TRACE_MARKER)
        if marker_pos < 0:
            continue
        fields = dict(FIELD_PATTERN.findall(line[marker_pos + len(TRACE_MARKER) :]))
        event = fields.get("event")
        try:
            epoch = parse_int(fields, "epoch")
            if event == "assign":
                epochs[epoch]["assignments"].append(
                    {
                        "worker": parse_int(fields, "worker"),
                        "pid": parse_int(fields, "pid"),
                        "tid": parse_int(fields, "tid"),
                        "cache_task": parse_int(fields, "cache_task"),
                        "backend_tasks": parse_int(fields, "backend_tasks"),
                        "total_shards": parse_int(fields, "total_shards"),
                    }
                )
            elif event == "pread":
                epochs[epoch]["preads"].append(
                    {
                        "worker": parse_int(fields, "worker"),
                        "pid": parse_int(fields, "pid"),
                        "tid": parse_int(fields, "tid"),
                        "posix_task": parse_int(fields, "posix_task"),
                        "block_hash": parse_optional_int(fields, "block_hash"),
                        "shard": parse_int(fields, "shard"),
                        "io_index": parse_int(fields, "io_index"),
                        "offset": parse_int(fields, "offset"),
                        "size": parse_int(fields, "size"),
                        "start_ns": parse_int(fields, "start_ns"),
                        "end_ns": parse_int(fields, "end_ns"),
                        "duration_ns": parse_int(fields, "duration_ns"),
                        "active_at_start": parse_int(fields, "active_at_start"),
                        "active_after": parse_int(fields, "active_after"),
                        "status": fields.get("status", "unknown"),
                    }
                )
            elif event == "xfer":
                epochs[epoch]["xfers"].append(
                    {
                        "worker": parse_int(fields, "worker"),
                        "pid": parse_int(fields, "pid"),
                        "tid": parse_int(fields, "tid"),
                        "cache_task": parse_int(fields, "cache_task"),
                        "shard_pos": parse_int(fields, "shard_pos"),
                        "total_shards": parse_int(fields, "total_shards"),
                        "block_hash": parse_optional_int(fields, "block_hash"),
                        "shard": parse_int(fields, "shard"),
                        "backend_owner": parse_int(fields, "backend_owner"),
                        "wait_start_ns": parse_int(fields, "wait_start_ns"),
                        "ready_ns": parse_int(fields, "ready_ns"),
                        "wait_ns": parse_int(fields, "wait_ns"),
                        "h2d_start_ns": parse_int(fields, "h2d_start_ns"),
                        "h2d_end_ns": parse_int(fields, "h2d_end_ns"),
                        "h2d_submit_ns": parse_int(fields, "h2d_submit_ns"),
                        "sync_start_ns": parse_int(fields, "sync_start_ns"),
                        "sync_end_ns": parse_int(fields, "sync_end_ns"),
                        "sync_ns": parse_int(fields, "sync_ns"),
                        "final": parse_int(fields, "final"),
                        "status": fields.get("status", "unknown"),
                    }
                )
            elif event == "load_begin":
                epochs[epoch]["load_begin"].append(
                    {
                        "worker": parse_int(fields, "worker"),
                        "pid": parse_int(fields, "pid"),
                        "start_ns": parse_int(fields, "start_ns"),
                    }
                )
            elif event == "load_end":
                epochs[epoch]["load_end"].append(
                    {
                        "worker": parse_int(fields, "worker"),
                        "pid": parse_int(fields, "pid"),
                        "end_ns": parse_int(fields, "end_ns"),
                        "duration_ns": parse_int(fields, "duration_ns"),
                    }
                )
            elif event in ("epoch_begin", "epoch_end"):
                epochs[epoch][event].append(
                    {
                        "worker": parse_int(fields, "worker"),
                        "pid": parse_int(fields, "pid"),
                    }
                )
            else:
                malformed.append((line_number, f"unknown event {event!r}"))
        except (TypeError, ValueError) as error:
            malformed.append((line_number, str(error)))
    return epochs, malformed


def percentile(sorted_values, percent):
    if not sorted_values:
        return 0.0
    position = (len(sorted_values) - 1) * percent / 100
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def interval_concurrency(intervals):
    valid = [(start, end) for start, end in intervals if end > start]
    if not valid:
        return {"average": 0.0, "peak": 0, "window_ns": 0, "total_ns": 0}

    changes = collections.defaultdict(lambda: [0, 0])
    for start, end in valid:
        changes[start][1] += 1
        changes[end][0] += 1

    active = 0
    peak = 0
    for timestamp in sorted(changes):
        ends, starts = changes[timestamp]
        active -= ends
        active += starts
        peak = max(peak, active)

    first_start = min(start for start, _ in valid)
    last_end = max(end for _, end in valid)
    window_ns = last_end - first_start
    total_ns = sum(end - start for start, end in valid)
    average = total_ns / window_ns if window_ns > 0 else 0.0
    return {
        "average": average,
        "peak": peak,
        "window_ns": window_ns,
        "total_ns": total_ns,
    }


def summarize_overlap(records, show_xfer_worker):
    xfers = records["xfers"]
    if not xfers:
        if records["load_begin"]:
            print(
                "\nS2H/H2D overlap trace: no xfer records. Rebuild/install the "
                "modified UCM C++ library and use shard launch mode."
            )
        return

    xfers_by_worker = collections.defaultdict(list)
    for record in xfers:
        xfers_by_worker[record["worker"]].append(record)

    load_begin_by_worker = {
        record["worker"]: record["start_ns"] for record in records["load_begin"]
    }
    load_duration_by_worker = {
        record["worker"]: record["duration_ns"] for record in records["load_end"]
    }
    pread_ready_by_shard = {}
    for record in records["preads"]:
        if record["status"] != "ok" or record["block_hash"] < 0:
            continue
        key = (record["block_hash"], record["shard"])
        pread_ready_by_shard[key] = max(
            pread_ready_by_shard.get(key, 0), record["end_ns"]
        )

    print(
        "\nBackend-ready wait in ucm_load_xfer "
        f"(immediate means <= {READY_WAIT_THRESHOLD_NS / 1000:.0f} us)"
    )
    print(
        "worker  shards  owner  immediate  waited  wait_sum_ms  "
        "wait_p99_us  wait_max_us"
    )
    overlap_by_worker = {}
    hol_by_worker = {}
    for worker in sorted(xfers_by_worker):
        worker_xfers = sorted(
            xfers_by_worker[worker], key=lambda record: record["shard_pos"]
        )
        waits_ns = sorted(record["wait_ns"] for record in worker_xfers)
        immediate = sum(duration <= READY_WAIT_THRESHOLD_NS for duration in waits_ns)
        owner_count = sum(record["backend_owner"] for record in worker_xfers)
        print(
            f"{worker:>6}  {len(worker_xfers):>6}  {owner_count:>5}  "
            f"{immediate:>9}  {len(worker_xfers) - immediate:>6}  "
            f"{sum(waits_ns) / 1e6:>11.3f}  "
            f"{percentile(waits_ns, 99) / 1000:>11.3f}  "
            f"{max(waits_ns, default=0) / 1000:>11.3f}"
        )

        h2d_xfers = sorted(
            (
                record
                for record in worker_xfers
                if record["h2d_end_ns"] > record["h2d_start_ns"]
            ),
            key=lambda record: record["h2d_start_ns"],
        )
        gaps_ns = [
            max(0, current["h2d_start_ns"] - previous["h2d_end_ns"])
            for previous, current in zip(h2d_xfers, h2d_xfers[1:])
        ]
        first_h2d_delay_ns = 0
        if h2d_xfers and worker in load_begin_by_worker:
            first_h2d_delay_ns = max(
                0, h2d_xfers[0]["h2d_start_ns"] - load_begin_by_worker[worker]
            )
        submit_span_ns = 0
        if h2d_xfers:
            submit_span_ns = h2d_xfers[-1]["h2d_end_ns"] - h2d_xfers[0]["h2d_start_ns"]
        overlap_by_worker[worker] = {
            "h2d_count": len(h2d_xfers),
            "first_h2d_delay_ns": first_h2d_delay_ns,
            "submit_span_ns": submit_span_ns,
            "gap_sum_ns": sum(gaps_ns),
            "gap_max_ns": max(gaps_ns, default=0),
            "sync_ns": max((record["sync_ns"] for record in worker_xfers), default=0),
            "load_ns": load_duration_by_worker.get(worker, 0),
        }

        hol_waits = 0
        hol_later_ready_total = 0
        hol_later_ready_max = 0
        for index, record in enumerate(worker_xfers):
            if record["wait_ns"] <= READY_WAIT_THRESHOLD_NS:
                continue
            later_ready = 0
            for later in worker_xfers[index + 1 :]:
                ready_ns = pread_ready_by_shard.get(
                    (later["block_hash"], later["shard"]), 0
                )
                if ready_ns and ready_ns < record["ready_ns"]:
                    later_ready += 1
            if later_ready:
                hol_waits += 1
                hol_later_ready_total += later_ready
                hol_later_ready_max = max(hol_later_ready_max, later_ready)
        hol_by_worker[worker] = {
            "waits": hol_waits,
            "later_ready_total": hol_later_ready_total,
            "later_ready_max": hol_later_ready_max,
        }

    if pread_ready_by_shard:
        print("\nFIFO head-of-line evidence")
        print("worker  blocked_heads  later_ready_total  later_ready_max")
        for worker in sorted(hol_by_worker):
            hol = hol_by_worker[worker]
            print(
                f"{worker:>6}  {hol['waits']:>13}  "
                f"{hol['later_ready_total']:>17}  {hol['later_ready_max']:>15}"
            )

    print("\nH2D feed timing (CPU submission timeline, not device execution time)")
    print(
        "worker  h2ds  first_h2d_ms  submit_span_ms  gap_sum_ms  "
        "gap_max_us  final_sync_ms  load_ms"
    )
    for worker in sorted(overlap_by_worker):
        timing = overlap_by_worker[worker]
        print(
            f"{worker:>6}  {timing['h2d_count']:>4}  "
            f"{timing['first_h2d_delay_ns'] / 1e6:>12.3f}  "
            f"{timing['submit_span_ns'] / 1e6:>14.3f}  "
            f"{timing['gap_sum_ns'] / 1e6:>10.3f}  "
            f"{timing['gap_max_ns'] / 1000:>10.3f}  "
            f"{timing['sync_ns'] / 1e6:>13.3f}  "
            f"{timing['load_ns'] / 1e6:>7.3f}"
        )

    measured_loads = [
        (worker, timing["load_ns"])
        for worker, timing in overlap_by_worker.items()
        if timing["load_ns"] > 0
    ]
    if measured_loads:
        slowest_worker, slowest_load_ns = max(measured_loads, key=lambda item: item[1])
        print("\nOverlap headline")
        print(f"  slowest load worker    : {slowest_worker}")
        print(f"  group load makespan    : {slowest_load_ns / 1e6:.3f} ms")
        print(
            "  interpretation         : backend wait and submit gaps are transfer-thread "
            "feed stalls; only stalls that empty the device queue are exposed in end-to-end time"
        )

    if show_xfer_worker is not None:
        details = sorted(
            xfers_by_worker.get(show_xfer_worker, []),
            key=lambda record: record["shard_pos"],
        )
        print(f"\nPer-shard transfer timing: worker {show_xfer_worker}")
        print(
            "shard_pos  shard  owner  wait_us  feed_gap_us  h2d_submit_us  "
            "sync_us  final  status"
        )
        previous_h2d_end_ns = 0
        for record in details:
            feed_gap_ns = 0
            if previous_h2d_end_ns and record["h2d_start_ns"]:
                feed_gap_ns = max(0, record["h2d_start_ns"] - previous_h2d_end_ns)
            print(
                f"{record['shard_pos']:>9}  {record['shard']:>5}  "
                f"{record['backend_owner']:>5}  {record['wait_ns'] / 1000:>7.3f}  "
                f"{feed_gap_ns / 1000:>11.3f}  "
                f"{record['h2d_submit_ns'] / 1000:>13.3f}  "
                f"{record['sync_ns'] / 1000:>7.3f}  {record['final']:>5}  "
                f"{record['status']}"
            )
            if record["h2d_end_ns"]:
                previous_h2d_end_ns = record["h2d_end_ns"]


def summarize_epoch(epoch, records, show_threads, show_xfer_worker):
    assignments = records["assignments"]
    preads = records["preads"]

    assigned_by_worker = collections.Counter()
    total_shards_by_worker = {}
    pids_by_worker = collections.defaultdict(set)
    for record in assignments:
        worker = record["worker"]
        assigned_by_worker[worker] += record["backend_tasks"]
        total_shards_by_worker[worker] = record["total_shards"]
        pids_by_worker[worker].add(record["pid"])

    preads_by_worker = collections.Counter(record["worker"] for record in preads)
    success_by_worker = collections.Counter(
        record["worker"] for record in preads if record["status"] == "ok"
    )
    failed_by_worker = preads_by_worker - success_by_worker
    thread_counts = collections.Counter(
        (record["worker"], record["pid"], record["tid"]) for record in preads
    )
    threads_by_worker = collections.defaultdict(set)
    intervals_by_worker = collections.defaultdict(list)
    for record in preads:
        worker = record["worker"]
        pids_by_worker[worker].add(record["pid"])
        threads_by_worker[worker].add((record["pid"], record["tid"]))
        intervals_by_worker[worker].append((record["start_ns"], record["end_ns"]))

    workers = sorted(
        set(assigned_by_worker)
        | set(preads_by_worker)
        | {record["worker"] for record in records["epoch_begin"]}
    )
    expected_unique = max(total_shards_by_worker.values(), default=0)
    total_assigned = sum(assigned_by_worker.values())
    duplicate_submissions = total_assigned - expected_unique

    print("\n" + "=" * 78)
    print(f"S2H trace summary: measured load epoch {epoch}")
    print("=" * 78)
    print("worker  pid       assigned  preads  ok  failed  threads  local_peak")
    for worker in workers:
        pid_text = (
            ",".join(str(pid) for pid in sorted(pids_by_worker.get(worker, set())))
            or "-"
        )
        local = interval_concurrency(intervals_by_worker.get(worker, []))
        print(
            f"{worker:>6}  {pid_text:<8}  {assigned_by_worker[worker]:>8}  "
            f"{preads_by_worker[worker]:>6}  {success_by_worker[worker]:>2}  "
            f"{failed_by_worker[worker]:>6}  "
            f"{len(threads_by_worker.get(worker, set())):>7}  "
            f"{local['peak']:>10}"
        )

    global_intervals = [(record["start_ns"], record["end_ns"]) for record in preads]
    global_concurrency = interval_concurrency(global_intervals)
    durations_us = sorted(record["duration_ns"] / 1000 for record in preads)
    total_bytes = sum(record["size"] for record in preads)
    print("\nTotals")
    print(f"  expected unique shards : {expected_unique}")
    print(f"  assigned backend tasks : {total_assigned}")
    print(f"  duplicate submissions  : {duplicate_submissions}")
    print(f"  recorded pread calls   : {len(preads)}")
    print(f"  successful preads      : {sum(success_by_worker.values())}")
    print(f"  failed preads          : {sum(failed_by_worker.values())}")
    print(f"  recorded bytes         : {total_bytes}")
    print(f"  workers with preads    : {len(intervals_by_worker)}")

    print("\nGlobal pread concurrency")
    print(f"  average concurrency    : {global_concurrency['average']:.3f}")
    print(f"  peak concurrency       : {global_concurrency['peak']}")
    print(f"  trace window           : {global_concurrency['window_ns'] / 1e6:.3f} ms")
    print(f"  summed pread time      : {global_concurrency['total_ns'] / 1e6:.3f} ms")

    if durations_us:
        print("\nPread latency")
        print(
            f"  average                : {sum(durations_us) / len(durations_us):.3f} us"
        )
        print(f"  p50                    : {percentile(durations_us, 50):.3f} us")
        print(f"  p90                    : {percentile(durations_us, 90):.3f} us")
        print(f"  p99                    : {percentile(durations_us, 99):.3f} us")
        print(f"  maximum                : {durations_us[-1]:.3f} us")

    summarize_overlap(records, show_xfer_worker)

    if show_threads:
        print("\nPosix thread task distribution")
        print("worker  pid       tid       pread_tasks")
        for (worker, pid, tid), count in sorted(
            thread_counts.items(), key=lambda item: (item[0][0], -item[1], item[0][2])
        ):
            print(f"{worker:>6}  {pid:<8}  {tid:<8}  {count:>11}")


def main():
    args = parse_args()
    stream = open_log(args.log_file)
    try:
        epochs, malformed = parse_log(stream)
    finally:
        if stream is not sys.stdin:
            stream.close()

    selected_epochs = sorted(epochs)
    if args.epoch is not None:
        selected_epochs = [epoch for epoch in selected_epochs if epoch == args.epoch]
    if not selected_epochs:
        print("No matching [S2H_TRACE] records found.", file=sys.stderr)
        return 2

    for epoch in selected_epochs:
        summarize_epoch(epoch, epochs[epoch], args.show_threads, args.show_xfer_worker)

    if malformed:
        print(
            f"\nWarning: ignored {len(malformed)} malformed trace lines. "
            f"First issue: line {malformed[0][0]}: {malformed[0][1]}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
