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


def parse_log(stream):
    epochs = collections.defaultdict(
        lambda: {
            "assignments": [],
            "preads": [],
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


def summarize_epoch(epoch, records, show_threads):
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
    print(
        "worker  pid       assigned  preads  ok  failed  threads  local_peak"
    )
    for worker in workers:
        pid_text = ",".join(str(pid) for pid in sorted(pids_by_worker[worker])) or "-"
        local = interval_concurrency(intervals_by_worker[worker])
        print(
            f"{worker:>6}  {pid_text:<8}  {assigned_by_worker[worker]:>8}  "
            f"{preads_by_worker[worker]:>6}  {success_by_worker[worker]:>2}  "
            f"{failed_by_worker[worker]:>6}  {len(threads_by_worker[worker]):>7}  "
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
        print(f"  average                : {sum(durations_us) / len(durations_us):.3f} us")
        print(f"  p50                    : {percentile(durations_us, 50):.3f} us")
        print(f"  p90                    : {percentile(durations_us, 90):.3f} us")
        print(f"  p99                    : {percentile(durations_us, 99):.3f} us")
        print(f"  maximum                : {durations_us[-1]:.3f} us")

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
        summarize_epoch(epoch, epochs[epoch], args.show_threads)

    if malformed:
        print(
            f"\nWarning: ignored {len(malformed)} malformed trace lines. "
            f"First issue: line {malformed[0][0]}: {malformed[0][1]}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
