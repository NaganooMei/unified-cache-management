#!/usr/bin/env python3

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

TRACE_MARKER = "COPY_START_TRACE "


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate host-process and device-stream start skew from toolkit logs."
    )
    parser.add_argument("log", type=Path)
    parser.add_argument("--expected-devices", type=int, required=True)
    parser.add_argument("--expected-iterations", type=int, required=True)
    parser.add_argument("--expected-start-gate")
    parser.add_argument("--max-barrier-exit-skew-us", type=float)
    parser.add_argument("--max-notify-submit-skew-us", type=float)
    parser.add_argument("--max-stream-start-skew-us", type=float)
    parser.add_argument("--require-timing-breakdown", action="store_true")
    return parser.parse_args()


def parse_trace_line(line, line_number):
    marker = line.find(TRACE_MARKER)
    if marker < 0:
        return None

    fields = {}
    for token in line[marker + len(TRACE_MARKER) :].strip().split():
        if "=" not in token:
            raise ValueError(f"line {line_number}: invalid token {token!r}")
        key, value = token.split("=", 1)
        fields[key] = value

    required = {
        "method",
        "device",
        "iteration",
        "barrier_exit_ns",
        "notify_submit_ns",
        "stream_start_skew_us",
    }
    missing = sorted(required - fields.keys())
    if missing:
        raise ValueError(f"line {line_number}: missing fields: {', '.join(missing)}")

    record = {
        "start_gate": fields.get("start_gate"),
        "method": fields["method"],
        "device": int(fields["device"]),
        "iteration": int(fields["iteration"]),
        "barrier_exit_ns": int(fields["barrier_exit_ns"]),
        "notify_submit_ns": int(fields["notify_submit_ns"]),
        "stream_start_skew_us": int(fields["stream_start_skew_us"]),
    }
    timing_fields = {
        "wall_start_ns",
        "release_submit_ns",
        "sync_enter_ns",
        "wall_end_ns",
        "device_gate_us",
        "device_copy_us",
    }
    timing_present = timing_fields & fields.keys()
    if timing_present and timing_present != timing_fields:
        missing_timing = sorted(timing_fields - fields.keys())
        raise ValueError(
            f"line {line_number}: incomplete timing breakdown: "
            f"{', '.join(missing_timing)}"
        )
    if timing_present:
        for key in timing_fields:
            record[key] = int(fields[key])
        if not (
            record["wall_start_ns"]
            <= record["release_submit_ns"]
            <= record["sync_enter_ns"]
            <= record["wall_end_ns"]
        ):
            raise ValueError(f"line {line_number}: invalid host timing order")
    return record


def percentile(values, percent):
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percent / 100) - 1)
    return ordered[index]


def average(values):
    return sum(values) / len(values)


def main():
    args = parse_args()
    groups = defaultdict(list)
    seen = set()

    with args.log.open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            record = parse_trace_line(line, line_number)
            if record is None:
                continue
            identity = (record["method"], record["iteration"], record["device"])
            if identity in seen:
                raise ValueError(
                    "duplicate trace record for "
                    f"method={identity[0]} iteration={identity[1]} device={identity[2]}"
                )
            seen.add(identity)
            if (
                args.expected_start_gate is not None
                and record["start_gate"] != args.expected_start_gate
            ):
                actual = record["start_gate"] or "missing"
                raise ValueError(
                    f"line {line_number}: expected start_gate="
                    f"{args.expected_start_gate}, found {actual}; rebuild dev-sandbox"
                )
            groups[(record["method"], record["iteration"])].append(record)

    if not groups:
        print(
            f"FAIL: no {TRACE_MARKER.strip()} records found in {args.log}",
            file=sys.stderr,
        )
        return 1


    timing_records = [
        record
        for records in groups.values()
        for record in records
        if "wall_start_ns" in record
    ]
    total_records = sum(len(records) for records in groups.values())
    if timing_records and len(timing_records) != total_records:
        raise ValueError("timing breakdown is present for only part of the trace records")
    if args.require_timing_breakdown and not timing_records:
        raise ValueError("timing breakdown is required but missing; rebuild dev-sandbox")

    failures = []
    methods = sorted({method for method, _ in groups})
    for method in methods:
        for iteration in range(args.expected_iterations):
            if (method, iteration) not in groups:
                failures.append(f"missing method={method} iteration={iteration}")

    print(
        "method iteration devices barrier_exit_skew_us "
        "notify_submit_skew_us max_stream_start_skew_us p50_stream_start_skew_us"
    )
    all_barrier_skews = []
    all_notify_skews = []
    all_stream_skews = []
    for (method, iteration), records in sorted(groups.items()):
        devices = {record["device"] for record in records}
        if (
            len(records) != args.expected_devices
            or len(devices) != args.expected_devices
        ):
            failures.append(
                f"method={method} iteration={iteration}: expected {args.expected_devices} "
                f"devices, found records={len(records)} unique_devices={len(devices)}"
            )

        barrier_values = [record["barrier_exit_ns"] for record in records]
        notify_values = [record["notify_submit_ns"] for record in records]
        stream_values = [record["stream_start_skew_us"] for record in records]
        barrier_skew = (max(barrier_values) - min(barrier_values)) / 1000
        notify_skew = (max(notify_values) - min(notify_values)) / 1000
        max_stream_skew = max(stream_values)
        p50_stream_skew = percentile(stream_values, 50)
        all_barrier_skews.append(barrier_skew)
        all_notify_skews.append(notify_skew)
        all_stream_skews.extend(stream_values)
        print(
            f"{method} {iteration} {len(devices)} {barrier_skew:.3f} "
            f"{notify_skew:.3f} {max_stream_skew} {p50_stream_skew}"
        )

    max_barrier_skew = max(all_barrier_skews)
    max_notify_skew = max(all_notify_skews)
    max_stream_skew = max(all_stream_skews)
    print(
        "summary "
        f"groups={len(groups)} max_barrier_exit_skew_us={max_barrier_skew:.3f} "
        f"max_notify_submit_skew_us={max_notify_skew:.3f} "
        f"p50_stream_start_skew_us={percentile(all_stream_skews, 50)} "
        f"p95_stream_start_skew_us={percentile(all_stream_skews, 95)} "
        f"max_stream_start_skew_us={max_stream_skew}"
    )
    if timing_records:
        host_release_submit = [
            (record["release_submit_ns"] - record["wall_start_ns"]) / 1000
            for record in timing_records
        ]
        host_control_submit = [
            (record["sync_enter_ns"] - record["release_submit_ns"]) / 1000
            for record in timing_records
        ]
        host_sync_wait = [
            (record["wall_end_ns"] - record["sync_enter_ns"]) / 1000
            for record in timing_records
        ]
        process_wall = [
            (record["wall_end_ns"] - record["wall_start_ns"]) / 1000
            for record in timing_records
        ]
        device_gate = [record["device_gate_us"] for record in timing_records]
        device_copy = [record["device_copy_us"] for record in timing_records]
        wall_minus_device = [
            wall - gate - copy
            for wall, gate, copy in zip(process_wall, device_gate, device_copy)
        ]
        metrics = (
            ("host_release_submit", host_release_submit),
            ("host_control_submit", host_control_submit),
            ("host_sync_wait", host_sync_wait),
            ("device_gate", device_gate),
            ("device_copy", device_copy),
            ("process_wall", process_wall),
            ("wall_minus_device", wall_minus_device),
        )
        output = ["timing_summary", f"samples={len(timing_records)}"]
        for name, values in metrics:
            output.append(f"{name}_avg_us={average(values):.3f}")
            output.append(f"{name}_p95_us={percentile(values, 95):.3f}")
        print(" ".join(output))

    thresholds = (
        (
            "barrier exit skew",
            max_barrier_skew,
            args.max_barrier_exit_skew_us,
        ),
        (
            "notify submit skew",
            max_notify_skew,
            args.max_notify_submit_skew_us,
        ),
        (
            "stream start skew",
            max_stream_skew,
            args.max_stream_start_skew_us,
        ),
    )
    for label, actual, limit in thresholds:
        if limit is not None and actual > limit:
            failures.append(f"{label} {actual:.3f} us exceeds limit {limit:.3f} us")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    print(
        "PASS: all expected trace records are present and all configured limits are met"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
