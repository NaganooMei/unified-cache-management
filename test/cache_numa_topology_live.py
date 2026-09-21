#!/usr/bin/env python3
"""Inspect live accelerator topology and exercise Cache NUMA placement decisions.

This diagnostic intentionally avoids importing torch and vLLM.  It loads the
topology parsing helpers directly from ``ucm/integration/vllm/device.py``, reads
the current host topology, and reports the placement selected for these common
single-node layouts:

* GQA, DP1 TP8
* GQA, DP8 TP1
* MLA, DP1 TP8
* MLA, DP8 TP1

With ``--verify-pages`` it also creates small anonymous memfd mappings and uses
the same Linux mbind/touch/move_pages sequence as CacheStore to verify that the
requested NUMA nodes can really hold pages in the current container/cgroup.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import math
import mmap
import os
import platform
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
DEVICE_SOURCE = ROOT / "ucm/integration/vllm/device.py"
SCENARIOS = (
    ("GQA", 1, 8),
    ("GQA", 8, 1),
    ("MLA", 1, 8),
    ("MLA", 8, 1),
)


def _load_production_topology_helpers() -> Tuple[
    Callable[[str], List[int]],
    Callable[[str], Dict[int, List[int]]],
    Callable[[str, str, int], Optional[int]],
    Callable[[str], Optional[str]],
    Callable[[str], Optional[int]],
]:
    """Load the exact stdlib-only topology helpers used by Device implementations."""

    names = {
        "_expand_cpu_list",
        "_parse_npu_topo_affinity",
        "_parse_cpu_numa_map",
        "_resolve_npu_numa_node",
        "_normalize_cuda_pci_bus_id",
        "_resolve_cuda_numa_node",
    }
    tree = ast.parse(DEVICE_SOURCE.read_text(encoding="utf-8-sig"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    found = {node.name for node in functions}
    if found != names:
        missing = ", ".join(sorted(names - found))
        raise RuntimeError(f"cannot load production topology helpers: {missing}")
    namespace = {
        "os": os,
        "re": re,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
    }
    exec(
        compile(
            ast.Module(body=functions, type_ignores=[]), str(DEVICE_SOURCE), "exec"
        ),
        namespace,
    )
    return (
        namespace["_expand_cpu_list"],
        namespace["_parse_npu_topo_affinity"],
        namespace["_resolve_npu_numa_node"],
        namespace["_normalize_cuda_pci_bus_id"],
        namespace["_resolve_cuda_numa_node"],
    )


(
    EXPAND_LIST,
    PARSE_NPU_AFFINITY,
    RESOLVE_NPU_NUMA,
    NORMALIZE_CUDA_BDF,
    RESOLVE_CUDA_NUMA,
) = _load_production_topology_helpers()


@dataclass(frozen=True)
class WorkerPlacement:
    model: str
    dp_rank: int
    tp_rank: int
    device_ordinal: int
    device_id: int
    segment: str
    detected_node: Optional[int]
    owner_target: Tuple[int, ...]
    source: str
    valid: bool = True


def _run(command: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"failed to run {' '.join(command)}: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}: {detail}"
        )
    return result.stdout


def _read(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError as error:
        raise RuntimeError(f"failed to read {path}: {error}") from error


def _allowed_memory_nodes() -> List[int]:
    memory_nodes = EXPAND_LIST(_read("/sys/devices/system/node/has_memory"))
    status = _read("/proc/self/status")
    match = re.search(r"^Mems_allowed_list:\s*(\S+)", status, re.MULTILINE)
    if match is None:
        raise RuntimeError("cannot find Mems_allowed_list in /proc/self/status")
    allowed = set(EXPAND_LIST(match.group(1)))
    nodes = [node for node in memory_nodes if node in allowed]
    if not nodes:
        raise RuntimeError("no allowed NUMA nodes with memory")
    return nodes


def _parse_cuda_topo_affinity(output: str) -> Dict[int, List[int]]:
    """Parse the CPU Affinity column from ``nvidia-smi topo -m``."""
    lines = output.splitlines()
    header = next((line for line in lines if "CPU Affinity" in line), "")
    if not header:
        return {}
    topology_columns = len(header.split("CPU Affinity", 1)[0].split())
    affinity: Dict[int, List[int]] = {}
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        match = re.fullmatch(r"GPU(\d+)", parts[0])
        affinity_index = 1 + topology_columns
        if match is None or affinity_index >= len(parts):
            continue
        cpu_list = parts[affinity_index]
        if re.fullmatch(r"\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*", cpu_list) is None:
            continue
        affinity[int(match.group(1))] = EXPAND_LIST(cpu_list)
    return affinity


def _parse_cuda_gpu_bus_ids(output: str) -> Dict[int, str]:
    """Parse ``nvidia-smi --query-gpu=index,pci.bus_id`` output."""
    devices: Dict[int, str] = {}
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        bdf = NORMALIZE_CUDA_BDF(parts[1])
        if bdf is not None:
            devices[int(parts[0])] = bdf
    return devices


def _detect_platform(requested: str) -> str:
    if requested != "auto":
        return requested
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return "cuda"
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or os.environ.get(
        "ASCEND_VISIBLE_DEVICES"
    ):
        return "npu"
    for candidate, command in (("cuda", "nvidia-smi"), ("npu", "npu-smi")):
        try:
            result = subprocess.run(
                [command, "--help"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            return candidate
    raise RuntimeError("cannot detect CUDA or NPU platform; pass --platform explicitly")


def _discover_devices(
    platform_type: str,
    topo: str,
    explicit: Optional[str],
    cuda_bus_ids: Optional[Dict[int, str]] = None,
) -> List[int]:
    if explicit:
        devices = EXPAND_LIST(explicit)
    elif platform_type == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            try:
                devices = [
                    int(item.strip()) for item in visible.split(",") if item.strip()
                ]
            except ValueError as error:
                raise RuntimeError(
                    "CUDA_VISIBLE_DEVICES contains UUIDs; pass --devices with eight "
                    "physical GPU IDs"
                ) from error
        else:
            devices = sorted((cuda_bus_ids or {}).keys())
    else:
        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES") or os.environ.get(
            "ASCEND_VISIBLE_DEVICES"
        )
        if visible:
            devices = [int(item.strip()) for item in visible.split(",") if item.strip()]
        else:
            devices = sorted(
                {
                    int(match.group(1))
                    for line in topo.splitlines()
                    if (match := re.match(r"^\s*NPU(\d+)\b", line)) is not None
                }
            )
    if len(devices) != len(set(devices)):
        raise RuntimeError(f"duplicate device IDs in device list: {devices}")
    if len(devices) < 8:
        raise RuntimeError(
            f"the DP1TP8/DP8TP1 matrix needs 8 visible devices, found {devices}; "
            "pass --devices with eight physical IDs"
        )
    return devices[:8]


def _segment_nodes(
    nodes: Sequence[int], segments: int, segment: int
) -> Tuple[int, ...]:
    """Mirror ShmNuma::DataNodes/SegmentNodes for a shared Buffer."""

    if not nodes:
        return ()
    if segments == 1:
        return (nodes[0],)
    groups = math.gcd(segments, len(nodes))
    per_group = len(nodes) // groups
    first = (segment % groups) * per_group
    return tuple(nodes[first : first + per_group])


def _worker_placements(
    model: str,
    dp_size: int,
    tp_size: int,
    devices: Sequence[int],
    detected: Dict[int, Optional[int]],
    allowed_nodes: Sequence[int],
    platform_type: str = "npu",
) -> List[WorkerPlacement]:
    placements: List[WorkerPlacement] = []
    for worker in range(dp_size * tp_size):
        dp_rank, tp_rank = divmod(worker, tp_size)
        device_id = devices[worker]
        detected_node = detected.get(device_id)
        valid = detected_node is None or detected_node in allowed_nodes
        if model == "GQA":
            segment = f"dp{dp_rank}-segment-{tp_rank}"
            if detected_node is not None:
                targets = (detected_node,)
                source = f"{platform_type.upper()} topology"
            elif platform_type == "npu":
                # Each GQA DP owns its TP segments. The fallback rank spreads
                # those independent segments across host-local NUMA nodes.
                fallback_rank = worker
                targets = (allowed_nodes[fallback_rank % len(allowed_nodes)],)
                source = f"local-worker fallback({fallback_rank})"
            else:
                targets = _segment_nodes(allowed_nodes, tp_size, tp_rank)
                source = "segment striping" if targets else "first-touch"
        else:
            segment = f"shared-segment-{tp_rank}"
            if detected_node is not None:
                targets = (detected_node,)
                source = f"{platform_type.upper()} topology if this DP owns segment"
            else:
                targets = _segment_nodes(allowed_nodes, tp_size, tp_rank)
                source = "segment striping" if targets else "first-touch"
        placements.append(
            WorkerPlacement(
                model=model,
                dp_rank=dp_rank,
                tp_rank=tp_rank,
                device_ordinal=worker,
                device_id=device_id,
                segment=segment,
                detected_node=detected_node,
                owner_target=targets,
                source=source,
                valid=valid,
            )
        )
    return placements


def _target_text(placement: WorkerPlacement) -> str:
    if not placement.valid:
        return f"INVALID:{placement.detected_node}"
    if not placement.owner_target:
        return "first-touch"
    return ",".join(str(node) for node in placement.owner_target)


def _print_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    text_rows = [[str(value) for value in row] for row in rows]
    widths = [len(value) for value in headers]
    for row in text_rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    fmt = "  ".join(f"{{:<{width}}}" for width in widths)
    print(fmt.format(*headers))
    print(fmt.format(*("-" * width for width in widths)))
    for row in text_rows:
        print(fmt.format(*row))


def _print_scenario(
    model: str,
    dp_size: int,
    tp_size: int,
    placements: Sequence[WorkerPlacement],
) -> Tuple[List[int], List[str]]:
    print(f"\n== {model} DP{dp_size} TP{tp_size} ==")
    _print_table(
        ("DP", "TP", "ordinal", "device", "topo", "buffer/segment", "target", "source"),
        (
            (
                item.dp_rank,
                item.tp_rank,
                item.device_ordinal,
                item.device_id,
                item.detected_node if item.detected_node is not None else "none",
                item.segment,
                _target_text(item),
                item.source,
            )
            for item in placements
        ),
    )

    verify_nodes: List[int] = []
    warnings: List[str] = []
    if model == "GQA":
        counts = Counter(
            node for item in placements if item.valid for node in item.owner_target
        )
        print(f"GQA DP-scoped segment distribution: {dict(sorted(counts.items()))}")
        verify_nodes.extend(counts)
    else:
        by_segment: Dict[str, List[WorkerPlacement]] = {}
        for item in placements:
            by_segment.setdefault(item.segment, []).append(item)
        summary_rows = []
        for segment, candidates in by_segment.items():
            choices = {_target_text(item) for item in candidates}
            candidate_devices = ",".join(str(item.device_id) for item in candidates)
            if len(choices) == 1:
                result = next(iter(choices))
                ownership = "deterministic placement"
            else:
                result = " | ".join(sorted(choices))
                ownership = "depends on DP owner race"
                warnings.append(
                    f"{segment} has different owner targets ({result}); the DP process "
                    "that creates the segment first determines its placement"
                )
            summary_rows.append((segment, candidate_devices, result, ownership))
            for item in candidates:
                if item.valid:
                    verify_nodes.extend(item.owner_target)
        _print_table(
            ("shared segment", "candidate devices", "possible target", "ownership"),
            summary_rows,
        )
    for item in placements:
        if not item.valid:
            warnings.append(
                f"device {item.device_id} resolves to NUMA {item.detected_node}, which "
                f"is outside the process Mems_allowed_list"
            )
    for warning in warnings:
        print(f"WARNING: {warning}")
    return verify_nodes, warnings


def _syscall_numbers() -> Tuple[int, int]:
    machine = platform.machine().lower()
    if machine in {"aarch64", "arm64"}:
        return 235, 239
    if machine in {"x86_64", "amd64"}:
        return 237, 279
    raise RuntimeError(f"unsupported architecture for NUMA syscalls: {machine}")


def _check_syscall(result: int, operation: str) -> None:
    if result >= 0:
        return
    error = ctypes.get_errno()
    raise RuntimeError(f"{operation} failed: {os.strerror(error)} (errno={error})")


def _verify_memfd_node(node: int, size: int) -> Dict[int, int]:
    if not hasattr(os, "memfd_create"):
        raise RuntimeError("Python does not expose os.memfd_create on this system")
    mbind_nr, move_pages_nr = _syscall_numbers()
    page_size = mmap.PAGESIZE
    size = ((size + page_size - 1) // page_size) * page_size
    fd = os.memfd_create(f"ucm_numa_probe_{node}", os.MFD_CLOEXEC)
    mapping: Optional[mmap.mmap] = None
    view = None
    try:
        os.ftruncate(fd, size)
        mapping = mmap.mmap(
            fd,
            size,
            flags=mmap.MAP_SHARED,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        view = (ctypes.c_char * size).from_buffer(mapping)
        address = ctypes.addressof(view)
        bits = ctypes.sizeof(ctypes.c_ulong) * 8
        word_count = node // bits + 1
        mask = (ctypes.c_ulong * word_count)()
        mask[node // bits] = 1 << (node % bits)
        max_node = word_count * bits + 1
        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall.restype = ctypes.c_long
        # linux/mempolicy.h: MPOL_BIND=2, MPOL_F_STATIC_NODES=(1 << 15).
        result = libc.syscall(
            ctypes.c_long(mbind_nr),
            ctypes.c_void_p(address),
            ctypes.c_ulong(size),
            ctypes.c_int(2 | (1 << 15)),
            ctypes.cast(mask, ctypes.c_void_p),
            ctypes.c_ulong(max_node),
            ctypes.c_uint(0),
        )
        _check_syscall(result, f"mbind node={node}")
        ctypes.memset(address, 0xA5, size)

        counts: Counter[int] = Counter()
        page_count = size // page_size
        batch = 4096
        for first in range(0, page_count, batch):
            count = min(batch, page_count - first)
            pages = (ctypes.c_void_p * count)(
                *(address + (first + index) * page_size for index in range(count))
            )
            status = (ctypes.c_int * count)(*([-1] * count))
            result = libc.syscall(
                ctypes.c_long(move_pages_nr),
                ctypes.c_int(0),
                ctypes.c_ulong(count),
                pages,
                ctypes.c_void_p(0),
                status,
                ctypes.c_int(0),
            )
            _check_syscall(result, f"move_pages query node={node}")
            for actual in status:
                if actual < 0:
                    raise RuntimeError(
                        f"move_pages page status failed: {os.strerror(-actual)} "
                        f"(status={actual})"
                    )
                counts[actual] += 1
        if set(counts) != {node}:
            raise RuntimeError(
                f"NUMA placement mismatch: expected node {node}, actual {dict(counts)}"
            )
        return dict(sorted(counts.items()))
    finally:
        # ctypes.from_buffer keeps an exported pointer until the view is deleted.
        if view is not None:
            del view
        if mapping is not None:
            mapping.close()
        os.close(fd)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--platform",
        choices=("auto", "cuda", "npu"),
        default="auto",
        help="accelerator platform (default: auto-detect)",
    )
    parser.add_argument(
        "--devices",
        help=(
            "eight physical device IDs, for example 0-7 or 0,1,2,3,4,5,6,7; "
            "defaults to the platform visible-device variable, then topology rows"
        ),
    )
    parser.add_argument(
        "--verify-pages",
        action="store_true",
        help="verify real memfd page placement with mbind and move_pages",
    )
    parser.add_argument(
        "--verify-mib-per-node",
        type=int,
        default=4,
        help="MiB allocated for each verified NUMA node (default: 4)",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="print raw accelerator topology and lscpu output",
    )
    parser.add_argument(
        "--fail-on-warning",
        action="store_true",
        help="return a non-zero status if the matrix reports concentration or owner races",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.verify_mib_per_node <= 0:
        raise RuntimeError("--verify-mib-per-node must be positive")

    platform_type = _detect_platform(args.platform)
    cuda_bus_ids: Dict[int, str] = {}
    if platform_type == "cuda":
        topo_command = ["nvidia-smi", "topo", "-m"]
        topo = _run(topo_command)
        cuda_bus_ids = _parse_cuda_gpu_bus_ids(
            _run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,pci.bus_id",
                    "--format=csv,noheader,nounits",
                ]
            )
        )
        affinity = _parse_cuda_topo_affinity(topo)
    else:
        topo_command = ["npu-smi", "info", "-t", "topo"]
        topo = _run(topo_command)
        affinity = PARSE_NPU_AFFINITY(topo)
    cpu_numa = _run(["lscpu", "-e=cpu,node"])
    devices = _discover_devices(platform_type, topo, args.devices, cuda_bus_ids)
    allowed_nodes = _allowed_memory_nodes()
    if platform_type == "cuda":
        missing = [device for device in devices if device not in cuda_bus_ids]
        if missing:
            raise RuntimeError(f"nvidia-smi did not report PCI IDs for GPUs {missing}")
        detected = {
            device: RESOLVE_CUDA_NUMA(cuda_bus_ids[device]) for device in devices
        }
    else:
        detected = {
            device: RESOLVE_NPU_NUMA(topo, cpu_numa, device) for device in devices
        }

    print("== Live host topology ==")
    print(f"platform: {platform_type}")
    print(f"architecture: {platform.machine()}")
    print(f"selected physical devices: {devices}")
    print(f"allowed memory NUMA nodes: {allowed_nodes}")
    print(f"topology exposes CPU Affinity: {bool(affinity)}")
    _print_table(
        ("ordinal", "device", "CPU affinity", "resolved NUMA"),
        (
            (
                ordinal,
                device,
                ",".join(str(cpu) for cpu in affinity.get(device, [])) or "none",
                detected[device] if detected[device] is not None else "none",
            )
            for ordinal, device in enumerate(devices)
        ),
    )
    if args.show_raw:
        print(f"\n-- {' '.join(topo_command)} --")
        print(topo.rstrip())
        print("\n-- lscpu -e=cpu,node --")
        print(cpu_numa.rstrip())

    verify_nodes: List[int] = []
    warnings: List[str] = []
    for model, dp_size, tp_size in SCENARIOS:
        placements = _worker_placements(
            model,
            dp_size,
            tp_size,
            devices,
            detected,
            allowed_nodes,
            platform_type,
        )
        scenario_nodes, scenario_warnings = _print_scenario(
            model, dp_size, tp_size, placements
        )
        verify_nodes.extend(scenario_nodes)
        warnings.extend(scenario_warnings)

    if args.verify_pages:
        print("\n== Live memfd mbind/move_pages verification ==")
        unique_nodes = sorted(set(verify_nodes))
        if not unique_nodes:
            print("SKIP: no deterministic NUMA targets were selected")
        for node in unique_nodes:
            if node not in allowed_nodes:
                print(f"FAIL node {node}: outside Mems_allowed_list")
                return 1
            try:
                counts = _verify_memfd_node(
                    node, args.verify_mib_per_node * 1024 * 1024
                )
            except RuntimeError as error:
                print(f"FAIL node {node}: {error}")
                return 1
            print(
                f"PASS node {node}: {args.verify_mib_per_node} MiB, "
                f"page distribution={counts}"
            )

    print(f"\nCompleted with {len(warnings)} warning(s).")
    return 1 if args.fail_on_warning and warnings else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
