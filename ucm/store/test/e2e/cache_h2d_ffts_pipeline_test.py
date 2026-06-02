# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
import os
import secrets
import statistics
import time
from dataclasses import dataclass

import numpy as np
import torch

from ucm.store.pipeline.connector import UcmPipelineStore


@dataclass
class LoadPerfResult:
    case_name: str
    transport: str
    block_num: int
    fragment_count: int
    shard_bytes: int
    object_target_bytes: int
    objects_per_shard: int
    max_object_bytes: int
    max_object_fragments: int
    bytes_per_load: int
    avg_seconds: float
    median_seconds: float
    min_seconds: float
    gbps: float


@dataclass(frozen=True)
class ModelCase:
    name: str
    tensor_sizes: list[int]


KIB = 1024
MIB = 1024 * KIB
GIB = 1024 * MIB
QWEN32B_FRAGMENT_COUNT = 128
MODEL_CASES = {
    "qwen32b_tp8": ModelCase("qwen32b_tp8", [32 * KIB] * QWEN32B_FRAGMENT_COUNT),
    "qwen32b_tp8_full": ModelCase(
        "qwen32b_tp8_full", [32 * KIB] * QWEN32B_FRAGMENT_COUNT
    ),
    "qwen32b_tp8_2m": ModelCase("qwen32b_tp8_2m", [32 * KIB] * 64),
    "qwen32b_tp8_1m": ModelCase("qwen32b_tp8_1m", [32 * KIB] * 32),
    "qwen32b_tp4": ModelCase("qwen32b_tp4", [64 * KIB] * QWEN32B_FRAGMENT_COUNT),
    "qwen32b_tp4_full": ModelCase(
        "qwen32b_tp4_full", [64 * KIB] * QWEN32B_FRAGMENT_COUNT
    ),
    "qwen32b_tp4_2m": ModelCase("qwen32b_tp4_2m", [64 * KIB] * 32),
    "qwen32b_tp4_1m": ModelCase("qwen32b_tp4_1m", [64 * KIB] * 16),
}


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "y", "on")


def parse_tensor_sizes() -> list[int]:
    sizes = os.getenv("UCM_FFTS_TENSOR_SIZES")
    if sizes:
        return [int(item.strip()) for item in sizes.split(",") if item.strip()]
    fragment_count = env_int("UCM_FFTS_FRAGMENT_COUNT", 128)
    fragment_bytes = env_int("UCM_FFTS_FRAGMENT_BYTES", 32768)
    return [fragment_bytes] * fragment_count


def parse_model_case() -> ModelCase:
    case_name = os.getenv("UCM_FFTS_MODEL_CASE")
    if not case_name:
        return ModelCase("custom", parse_tensor_sizes())

    name = case_name.strip().lower()
    if "," in name:
        raise ValueError("UCM_FFTS_MODEL_CASE only accepts one case")
    case = MODEL_CASES.get(name)
    if case is None:
        valid = sorted(MODEL_CASES)
        raise ValueError(f"unknown UCM_FFTS_MODEL_CASE {name}; valid={valid}")
    return case


def format_bytes(value: int) -> str:
    if value == 0:
        return "0B"
    if value % GIB == 0:
        return f"{value // GIB}GiB"
    if value % MIB == 0:
        return f"{value // MIB}MiB"
    if value % KIB == 0:
        return f"{value // KIB}KiB"
    return f"{value}B"


def tensor_size_histogram(tensor_sizes: list[int]) -> str:
    counts: dict[int, int] = {}
    for size in tensor_sizes:
        counts[size] = counts.get(size, 0) + 1
    return ", ".join(f"{count}x{format_bytes(size)}" for size, count in sorted(counts.items()))


def build_object_plan(tensor_sizes: list[int], target_bytes: int) -> list[list[int]]:
    if not tensor_sizes:
        return []
    if target_bytes <= 0:
        return [tensor_sizes]

    plan: list[list[int]] = []
    current: list[int] = []
    current_bytes = 0
    for size in tensor_sizes:
        exceeds_target = current_bytes >= target_bytes or size > target_bytes - current_bytes
        if current and exceeds_target:
            plan.append(current)
            current = []
            current_bytes = 0
        current.append(size)
        current_bytes += size
    if current:
        plan.append(current)
    return plan


def object_plan_stats(tensor_sizes: list[int], target_bytes: int) -> tuple[int, int, int]:
    plan = build_object_plan(tensor_sizes, target_bytes)
    if not plan:
        return 0, 0, 0
    return (
        len(plan),
        max(sum(item) for item in plan),
        max(len(item) for item in plan),
    )


def cache_buffer_capacity_gb_for_case(tensor_sizes: list[int]) -> int:
    configured = os.getenv("UCM_FFTS_CACHE_BUFFER_CAPACITY_GB")
    if configured is not None:
        return int(configured)
    min_bytes = sum(tensor_sizes) * 1024
    return max(4, (min_bytes + GIB - 1) // GIB)


def torch_device(device_type: str, device_id: int) -> str:
    return f"{device_type}:{device_id}"


def prepare_torch_backend(device_type: str) -> None:
    if device_type != "npu" or hasattr(torch, "npu"):
        return
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "UCM_FFTS_TORCH_DEVICE=npu requires torch_npu to be importable"
        ) from exc


def synchronize_device(device_type: str, device_id: int) -> None:
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device_id)
        return
    if device_type == "npu" and hasattr(torch, "npu"):
        try:
            torch.npu.synchronize(device_id)
        except TypeError:
            torch.npu.synchronize()


def make_tensors(
    block_num: int, tensor_sizes: list[int], device: str
) -> list[list[torch.Tensor]]:
    tensors = []
    element_size = torch.tensor([], dtype=torch.bfloat16).element_size()
    if any(tensor_size <= 0 or tensor_size % element_size != 0 for tensor_size in tensor_sizes):
        raise ValueError(f"invalid tensor byte sizes for bfloat16: {tensor_sizes}")
    for _ in range(block_num):
        tensors.append(
            [
                torch.rand(
                    [tensor_size // element_size],
                    dtype=torch.bfloat16,
                    device=device,
                )
                for tensor_size in tensor_sizes
            ]
        )
    return tensors


def make_empty_like(tensors: list[list[torch.Tensor]]) -> list[list[torch.Tensor]]:
    return [[torch.empty_like(tensor) for tensor in row] for row in tensors]


def poison_tensors(
    tensors: list[list[torch.Tensor]],
    device_type: str,
    device_id: int,
) -> None:
    with torch.no_grad():
        for row in tensors:
            for tensor in row:
                tensor.fill_(-1)
    synchronize_device(device_type, device_id)


def tensor_ptrs(tensors: list[list[torch.Tensor]]) -> np.ndarray:
    return np.asarray(
        [[tensor.data_ptr() for tensor in row] for row in tensors], dtype=np.uint64
    )


def cmp_and_print_diff(a, b, rtol=0.0, atol=0.0):
    for r, (row_a, row_b) in enumerate(zip(a, b)):
        for c, (ta, tb) in enumerate(zip(row_a, row_b)):
            if not torch.allclose(ta, tb, rtol=rtol, atol=atol):
                mask = ~torch.isclose(ta, tb, rtol=rtol, atol=atol)
                diff_a = ta[mask].cpu()
                diff_b = tb[mask].cpu()
                print(f"DIFF at [{r}][{c}]  total {mask.sum().item()} element(s)")
                print("  a val:", diff_a.flatten())
                print("  b val:", diff_b.flatten())
                assert False


def build_config(
    unique_id: str,
    transport: str,
    tensor_sizes: list[int],
    cache_buffer_capacity_gb: int,
    object_target_bytes: int,
) -> dict:
    shard_size = sum(tensor_sizes)
    return {
        "store_pipeline": "Cache|Empty",
        "unique_id": unique_id,
        "timeout_ms": env_int("UCM_FFTS_TIMEOUT_MS", 30000),
        "tensor_size_list": tensor_sizes,
        "shard_size": shard_size,
        "block_size": shard_size,
        "share_buffer_enable": env_bool("UCM_FFTS_SHARE_BUFFER_ENABLE", True),
        "cache_buffer_capacity_gb": cache_buffer_capacity_gb,
        "cache_load_exclusive_buffer_number": env_int(
            "UCM_FFTS_LOAD_EXCLUSIVE_BUFFER_NUMBER", 64
        ),
        "waiting_queue_depth": env_int("UCM_FFTS_WAITING_QUEUE_DEPTH", 64),
        "running_queue_depth": env_int("UCM_FFTS_RUNNING_QUEUE_DEPTH", 4096),
        "cache_stream_number": env_int("UCM_FFTS_CACHE_STREAM_NUMBER", 4),
        "cache_h2d_transport": transport,
        "cache_h2d_ffts_pipeline_depth": env_int("UCM_FFTS_PIPELINE_DEPTH", 2),
        "cache_h2d_ffts_max_ready_lanes": env_int("UCM_FFTS_MAX_READY_LANES", 8),
        "cache_h2d_ffts_object_target_bytes": object_target_bytes,
    }


def prepare_cache(
    worker: UcmPipelineStore,
    lookup_store: UcmPipelineStore,
    block_ids: list[bytes],
    shard_indexes: list[int],
    src_ptrs: np.ndarray,
) -> None:
    assert not any(lookup_store.lookup(block_ids))
    assert lookup_store.lookup_on_prefix(block_ids) == -1
    task = worker.dump_data(block_ids, shard_indexes, src_ptrs)
    worker.wait(task)
    assert all(lookup_store.lookup(block_ids))
    assert lookup_store.lookup_on_prefix(block_ids) + 1 == len(block_ids)


def time_load(
    worker: UcmPipelineStore,
    block_ids: list[bytes],
    shard_indexes: list[int],
    dst_ptrs: np.ndarray,
) -> float:
    start = time.perf_counter()
    task = worker.load_data(block_ids, shard_indexes, dst_ptrs)
    worker.wait(task)
    return time.perf_counter() - start


def validate_load(
    worker: UcmPipelineStore,
    block_ids: list[bytes],
    shard_indexes: list[int],
    src_tensors: list[list[torch.Tensor]],
    dst_tensors: list[list[torch.Tensor]],
    dst_ptrs: np.ndarray,
    device_type: str,
    device_id: int,
) -> None:
    poison_tensors(dst_tensors, device_type, device_id)
    time_load(worker, block_ids, shard_indexes, dst_ptrs)
    cmp_and_print_diff(src_tensors, dst_tensors)


def run_transport(
    case_name: str,
    transport: str,
    tensor_sizes: list[int],
    block_num: int,
    device_type: str,
    device_id: int,
    warmup: int,
    repeat: int,
    cache_buffer_capacity_gb: int,
    object_target_bytes: int,
) -> LoadPerfResult:
    device = torch_device(device_type, device_id)
    unique_id = f"h2d-ffts-{case_name}-{transport}-{secrets.token_hex(8)}"
    config = build_config(
        unique_id,
        transport,
        tensor_sizes,
        cache_buffer_capacity_gb,
        object_target_bytes,
    )
    worker = UcmPipelineStore(config | {"device_id": device_id})
    share_buffer_enable = bool(config["share_buffer_enable"])
    lookup_store = UcmPipelineStore(config) if share_buffer_enable else worker

    block_ids = [secrets.token_bytes(16) for _ in range(block_num)]
    shard_indexes = [0 for _ in range(block_num)]
    src_tensors = make_tensors(block_num, tensor_sizes, device)
    dst_tensors = make_empty_like(src_tensors)
    synchronize_device(device_type, device_id)

    src_ptrs = tensor_ptrs(src_tensors)
    dst_ptrs = tensor_ptrs(dst_tensors)
    prepare_cache(worker, lookup_store, block_ids, shard_indexes, src_ptrs)

    validate_load(
        worker,
        block_ids,
        shard_indexes,
        src_tensors,
        dst_tensors,
        dst_ptrs,
        device_type,
        device_id,
    )

    for _ in range(warmup):
        time_load(worker, block_ids, shard_indexes, dst_ptrs)

    samples = [
        time_load(worker, block_ids, shard_indexes, dst_ptrs) for _ in range(repeat)
    ]
    validate_load(
        worker,
        block_ids,
        shard_indexes,
        src_tensors,
        dst_tensors,
        dst_ptrs,
        device_type,
        device_id,
    )

    bytes_per_load = block_num * sum(tensor_sizes)
    avg_seconds = sum(samples) / len(samples)
    objects_per_shard, max_object_bytes, max_object_fragments = object_plan_stats(
        tensor_sizes, object_target_bytes
    )
    if transport != "ffts_pipeline":
        objects_per_shard = 0
        max_object_bytes = 0
        max_object_fragments = 0
    return LoadPerfResult(
        case_name=case_name,
        transport=transport,
        block_num=block_num,
        fragment_count=len(tensor_sizes),
        shard_bytes=sum(tensor_sizes),
        object_target_bytes=object_target_bytes,
        objects_per_shard=objects_per_shard,
        max_object_bytes=max_object_bytes,
        max_object_fragments=max_object_fragments,
        bytes_per_load=bytes_per_load,
        avg_seconds=avg_seconds,
        median_seconds=statistics.median(samples),
        min_seconds=min(samples),
        gbps=bytes_per_load / avg_seconds / 1e9,
    )


def print_result(result: LoadPerfResult) -> None:
    object_info = ""
    if result.transport == "ffts_pipeline":
        object_info = (
            f"object_target={format_bytes(result.object_target_bytes)}, "
            f"objects_per_shard={result.objects_per_shard}, "
            f"max_object={format_bytes(result.max_object_bytes)}, "
            f"max_object_fragments={result.max_object_fragments}, "
        )
    print(
        f"{result.case_name}/{result.transport}: "
        f"blocks={result.block_num}, fragments={result.fragment_count}, "
        f"shard={format_bytes(result.shard_bytes)}, "
        f"{object_info}"
        f"bytes={result.bytes_per_load}, "
        f"avg={result.avg_seconds * 1e3:.3f}ms, "
        f"median={result.median_seconds * 1e3:.3f}ms, "
        f"min={result.min_seconds * 1e3:.3f}ms, "
        f"bw={result.gbps:.3f}GB/s"
    )


def print_case_config(
    case: ModelCase,
    block_num: int,
    warmup: int,
    repeat: int,
    device_type: str,
    device_id: int,
    cache_buffer_capacity_gb: int,
    object_target_bytes: int,
    transport: str = "ffts_pipeline",
) -> None:
    shard_bytes = sum(case.tensor_sizes)
    object_info = ""
    if transport == "ffts_pipeline":
        objects_per_shard, max_object_bytes, max_object_fragments = object_plan_stats(
            case.tensor_sizes, object_target_bytes
        )
        object_info = (
            f"object_target={format_bytes(object_target_bytes)}, "
            f"objects_per_shard={objects_per_shard}, "
            f"max_object={format_bytes(max_object_bytes)}, "
            f"max_object_fragments={max_object_fragments}, "
        )
    print(
        "case_config: "
        f"case={case.name}, device={device_type}:{device_id}, blocks={block_num}, "
        f"fragments={len(case.tensor_sizes)}, shard={format_bytes(shard_bytes)}, "
        f"{object_info}"
        f"tensor_sizes=[{tensor_size_histogram(case.tensor_sizes)}], "
        f"cache_buffer_capacity_gb={cache_buffer_capacity_gb}, "
        f"warmup={warmup}, repeat={repeat}"
    )


def print_summary_table(results: list[LoadPerfResult]) -> None:
    if not results:
        return
    print(
        "summary: "
        "case,transport,blocks,fragments,shard,object_target,objects_per_shard,"
        "max_object,max_object_fragments,bytes,avg_ms,median_ms,min_ms,gbps"
    )
    for result in results:
        objects_per_shard = (
            str(result.objects_per_shard) if result.transport == "ffts_pipeline" else ""
        )
        max_object = (
            format_bytes(result.max_object_bytes) if result.transport == "ffts_pipeline" else ""
        )
        max_object_fragments = (
            str(result.max_object_fragments) if result.transport == "ffts_pipeline" else ""
        )
        print(
            "summary: "
            f"{result.case_name},{result.transport},{result.block_num},"
            f"{result.fragment_count},{format_bytes(result.shard_bytes)},"
            f"{format_bytes(result.object_target_bytes)},{objects_per_shard},"
            f"{max_object},{max_object_fragments},{result.bytes_per_load},"
            f"{result.avg_seconds * 1e3:.3f},"
            f"{result.median_seconds * 1e3:.3f},{result.min_seconds * 1e3:.3f},"
            f"{result.gbps:.3f}"
        )


def main():
    os.environ.setdefault("UC_LOGGER_LEVEL", "info")

    case = parse_model_case()
    block_num = env_int("UCM_FFTS_BLOCK_NUM", 16)
    warmup = env_int("UCM_FFTS_WARMUP", 2)
    repeat = env_int("UCM_FFTS_REPEAT", 10)
    if block_num <= 0 or warmup < 0 or repeat <= 0:
        raise ValueError("UCM_FFTS_BLOCK_NUM and UCM_FFTS_REPEAT must be positive")
    device_type = os.getenv("UCM_FFTS_TORCH_DEVICE", "cuda")
    device_id = env_int("UCM_FFTS_DEVICE_ID", 0)
    prepare_torch_backend(device_type)
    min_gbps = env_float("UCM_FFTS_MIN_GBPS", 0.0)
    object_target_bytes = env_int("UCM_FFTS_OBJECT_TARGET_BYTES", 0)

    cache_buffer_capacity_gb = cache_buffer_capacity_gb_for_case(case.tensor_sizes)
    print_case_config(
        case,
        block_num,
        warmup,
        repeat,
        device_type,
        device_id,
        cache_buffer_capacity_gb,
        object_target_bytes,
    )
    result = run_transport(
        case.name,
        "ffts_pipeline",
        case.tensor_sizes,
        block_num,
        device_type,
        device_id,
        warmup,
        repeat,
        cache_buffer_capacity_gb,
        object_target_bytes,
    )
    print_result(result)

    if min_gbps > 0:
        assert result.gbps >= min_gbps, (
            f"FFTS pipeline bandwidth {result.gbps:.3f}GB/s is lower than "
            f"UCM_FFTS_MIN_GBPS={min_gbps:.3f}GB/s"
        )
    print_summary_table([result])


if __name__ == "__main__":
    main()
