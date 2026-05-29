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
    transport: str
    bytes_per_load: int
    avg_seconds: float
    median_seconds: float
    min_seconds: float
    gbps: float


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def parse_tensor_sizes() -> list[int]:
    sizes = os.getenv("UCM_FFTS_TENSOR_SIZES")
    if sizes:
        return [int(item.strip()) for item in sizes.split(",") if item.strip()]
    fragment_count = env_int("UCM_FFTS_FRAGMENT_COUNT", 128)
    fragment_bytes = env_int("UCM_FFTS_FRAGMENT_BYTES", 32768)
    return [fragment_bytes] * fragment_count


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
) -> dict:
    shard_size = sum(tensor_sizes)
    return {
        "store_pipeline": "Cache|Empty",
        "unique_id": unique_id,
        "timeout_ms": env_int("UCM_FFTS_TIMEOUT_MS", 30000),
        "tensor_size_list": tensor_sizes,
        "shard_size": shard_size,
        "block_size": shard_size,
        "share_buffer_enable": True,
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
    }


def prepare_cache(
    worker: UcmPipelineStore,
    scheduler: UcmPipelineStore,
    block_ids: list[bytes],
    shard_indexes: list[int],
    src_ptrs: np.ndarray,
) -> None:
    assert not any(scheduler.lookup(block_ids))
    assert scheduler.lookup_on_prefix(block_ids) == -1
    task = worker.dump_data(block_ids, shard_indexes, src_ptrs)
    worker.wait(task)
    assert all(scheduler.lookup(block_ids))
    assert scheduler.lookup_on_prefix(block_ids) + 1 == len(block_ids)


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


def run_transport(
    transport: str,
    tensor_sizes: list[int],
    block_num: int,
    device_type: str,
    device_id: int,
    warmup: int,
    repeat: int,
    cache_buffer_capacity_gb: int,
) -> LoadPerfResult:
    device = torch_device(device_type, device_id)
    unique_id = f"h2d-ffts-{transport}-{secrets.token_hex(8)}"
    config = build_config(unique_id, transport, tensor_sizes, cache_buffer_capacity_gb)
    worker = UcmPipelineStore(config | {"device_id": device_id})
    scheduler = UcmPipelineStore(config)

    block_ids = [secrets.token_bytes(16) for _ in range(block_num)]
    shard_indexes = [0 for _ in range(block_num)]
    src_tensors = make_tensors(block_num, tensor_sizes, device)
    dst_tensors = make_empty_like(src_tensors)
    synchronize_device(device_type, device_id)

    src_ptrs = tensor_ptrs(src_tensors)
    dst_ptrs = tensor_ptrs(dst_tensors)
    prepare_cache(worker, scheduler, block_ids, shard_indexes, src_ptrs)

    for _ in range(warmup):
        time_load(worker, block_ids, shard_indexes, dst_ptrs)
    cmp_and_print_diff(src_tensors, dst_tensors)

    samples = [
        time_load(worker, block_ids, shard_indexes, dst_ptrs) for _ in range(repeat)
    ]
    cmp_and_print_diff(src_tensors, dst_tensors)

    bytes_per_load = block_num * sum(tensor_sizes)
    avg_seconds = sum(samples) / len(samples)
    return LoadPerfResult(
        transport=transport,
        bytes_per_load=bytes_per_load,
        avg_seconds=avg_seconds,
        median_seconds=statistics.median(samples),
        min_seconds=min(samples),
        gbps=bytes_per_load / avg_seconds / 1e9,
    )


def print_result(result: LoadPerfResult) -> None:
    print(
        f"{result.transport}: bytes={result.bytes_per_load}, "
        f"avg={result.avg_seconds * 1e3:.3f}ms, "
        f"median={result.median_seconds * 1e3:.3f}ms, "
        f"min={result.min_seconds * 1e3:.3f}ms, "
        f"bw={result.gbps:.3f}GB/s"
    )


def main():
    os.environ.setdefault("UC_LOGGER_LEVEL", "info")

    tensor_sizes = parse_tensor_sizes()
    block_num = env_int("UCM_FFTS_BLOCK_NUM", 16)
    warmup = env_int("UCM_FFTS_WARMUP", 2)
    repeat = env_int("UCM_FFTS_REPEAT", 10)
    if block_num <= 0 or warmup < 0 or repeat <= 0:
        raise ValueError("UCM_FFTS_BLOCK_NUM and UCM_FFTS_REPEAT must be positive")
    device_type = os.getenv("UCM_FFTS_TORCH_DEVICE", "cuda")
    device_id = env_int("UCM_FFTS_DEVICE_ID", 0)
    prepare_torch_backend(device_type)
    cache_buffer_capacity_gb = env_int("UCM_FFTS_CACHE_BUFFER_CAPACITY_GB", 4)
    max_slowdown = env_float("UCM_FFTS_MAX_SLOWDOWN", 5.0)
    min_gbps = env_float("UCM_FFTS_MIN_GBPS", 0.0)
    compare_ce = os.getenv("UCM_FFTS_COMPARE_CE", "1") == "1"

    print(
        "config: "
        f"device={device_type}:{device_id}, blocks={block_num}, "
        f"fragments={len(tensor_sizes)}, shard_bytes={sum(tensor_sizes)}, "
        f"warmup={warmup}, repeat={repeat}"
    )

    ce_result = None
    if compare_ce:
        ce_result = run_transport(
            "ce",
            tensor_sizes,
            block_num,
            device_type,
            device_id,
            warmup,
            repeat,
            cache_buffer_capacity_gb,
        )
        print_result(ce_result)

    ffts_result = run_transport(
        "ffts_pipeline",
        tensor_sizes,
        block_num,
        device_type,
        device_id,
        warmup,
        repeat,
        cache_buffer_capacity_gb,
    )
    print_result(ffts_result)

    if min_gbps > 0:
        assert ffts_result.gbps >= min_gbps, (
            f"FFTS pipeline bandwidth {ffts_result.gbps:.3f}GB/s is lower than "
            f"UCM_FFTS_MIN_GBPS={min_gbps:.3f}GB/s"
        )
    if ce_result is not None:
        slowdown = ffts_result.avg_seconds / ce_result.avg_seconds
        print(f"ffts_vs_ce_slowdown={slowdown:.3f}x")
        assert slowdown <= max_slowdown, (
            f"FFTS pipeline load is {slowdown:.3f}x slower than CE, "
            f"exceeding UCM_FFTS_MAX_SLOWDOWN={max_slowdown:.3f}"
        )


if __name__ == "__main__":
    main()
