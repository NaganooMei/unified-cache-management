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

from cache_h2d_ffts_pipeline_test import (
    GIB,
    KIB,
    build_config,
    cmp_and_print_diff,
    env_bool,
    env_float,
    env_int,
    format_bytes,
    make_empty_like,
    make_tensors,
    object_plan_stats,
    poison_tensors,
    prepare_cache,
    prepare_torch_backend,
    synchronize_device,
    tensor_ptrs,
    tensor_size_histogram,
    torch_device,
)
from ucm.store.pipeline.connector import UcmPipelineStore


@dataclass(frozen=True)
class Dsv4StoreShape:
    name: str
    tensor_sizes: list[int]
    shard_bytes: int


@dataclass
class Dsv4StoreContext:
    shape: Dsv4StoreShape
    worker: UcmPipelineStore
    lookup_store: UcmPipelineStore
    block_ids: list[bytes]
    shard_indexes: list[int]
    src_tensors: list[list]
    dst_tensors: list[list]
    dst_ptrs: np.ndarray


@dataclass
class Dsv4PerfResult:
    case_name: str
    transport: str
    hit_blocks: int
    fa_rows: int
    wa_rows: int
    fa_fragments: int
    wa_fragments: int
    fa_shard_bytes: int
    wa_shard_bytes: int
    fa_payload_bytes: int
    wa_payload_bytes: int
    fa_objects_per_shard: int
    wa_objects_per_shard: int
    fa_max_object_bytes: int
    wa_max_object_bytes: int
    fa_max_object_fragments: int
    wa_max_object_fragments: int
    object_target_bytes: int
    bytes_per_load: int
    avg_seconds: float
    median_seconds: float
    min_seconds: float
    gbps: float


DSV4_FA = Dsv4StoreShape(
    "fa",
    [128 * KIB] * 21 + [16 * KIB] * 21 + [4 * KIB] * 20 + [256] * 21,
    3_186_688,
)
DSV4_WA = Dsv4StoreShape(
    "wa",
    [128 * KIB] * 43 + [16 * KIB] * 42 + [4 * KIB] * 42,
    6_496_256,
)


def cache_buffer_capacity_gb(shard_bytes: int) -> int:
    configured = os.getenv("UCM_FFTS_CACHE_BUFFER_CAPACITY_GB")
    if configured is not None:
        return int(configured)
    min_bytes = shard_bytes * 1024
    return max(4, (min_bytes + GIB - 1) // GIB)


def store_cache_buffer_capacity_gb(shape: Dsv4StoreShape) -> int:
    configured = os.getenv(f"UCM_DSV4_{shape.name.upper()}_CACHE_BUFFER_CAPACITY_GB")
    if configured is not None:
        return int(configured)
    return cache_buffer_capacity_gb(shape.shard_bytes)


def build_store_config(
    unique_id: str,
    shape: Dsv4StoreShape,
    transport: str,
    object_target_bytes: int,
) -> dict:
    config = build_config(
        unique_id,
        transport,
        shape.tensor_sizes,
        store_cache_buffer_capacity_gb(shape),
        object_target_bytes,
    )
    config["shard_size"] = shape.shard_bytes
    config["block_size"] = shape.shard_bytes
    return config


def make_store_context(
    case_name: str,
    shape: Dsv4StoreShape,
    row_count: int,
    keys: list[bytes],
    device: str,
    device_type: str,
    device_id: int,
    transport: str,
    object_target_bytes: int,
) -> Dsv4StoreContext:
    unique_id = f"h2d-dsv4-{case_name}-{shape.name}-{transport}-{secrets.token_hex(8)}"
    config = build_store_config(unique_id, shape, transport, object_target_bytes)
    worker = UcmPipelineStore(config | {"device_id": device_id})
    lookup_store = UcmPipelineStore(config) if bool(config["share_buffer_enable"]) else worker

    src_tensors = make_tensors(row_count, shape.tensor_sizes, device)
    dst_tensors = make_empty_like(src_tensors)
    synchronize_device(device_type, device_id)

    src_ptrs = tensor_ptrs(src_tensors)
    dst_ptrs = tensor_ptrs(dst_tensors)
    shard_indexes = [0 for _ in range(row_count)]
    prepare_cache(worker, lookup_store, keys, shard_indexes, src_ptrs)
    return Dsv4StoreContext(
        shape=shape,
        worker=worker,
        lookup_store=lookup_store,
        block_ids=keys,
        shard_indexes=shard_indexes,
        src_tensors=src_tensors,
        dst_tensors=dst_tensors,
        dst_ptrs=dst_ptrs,
    )


def submit_dsv4_load(fa: Dsv4StoreContext, wa: Dsv4StoreContext):
    fa_task = fa.worker.load_data(fa.block_ids, fa.shard_indexes, fa.dst_ptrs)
    wa_task = wa.worker.load_data(wa.block_ids, wa.shard_indexes, wa.dst_ptrs)
    return fa_task, wa_task


def wait_dsv4_load(fa: Dsv4StoreContext, wa: Dsv4StoreContext, tasks) -> None:
    fa_task, wa_task = tasks
    fa.worker.wait(fa_task)
    wa.worker.wait(wa_task)


def time_dsv4_load(fa: Dsv4StoreContext, wa: Dsv4StoreContext) -> float:
    start = time.perf_counter()
    tasks = submit_dsv4_load(fa, wa)
    wait_dsv4_load(fa, wa, tasks)
    return time.perf_counter() - start


def validate_dsv4_load(
    fa: Dsv4StoreContext,
    wa: Dsv4StoreContext,
    device_type: str,
    device_id: int,
) -> None:
    poison_tensors(fa.dst_tensors, device_type, device_id)
    poison_tensors(wa.dst_tensors, device_type, device_id)
    time_dsv4_load(fa, wa)
    cmp_and_print_diff(fa.src_tensors, fa.dst_tensors)
    cmp_and_print_diff(wa.src_tensors, wa.dst_tensors)


def print_store_config(
    shape: Dsv4StoreShape,
    rows: int,
    transport: str,
    object_target_bytes: int,
) -> None:
    objects_per_shard = 0
    max_object_bytes = 0
    max_object_fragments = 0
    if transport == "ffts_pipeline":
        objects_per_shard, max_object_bytes, max_object_fragments = object_plan_stats(
            shape.tensor_sizes, object_target_bytes
        )
    print(
        f"dsv4_{shape.name}_config: rows={rows}, "
        f"fragments={len(shape.tensor_sizes)}, "
        f"payload={format_bytes(sum(shape.tensor_sizes))}, "
        f"shard={format_bytes(shape.shard_bytes)}, "
        f"tensor_sizes=[{tensor_size_histogram(shape.tensor_sizes)}], "
        f"object_target={format_bytes(object_target_bytes)}, "
        f"objects_per_shard={objects_per_shard}, "
        f"max_object={format_bytes(max_object_bytes)}, "
        f"max_object_fragments={max_object_fragments}, "
        f"cache_buffer_capacity_gb={store_cache_buffer_capacity_gb(shape)}"
    )


def print_expected_io(hit_blocks: int, transport: str, object_target_bytes: int) -> None:
    copy_128k = 21 * hit_blocks + 43
    copy_16k = 21 * hit_blocks + 42
    copy_4k = 20 * hit_blocks + 42
    copy_256b = 21 * hit_blocks
    ce_fragment_copies = copy_128k + copy_16k + copy_4k + copy_256b
    ffts_objects = 0
    if transport == "ffts_pipeline":
        fa_objects, _, _ = object_plan_stats(DSV4_FA.tensor_sizes, object_target_bytes)
        wa_objects, _, _ = object_plan_stats(DSV4_WA.tensor_sizes, object_target_bytes)
        ffts_objects = hit_blocks * fa_objects + wa_objects
    bytes_per_load = hit_blocks * sum(DSV4_FA.tensor_sizes) + sum(DSV4_WA.tensor_sizes)
    print(
        "dsv4_expected_io: "
        f"128K={copy_128k}, 16K={copy_16k}, 4K={copy_4k}, 256B={copy_256b}, "
        f"ce_fragment_copies={ce_fragment_copies}, "
        f"ffts_objects={ffts_objects}, bytes={bytes_per_load}"
    )


def make_result(
    transport: str,
    hit_blocks: int,
    object_target_bytes: int,
    samples: list[float],
) -> Dsv4PerfResult:
    fa_objects, fa_max_object, fa_max_frags = object_plan_stats(
        DSV4_FA.tensor_sizes, object_target_bytes
    )
    wa_objects, wa_max_object, wa_max_frags = object_plan_stats(
        DSV4_WA.tensor_sizes, object_target_bytes
    )
    if transport != "ffts_pipeline":
        fa_objects = wa_objects = 0
        fa_max_object = wa_max_object = 0
        fa_max_frags = wa_max_frags = 0

    bytes_per_load = hit_blocks * sum(DSV4_FA.tensor_sizes) + sum(DSV4_WA.tensor_sizes)
    avg_seconds = sum(samples) / len(samples)
    return Dsv4PerfResult(
        case_name="dsv4_fawa",
        transport=transport,
        hit_blocks=hit_blocks,
        fa_rows=hit_blocks,
        wa_rows=1,
        fa_fragments=len(DSV4_FA.tensor_sizes),
        wa_fragments=len(DSV4_WA.tensor_sizes),
        fa_shard_bytes=DSV4_FA.shard_bytes,
        wa_shard_bytes=DSV4_WA.shard_bytes,
        fa_payload_bytes=sum(DSV4_FA.tensor_sizes),
        wa_payload_bytes=sum(DSV4_WA.tensor_sizes),
        fa_objects_per_shard=fa_objects,
        wa_objects_per_shard=wa_objects,
        fa_max_object_bytes=fa_max_object,
        wa_max_object_bytes=wa_max_object,
        fa_max_object_fragments=fa_max_frags,
        wa_max_object_fragments=wa_max_frags,
        object_target_bytes=object_target_bytes,
        bytes_per_load=bytes_per_load,
        avg_seconds=avg_seconds,
        median_seconds=statistics.median(samples),
        min_seconds=min(samples),
        gbps=bytes_per_load / avg_seconds / 1e9,
    )


def print_result(result: Dsv4PerfResult) -> None:
    print(
        f"{result.case_name}/{result.transport}: "
        f"hit_blocks={result.hit_blocks}, "
        f"fa_rows={result.fa_rows}, wa_rows={result.wa_rows}, "
        f"fa_fragments={result.fa_fragments}, wa_fragments={result.wa_fragments}, "
        f"fa_payload={format_bytes(result.fa_payload_bytes)}, "
        f"wa_payload={format_bytes(result.wa_payload_bytes)}, "
        f"object_target={format_bytes(result.object_target_bytes)}, "
        f"fa_objects_per_shard={result.fa_objects_per_shard}, "
        f"wa_objects_per_shard={result.wa_objects_per_shard}, "
        f"fa_max_object={format_bytes(result.fa_max_object_bytes)}, "
        f"wa_max_object={format_bytes(result.wa_max_object_bytes)}, "
        f"fa_max_object_fragments={result.fa_max_object_fragments}, "
        f"wa_max_object_fragments={result.wa_max_object_fragments}, "
        f"bytes={result.bytes_per_load}, "
        f"avg={result.avg_seconds * 1e3:.3f}ms, "
        f"median={result.median_seconds * 1e3:.3f}ms, "
        f"min={result.min_seconds * 1e3:.3f}ms, "
        f"bw={result.gbps:.3f}GB/s"
    )


def print_summary(result: Dsv4PerfResult) -> None:
    print(
        "summary: "
        "case,transport,hit_blocks,fa_rows,wa_rows,fa_fragments,wa_fragments,"
        "fa_payload,wa_payload,fa_shard,wa_shard,object_target,"
        "fa_objects_per_shard,wa_objects_per_shard,fa_max_object,wa_max_object,"
        "fa_max_object_fragments,wa_max_object_fragments,bytes,avg_ms,median_ms,min_ms,gbps"
    )
    print(
        "summary: "
        f"{result.case_name},{result.transport},{result.hit_blocks},"
        f"{result.fa_rows},{result.wa_rows},{result.fa_fragments},{result.wa_fragments},"
        f"{format_bytes(result.fa_payload_bytes)},{format_bytes(result.wa_payload_bytes)},"
        f"{format_bytes(result.fa_shard_bytes)},{format_bytes(result.wa_shard_bytes)},"
        f"{format_bytes(result.object_target_bytes)},{result.fa_objects_per_shard},"
        f"{result.wa_objects_per_shard},{format_bytes(result.fa_max_object_bytes)},"
        f"{format_bytes(result.wa_max_object_bytes)},{result.fa_max_object_fragments},"
        f"{result.wa_max_object_fragments},{result.bytes_per_load},"
        f"{result.avg_seconds * 1e3:.3f},{result.median_seconds * 1e3:.3f},"
        f"{result.min_seconds * 1e3:.3f},{result.gbps:.3f}"
    )


def main():
    os.environ.setdefault("UC_LOGGER_LEVEL", "info")

    hit_blocks = env_int("UCM_FFTS_BLOCK_NUM", 16)
    warmup = env_int("UCM_FFTS_WARMUP", 2)
    repeat = env_int("UCM_FFTS_REPEAT", 10)
    if hit_blocks <= 0 or warmup < 0 or repeat <= 0:
        raise ValueError("UCM_FFTS_BLOCK_NUM and UCM_FFTS_REPEAT must be positive")

    transport = os.getenv("UCM_FFTS_H2D_TRANSPORT", "ffts_pipeline").strip().lower()
    if transport not in ("ce", "ffts_pipeline"):
        raise ValueError("UCM_FFTS_H2D_TRANSPORT must be ce or ffts_pipeline")
    object_target_bytes = env_int("UCM_FFTS_OBJECT_TARGET_BYTES", 0)
    if transport == "ce":
        object_target_bytes = 0

    device_type = os.getenv("UCM_FFTS_TORCH_DEVICE", "cuda")
    device_id = env_int("UCM_FFTS_DEVICE_ID", 0)
    validate = env_bool("UCM_FFTS_VALIDATE", True)
    min_gbps = env_float("UCM_FFTS_MIN_GBPS", 0.0)
    prepare_torch_backend(device_type)
    device = torch_device(device_type, device_id)

    print(
        "case_config: "
        f"case=dsv4_fawa, transport={transport}, device={device}, "
        f"hit_blocks={hit_blocks}, warmup={warmup}, repeat={repeat}, "
        f"validate={validate}, share_buffer_enable={env_bool('UCM_FFTS_SHARE_BUFFER_ENABLE', True)}"
    )
    print_store_config(DSV4_FA, hit_blocks, transport, object_target_bytes)
    print_store_config(DSV4_WA, 1, transport, object_target_bytes)
    print_expected_io(hit_blocks, transport, object_target_bytes)

    fa_keys = [secrets.token_bytes(16) for _ in range(hit_blocks)]
    wa_keys = fa_keys[-1:]
    fa = make_store_context(
        "dsv4_fawa",
        DSV4_FA,
        hit_blocks,
        fa_keys,
        device,
        device_type,
        device_id,
        transport,
        object_target_bytes,
    )
    wa = make_store_context(
        "dsv4_fawa",
        DSV4_WA,
        1,
        wa_keys,
        device,
        device_type,
        device_id,
        transport,
        object_target_bytes,
    )

    if validate:
        validate_dsv4_load(fa, wa, device_type, device_id)
    for _ in range(warmup):
        time_dsv4_load(fa, wa)
    samples = [time_dsv4_load(fa, wa) for _ in range(repeat)]
    if validate:
        validate_dsv4_load(fa, wa, device_type, device_id)

    result = make_result(transport, hit_blocks, object_target_bytes, samples)
    print_result(result)
    if min_gbps > 0:
        assert result.gbps >= min_gbps, (
            f"DSV4 FAWA bandwidth {result.gbps:.3f}GB/s is lower than "
            f"UCM_FFTS_MIN_GBPS={min_gbps:.3f}GB/s"
        )
    print_summary(result)


if __name__ == "__main__":
    main()
