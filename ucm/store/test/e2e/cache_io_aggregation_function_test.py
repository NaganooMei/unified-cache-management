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

import torch

from ucm.store.pipeline.connector import UcmPipelineStore


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def parse_tensor_sizes() -> list[int]:
    sizes = os.getenv("UCM_IO_AGGREGATION_TENSOR_SIZES")
    if sizes:
        return [int(item.strip()) for item in sizes.split(",") if item.strip()]
    tensor_count = env_int("UCM_IO_AGGREGATION_TENSOR_COUNT", 128)
    tensor_bytes = env_int("UCM_IO_AGGREGATION_TENSOR_BYTES", 32768)
    return [tensor_bytes] * tensor_count


def prepare_torch_backend(device_type: str) -> None:
    if device_type != "npu" or hasattr(torch, "npu"):
        return
    try:
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "UCM_IO_AGGREGATION_TORCH_DEVICE=npu requires torch_npu to be importable"
        ) from exc


def make_device(device_type: str, device_id: int) -> str:
    return f"{device_type}:{device_id}"


def synchronize_device(device_type: str, device_id: int) -> None:
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device_id)
        return
    if device_type == "npu" and hasattr(torch, "npu"):
        try:
            torch.npu.synchronize(device_id)
        except TypeError:
            torch.npu.synchronize()


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


def make_tensors(
    request_size: int,
    tensor_sizes: list[int],
    device: str,
) -> list[list[torch.Tensor]]:
    element_size = torch.tensor([], dtype=torch.bfloat16).element_size()
    if any(size <= 0 or size % element_size != 0 for size in tensor_sizes):
        raise ValueError(f"invalid tensor byte sizes for bfloat16: {tensor_sizes}")
    return [
        [
            torch.rand(
                [tensor_size // element_size],
                dtype=torch.bfloat16,
                device=device,
            )
            for tensor_size in tensor_sizes
        ]
        for _ in range(request_size)
    ]


def e2e_test(
    worker: UcmPipelineStore,
    scheduler: UcmPipelineStore,
    tensor_sizes: list[int],
    request_size: int,
    device: str,
    device_type: str,
    device_id: int,
):
    block_ids = [secrets.token_bytes(16) for _ in range(request_size)]
    founds = scheduler.lookup(block_ids)
    assert not any(founds)
    assert scheduler.lookup_on_prefix(block_ids) == -1

    shard_indexes = [0 for _ in range(request_size)]
    src_tensors = make_tensors(request_size, tensor_sizes, device)
    synchronize_device(device_type, device_id)
    task = worker.dump(block_ids, shard_indexes, src_tensors)
    worker.wait(task)

    founds = scheduler.lookup(block_ids)
    assert all(founds)
    assert scheduler.lookup_on_prefix(block_ids) + 1 == request_size

    dst_tensors = [[torch.empty_like(t) for t in row] for row in src_tensors]
    task = worker.load(block_ids, shard_indexes, dst_tensors)
    worker.wait(task)
    cmp_and_print_diff(src_tensors, dst_tensors)


def main():
    os.environ.setdefault("UC_LOGGER_LEVEL", "debug")

    device_type = os.getenv("UCM_IO_AGGREGATION_TORCH_DEVICE", "cuda")
    device_id = env_int("UCM_IO_AGGREGATION_DEVICE_ID", 0)
    prepare_torch_backend(device_type)
    device = make_device(device_type, device_id)

    tensor_sizes = parse_tensor_sizes()
    shard_size = sum(tensor_sizes)
    request_size = env_int("UCM_IO_AGGREGATION_FUNCTION_REQUEST_SIZE", 16)
    test_batch_number = env_int("UCM_IO_AGGREGATION_FUNCTION_BATCHES", 4)

    config = {}
    config["store_pipeline"] = "Cache|Empty"
    config["unique_id"] = secrets.token_hex(8)
    config["timeout_ms"] = env_int("UCM_IO_AGGREGATION_TIMEOUT_MS", 30000)
    config["tensor_size_list"] = tensor_sizes
    config["shard_size"] = shard_size
    config["block_size"] = shard_size
    config["share_buffer_enable"] = True
    config["cache_buffer_capacity_gb"] = env_int(
        "UCM_IO_AGGREGATION_CACHE_BUFFER_CAPACITY_GB", 4
    )
    config["cache_load_exclusive_buffer_number"] = env_int(
        "UCM_IO_AGGREGATION_LOAD_EXCLUSIVE_BUFFER_NUMBER", 64
    )
    config["waiting_queue_depth"] = env_int(
        "UCM_IO_AGGREGATION_WAITING_QUEUE_DEPTH", 64
    )
    config["running_queue_depth"] = env_int(
        "UCM_IO_AGGREGATION_RUNNING_QUEUE_DEPTH", 4096
    )
    config["cache_stream_number"] = env_int("UCM_IO_AGGREGATION_CACHE_STREAM_NUMBER", 4)
    config["cache_io_aggregation"] = True

    worker = UcmPipelineStore(config | {"device_id": device_id})
    scheduler = UcmPipelineStore(config)

    for _ in range(test_batch_number):
        e2e_test(
            worker,
            scheduler,
            tensor_sizes,
            request_size,
            device,
            device_type,
            device_id,
        )


if __name__ == "__main__":
    main()
