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
"""Mooncake ``UcmMooncakeStoreV1`` multi-process smoke test: one scheduler + two workers.

Prerequisites (Linux + Ascend):

- Python package ``mooncake`` (see Mooncake build docs).
- Start Mooncake master before running, for example::

    mooncake_master \\
      --port 50088 \\
      --eviction_high_watermark_ratio 0.9 \\
      --eviction_ratio 0.1 \\
      --default_kv_lease_ttl 11000

- Recommended environment (from UCM prefix-cache docs; adjust paths as needed)::

    export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$LD_LIBRARY_PATH
    export PYTHONHASHSEED=0
    export HCCL_INTRA_ROCE_ENABLE=1
    export HCCL_RDMA_TIMEOUT=17
    export ASCEND_CONNECT_TIMEOUT=10000

Optional environment overrides:

- ``UCM_MOONCAKE_MASTER`` (default ``127.0.0.1:50088``)
- ``UCM_MOONCAKE_METADATA`` (default ``P2PHANDSHAKE``)
- ``UCM_MOONCAKE_LOCAL_HOSTNAME`` (default ``127.0.0.1``)
- ``UCM_MOONCAKE_GLOBAL_SEGMENT`` (default ``256MB``)
- ``UCM_MOONCAKE_LOCAL_BUFFER`` (default ``256MB``)
- ``UCM_MOONCAKE_UNIQUE_ID`` (optional cluster id string)

This test targets the UCM store layer only (not vLLM). Worker0 uses ``npu:0`` and
worker1 uses ``npu:1`` (at least two NPUs required). Workers exercise the low-level
``dump_data`` / ``load_data`` APIs (device pointer matrices), not ``dump`` / ``load``.

Run directly (no pytest required) from the repo root with ``PYTHONPATH`` set so
that ``ucm`` is importable, for example::

    PYTHONPATH=. python ucm/store/test/e2e/mooncake_scheduler_two_workers_test.py
"""
from __future__ import annotations

import multiprocessing
import os
import secrets
import sys
import time
from typing import Any

import numpy as np
import torch

from ucm.store.factory_v1 import UcmConnectorFactoryV1

ROLE_SCHEDULER = 0
ROLE_WORKER0 = 1
ROLE_WORKER1 = 2
_NUM_PARTIES = 3


def _mooncake_base_config() -> dict[str, Any]:
    return {
        "protocol": "ascend",
        "local_hostname": os.environ.get("UCM_MOONCAKE_LOCAL_HOSTNAME", "127.0.0.1"),
        "metadata_server": os.environ.get("UCM_MOONCAKE_METADATA", "P2PHANDSHAKE"),
        "master_server_address": os.environ.get(
            "UCM_MOONCAKE_MASTER", "127.0.0.1:50088"
        ),
        "device_name": "",
        "global_segment_size": os.environ.get("UCM_MOONCAKE_GLOBAL_SEGMENT", "256MB"),
        "local_buffer_size": os.environ.get("UCM_MOONCAKE_LOCAL_BUFFER", "256MB"),
        "executor_workers": 2,
        "unique_id": os.environ.get("UCM_MOONCAKE_UNIQUE_ID", "mooncake_e2e_test"),
    }


def _scheduler_config(base: dict[str, Any]) -> dict[str, Any]:
    cfg = {k: v for k, v in base.items() if k != "device_id"}
    cfg.pop("device_id", None)
    return cfg


def _make_block_tensor(row_idx: int, numel: int, device_id: int) -> torch.Tensor:
    return torch.full(
        (numel,),
        fill_value=float(row_idx + 1),
        dtype=torch.bfloat16,
        device=f"npu:{device_id}",
    )


def _tensor_rows_to_addr_array(tensors: list[list[torch.Tensor]]) -> np.ndarray:
    """Build a pointer matrix for ``load_data`` / ``dump_data`` (same layout as connector)."""
    return np.asarray(
        [[int(t.data_ptr()) for t in row] for row in tensors],
        dtype=np.uint64,
    )


def _worker_config(
    base: dict[str, Any], device_id: int, tensors: list[list[torch.Tensor]]
) -> dict[str, Any]:
    ptrs: list[int] = []
    sizes: list[int] = []
    for row in tensors:
        for t in row:
            ptrs.append(int(t.data_ptr()))
            sizes.append(int(t.numel() * t.element_size()))
    sample = tensors[0][0]
    tensor_size_list = (int(sample.numel() * sample.element_size()),)
    cfg = dict(base)
    cfg["device_id"] = device_id
    cfg["tensor_size_list"] = tensor_size_list
    cfg["register_buffer_ptrs"] = tuple(ptrs)
    cfg["register_buffer_sizes"] = tuple(sizes)
    return cfg


def _mooncake_party(
    role: int,
    barrier: multiprocessing.Barrier,
    block_ids: list[bytes],
    tensor_numel: int,
) -> None:
    store = None
    try:
        base = _mooncake_base_config()
        if role == ROLE_SCHEDULER:
            store = UcmConnectorFactoryV1.create_connector(
                "UcmMooncakeStoreV1", _scheduler_config(base)
            )
        elif role == ROLE_WORKER0:
            torch.npu.set_device(0)
            src_tensors = [
                [_make_block_tensor(i, tensor_numel, 0)] for i in range(len(block_ids))
            ]
            cfg = _worker_config(base, 0, src_tensors)
            store = UcmConnectorFactoryV1.create_connector("UcmMooncakeStoreV1", cfg)
        else:
            torch.npu.set_device(1)
            dst_tensors = [
                [torch.zeros((tensor_numel,), dtype=torch.bfloat16, device="npu:1")]
                for _ in block_ids
            ]
            cfg = _worker_config(base, 1, dst_tensors)
            store = UcmConnectorFactoryV1.create_connector("UcmMooncakeStoreV1", cfg)

        barrier.wait()

        if role == ROLE_WORKER0:
            assert store is not None
            shard_indexes = [0] * len(block_ids)
            src_addrs = _tensor_rows_to_addr_array(src_tensors)
            task = store.dump_data(block_ids, shard_indexes, src_addrs)
            store.wait(task)

        barrier.wait()

        if role == ROLE_WORKER1:
            assert store is not None
            shard_indexes = [0] * len(block_ids)
            dst_addrs = _tensor_rows_to_addr_array(dst_tensors)
            task = store.load_data(block_ids, shard_indexes, dst_addrs)
            store.wait(task)
            for i, row in enumerate(dst_tensors):
                expected = torch.full(
                    (tensor_numel,),
                    float(i + 1),
                    dtype=torch.bfloat16,
                )
                if not torch.allclose(row[0].cpu(), expected):
                    raise AssertionError(
                        f"load mismatch at block row {i}: got {row[0].cpu()[:8]}, "
                        f"expected {expected[:8]}"
                    )

        barrier.wait()

        if role == ROLE_SCHEDULER:
            assert store is not None
            hits = store.lookup(block_ids)
            if not all(hits):
                raise AssertionError(f"scheduler lookup expected all True, got {hits}")
            prefix_idx = store.lookup_on_prefix(block_ids)
            if prefix_idx != len(block_ids) - 1:
                raise AssertionError(
                    f"lookup_on_prefix expected {len(block_ids) - 1}, got {prefix_idx}"
                )

        barrier.wait()
    finally:
        if store is not None:
            store.shutdown()


def _run_multiprocess_e2e(block_count: int = 6, tensor_numel: int = 256) -> None:
    multiprocessing.set_start_method("spawn", force=True)
    block_ids = [secrets.token_bytes(16) for _ in range(block_count)]
    barrier = multiprocessing.Barrier(_NUM_PARTIES)
    workers: list[multiprocessing.Process] = []
    for role in (ROLE_SCHEDULER, ROLE_WORKER0, ROLE_WORKER1):
        p = multiprocessing.Process(
            target=_mooncake_party,
            args=(role, barrier, block_ids, tensor_numel),
        )
        workers.append(p)
        p.start()
    for p in workers:
        p.join()
    for p in workers:
        if p.exitcode != 0:
            raise RuntimeError(f"mooncake e2e child exit code {p.exitcode}")


def _ascend_two_npu_available() -> bool:
    if not hasattr(torch, "npu"):
        return False
    try:
        return int(torch.npu.device_count()) >= 2
    except Exception:
        return False


def run_mooncake_scheduler_two_workers_main() -> None:
    if not hasattr(torch, "npu"):
        print("SKIP: torch.npu is not available (need Ascend environment).", file=sys.stderr)
        sys.exit(0)
    if not _ascend_two_npu_available():
        print(
            "SKIP: need at least 2 NPUs (set CUDA/NPU devices for worker0 and worker1).",
            file=sys.stderr,
        )
        sys.exit(0)
    time.sleep(0.5)
    _run_multiprocess_e2e()


if __name__ == "__main__":
    run_mooncake_scheduler_two_workers_main()
