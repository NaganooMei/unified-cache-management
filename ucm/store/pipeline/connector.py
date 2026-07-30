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
import array
import copy
import ctypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import torch

from ucm.store.pipeline import ucmpipelinestore
from ucm.store.ucmstore_v1 import Task, UcmKVStoreBaseV1

_preloaded_libraries: Dict[Path, ctypes.CDLL] = {}
StoreNotFoundError = ucmpipelinestore.StoreNotFoundError
StoreUnhealthyError = ucmpipelinestore.StoreUnhealthyError


def _preload_library(path: Path) -> None:
    if os.name != "posix" or not path.exists():
        return
    resolved = path.resolve()
    if resolved in _preloaded_libraries:
        return
    _preloaded_libraries[resolved] = ctypes.CDLL(
        str(resolved),
        mode=getattr(os, "RTLD_NOW", 0) | getattr(os, "RTLD_GLOBAL", 0),
    )


def _preload_metrics(store_dir: Path) -> None:
    _preload_library(store_dir.parent / "shared/metrics/libmetrics.so")


class UcmPipelineStoreBuilder:
    registry_: Dict[
        str, Callable[[Dict[str, object], ucmpipelinestore.PipelineStore], None]
    ] = {}

    @classmethod
    def register(
        cls,
        name: str,
        builder: Callable[[Dict[str, object], ucmpipelinestore.PipelineStore], None],
    ) -> None:
        if name in cls.registry_:
            raise ValueError(f"Builder '{name}' is already registered.")
        cls.registry_[name] = builder

    @classmethod
    def get(
        cls, name: str
    ) -> Callable[[Dict[str, object], ucmpipelinestore.PipelineStore], None]:
        return cls.registry_.get(name)


@dataclass
class UcmPipelineStoreTransTask(Task):
    task_id: int


@dataclass
class UcmPipelineStoreBroadcastStats:
    root_load_cost: float
    broadcast_cost: float
    scatter_cost: float
    total_cost: float


@dataclass
class _BroadcastResource:
    buffer: torch.Tensor
    load_addrs: np.ndarray
    source_offsets: np.ndarray
    status: torch.Tensor


class UcmPipelineStore(UcmKVStoreBaseV1):
    def __init__(self, config: Dict[str, object]) -> None:
        super().__init__(config)
        health_config = copy.deepcopy(config.get("store_health", {}))
        self.store_ = ucmpipelinestore.PipelineStore(health_config)
        builder = UcmPipelineStoreBuilder.get(config["store_pipeline"])
        if builder is None:
            raise ValueError(f"unknown store pipeline: {config['store_pipeline']}")
        builder(config, self.store_)
        self._broadcast_enabled = bool(config.get("cache_tp_broadcast_scatter", False))
        self._broadcast_resources: Dict[tuple, _BroadcastResource] = {}
        self._broadcast_streams: Dict[str, Any] = {}

    def cc_store(self) -> int:
        return self.store_.Self()

    def lookup(self, block_ids: List[bytes]) -> List[bool]:
        flat = np.frombuffer(b"".join(block_ids), dtype=np.uint8)
        res = self.store_.Lookup(flat)
        return np.frombuffer(res, dtype=bool)

    def lookup_on_prefix(self, block_ids: List[bytes]) -> int:
        flat = np.frombuffer(b"".join(block_ids), dtype=np.uint8)
        return self.store_.LookupOnPrefix(flat)

    def prefetch(self, block_ids: List[bytes]) -> None:
        flat = np.frombuffer(b"".join(block_ids), dtype=np.uint8)
        self.store_.Prefetch(flat)

    def _tensor_normalize(self, tensors: List[List[torch.Tensor]]) -> np.ndarray:
        n_rows = len(tensors)
        n_cols = len(tensors[0])
        flat = np.fromiter(
            (t for row in tensors for t in row), dtype=object, count=n_rows * n_cols
        )
        ptrs = np.vectorize(torch.Tensor.data_ptr, otypes=[np.uint64])(flat)
        return ptrs.reshape(n_rows, n_cols)

    def load(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        dst_tensor: List[List[torch.Tensor]],
    ) -> Task:
        ids = np.frombuffer(b"".join(block_ids), dtype=np.uint8)
        indexes = array.array("Q", shard_index)
        addrs = self._tensor_normalize(dst_tensor)
        task_id = self.store_.Load(ids, indexes, addrs)
        return UcmPipelineStoreTransTask(task_id)

    def dump(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        src_tensor: List[List[torch.Tensor]],
    ) -> Task:
        ids = np.frombuffer(b"".join(block_ids), dtype=np.uint8)
        indexes = array.array("Q", shard_index)
        addrs = self._tensor_normalize(src_tensor)
        task_id = self.store_.Dump(ids, indexes, addrs)
        return UcmPipelineStoreTransTask(task_id)

    def load_data(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        dst_addr: List[List[int]] | np.ndarray,
    ) -> Task:
        ids = np.frombuffer(b"".join(block_ids), dtype=np.uint8)
        indexes = array.array("Q", shard_index)
        if isinstance(dst_addr, np.ndarray):
            addrs = dst_addr
        else:
            addrs = np.array(dst_addr, dtype=np.uint64)
        task_id = self.store_.Load(ids, indexes, addrs)
        return UcmPipelineStoreTransTask(task_id)

    def dump_data(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        src_addr: List[List[int]] | np.ndarray,
        prerequisite_handle: int = 0,
    ) -> Task:
        ids = np.frombuffer(b"".join(block_ids), dtype=np.uint8)
        indexes = array.array("Q", shard_index)
        if isinstance(src_addr, np.ndarray):
            addrs = src_addr
        else:
            addrs = np.array(src_addr, dtype=np.uint64)
        task_id = self.store_.Dump(ids, indexes, addrs, prerequisite_handle)
        return UcmPipelineStoreTransTask(task_id)

    def scatter_from_contiguous(
        self,
        source_addr: int,
        source_offsets: List[int] | np.ndarray,
        destination_addrs: List[List[int]] | np.ndarray,
    ) -> None:
        offsets = np.ascontiguousarray(source_offsets, dtype=np.uint64)
        addrs = np.ascontiguousarray(destination_addrs, dtype=np.uint64)
        if offsets.ndim != 1:
            raise ValueError(f"source_offsets must be 1D, got shape={offsets.shape}")
        if addrs.ndim != 2:
            raise ValueError(f"destination_addrs must be 2D, got shape={addrs.shape}")
        if offsets.shape[0] != addrs.shape[0]:
            raise ValueError(
                "scatter row mismatch: "
                f"offsets={offsets.shape[0]}, addrs={addrs.shape[0]}"
            )
        self.store_.ScatterFromContiguous(int(source_addr), offsets, addrs)

    @staticmethod
    def _broadcast_tensor_sizes(
        tensors: List[List[torch.Tensor]],
    ) -> tuple[torch.device, torch.dtype, tuple[int, ...]]:
        if not tensors or not tensors[0]:
            raise ValueError("broadcast load requires a non-empty tensor matrix")

        column_number = len(tensors[0])
        first = tensors[0][0]
        device = first.device
        dtype = first.dtype
        tensor_sizes = tuple(
            tensor.numel() * tensor.element_size() for tensor in tensors[0]
        )
        for row_index, row in enumerate(tensors):
            if len(row) != column_number:
                raise ValueError(
                    "broadcast tensor column mismatch: "
                    f"row0={column_number}, row{row_index}={len(row)}"
                )
            row_sizes = tuple(tensor.numel() * tensor.element_size() for tensor in row)
            if row_sizes != tensor_sizes:
                raise ValueError(
                    "broadcast tensor size mismatch: "
                    f"row0={tensor_sizes}, row{row_index}={row_sizes}"
                )
            for tensor in row:
                if tensor.device != device or tensor.dtype != dtype:
                    raise ValueError(
                        "broadcast load requires tensors with one device and dtype"
                    )
        return device, dtype, tensor_sizes

    def _get_broadcast_resource(
        self, tensors: List[List[torch.Tensor]]
    ) -> _BroadcastResource:
        device, dtype, tensor_sizes = self._broadcast_tensor_sizes(tensors)
        row_number = len(tensors)
        bytes_per_row = sum(tensor_sizes)
        total_bytes = row_number * bytes_per_row
        element_size = tensors[0][0].element_size()
        if total_bytes % element_size != 0:
            raise ValueError(
                f"broadcast payload {total_bytes} is not aligned to {element_size}"
            )

        key = (str(device), dtype, row_number, tensor_sizes)
        resource = self._broadcast_resources.get(key)
        if resource is not None:
            return resource

        buffer = torch.empty([total_bytes // element_size], dtype=dtype, device=device)
        tensor_offsets = np.asarray(
            [0] + list(np.cumsum(tensor_sizes[:-1], dtype=np.uint64)),
            dtype=np.uint64,
        )
        source_offsets = np.arange(row_number, dtype=np.uint64) * bytes_per_row
        load_addrs = (
            np.uint64(buffer.data_ptr())
            + source_offsets[:, None]
            + tensor_offsets[None, :]
        )
        resource = _BroadcastResource(
            buffer=buffer,
            load_addrs=np.ascontiguousarray(load_addrs, dtype=np.uint64),
            source_offsets=np.ascontiguousarray(source_offsets, dtype=np.uint64),
            status=torch.empty([1], dtype=torch.int32, device=device),
        )
        self._broadcast_resources[key] = resource
        return resource

    def _get_broadcast_stream(self, device: torch.device):
        key = str(device)
        stream = self._broadcast_streams.get(key)
        if stream is not None:
            return stream
        if device.type == "cuda":
            stream = torch.cuda.Stream(device=device)
        elif device.type == "npu":
            stream = torch.npu.Stream()
        else:
            raise ValueError(f"unsupported broadcast device: {device}")
        self._broadcast_streams[key] = stream
        return stream

    @staticmethod
    def _run_broadcast(
        tensor: torch.Tensor, src_rank: int, process_group, stream
    ) -> None:
        if tensor.device.type == "cuda":
            with torch.cuda.stream(stream):
                torch.distributed.broadcast(tensor, src=src_rank, group=process_group)
        else:
            with torch.npu.stream(stream):
                torch.distributed.broadcast(tensor, src=src_rank, group=process_group)
        stream.synchronize()

    def load_broadcast(
        self,
        block_ids: List[bytes],
        shard_index: List[int],
        dst_tensor: List[List[torch.Tensor]],
        src_rank: int = 0,
        process_group=None,
    ) -> UcmPipelineStoreBroadcastStats:
        if not self._broadcast_enabled:
            raise RuntimeError(
                "broadcast load requires cache_tp_broadcast_scatter=True"
            )
        if not torch.distributed.is_initialized():
            raise RuntimeError("broadcast load requires an initialized process group")
        if len(block_ids) != len(dst_tensor) or len(shard_index) != len(block_ids):
            raise ValueError(
                "broadcast load dimension mismatch: "
                f"blocks={len(block_ids)}, indexes={len(shard_index)}, "
                f"destinations={len(dst_tensor)}"
            )

        resource = self._get_broadcast_resource(dst_tensor)
        stream = self._get_broadcast_stream(resource.buffer.device)
        destination_addrs = self._tensor_normalize(dst_tensor)
        total_start = time.perf_counter()

        root_load_cost = 0.0
        root_error = None
        if torch.distributed.get_rank() == src_rank:
            root_load_start = time.perf_counter()
            try:
                task = self.load_data(block_ids, shard_index, resource.load_addrs)
                self.wait(task)
            except Exception as error:  # Keep all ranks in the status collective.
                root_error = error
            root_load_cost = time.perf_counter() - root_load_start

        resource.status.fill_(0 if root_error is not None else 1)
        torch.distributed.broadcast(resource.status, src=src_rank, group=process_group)
        if int(resource.status.item()) == 0:
            if root_error is not None:
                raise root_error
            raise RuntimeError(f"broadcast root rank {src_rank} failed to load data")

        broadcast_start = time.perf_counter()
        self._run_broadcast(resource.buffer, src_rank, process_group, stream)
        broadcast_cost = time.perf_counter() - broadcast_start

        scatter_start = time.perf_counter()
        self.scatter_from_contiguous(
            resource.buffer.data_ptr(),
            resource.source_offsets,
            destination_addrs,
        )
        scatter_cost = time.perf_counter() - scatter_start
        total_cost = time.perf_counter() - total_start
        return UcmPipelineStoreBroadcastStats(
            root_load_cost=root_load_cost,
            broadcast_cost=broadcast_cost,
            scatter_cost=scatter_cost,
            total_cost=total_cost,
        )

    def wait(self, task: Task) -> None:
        return self.store_.Wait(task.task_id)

    def check(self, task: Task) -> bool:
        return self.store_.Check(task.task_id)


def _cache_ds3fs_pipeline_builder(
    config: Dict[str, object], pipeline: ucmpipelinestore.PipelineStore
):
    store_dir = Path(__file__).resolve().parent.parent
    ds3fs_config = copy.deepcopy(config)
    if config.get("device_id", -1) >= 0:
        ds3fs_config |= {"tensor_size": config["shard_size"]}
    pipeline.Stack("Ds3fs", str(store_dir / "ds3fs/libds3fsstore.so"), ds3fs_config)
    _preload_metrics(store_dir)
    pipeline.Stack("Cache", str(store_dir / "cache/libcachestore.so"), config)


def _cache_empty_pipeline_builder(
    config: Dict[str, object], pipeline: ucmpipelinestore.PipelineStore
):
    store_dir = Path(__file__).resolve().parent.parent
    pipeline.Stack("Empty", str(store_dir / "empty/libemptystore.so"), config)
    _preload_metrics(store_dir)
    pipeline.Stack("Cache", str(store_dir / "cache/libcachestore.so"), config)


def _cache_posix_pipeline_builder(
    config: Dict[str, object], pipeline: ucmpipelinestore.PipelineStore
):
    store_dir = Path(__file__).resolve().parent.parent
    posix_config = copy.deepcopy(config)
    if config.get("device_id", -1) >= 0:
        posix_config |= {"tensor_size": config["shard_size"]}
    _preload_metrics(store_dir)
    pipeline.Stack("Posix", str(store_dir / "posix/libposixstore.so"), posix_config)
    pipeline.Stack("Cache", str(store_dir / "cache/libcachestore.so"), config)


def _build_cache_compress_posix_pipeline(
    config: Dict[str, object], pipeline: ucmpipelinestore.PipelineStore
) -> None:
    store_dir = Path(__file__).resolve().parent.parent
    posix_config = copy.deepcopy(config)

    if config.get("device_id", -1) >= 0:
        if (posix_config["block_size"] % posix_config["shard_size"]) != 0:
            print(
                "_build_cache_compress_posix_pipeline: error paraments "
                f"{posix_config['block_size']} {posix_config['shard_size']}"
            )
            return
        layers = posix_config["block_size"] // posix_config["shard_size"]
        posix_config["shard_size"] = (
            (posix_config["shard_size"] * posix_config["compress_ratio"] // 32)
            // 4096
            * 4096
        )
        posix_config["tensor_size"] = int(posix_config["shard_size"])
        posix_config["block_size"] = int(posix_config["shard_size"] * layers)

    _preload_metrics(store_dir)
    pipeline.Stack("Posix", str(store_dir / "posix/libposixstore.so"), posix_config)
    pipeline.Stack("Compress", str(store_dir / "compress/libcompressor.so"), config)
    pipeline.Stack("Cache", str(store_dir / "cache/libcachestore.so"), config)


def _empty_pipeline_builder(
    config: Dict[str, object], pipeline: ucmpipelinestore.PipelineStore
):
    store_dir = Path(__file__).resolve().parent.parent
    pipeline.Stack("Empty", str(store_dir / "empty/libemptystore.so"), config)


def _fake_pipeline_builder(
    config: Dict[str, object], pipeline: ucmpipelinestore.PipelineStore
):
    store_dir = Path(__file__).resolve().parent.parent
    fake_config = copy.deepcopy(config)
    fake_config["share_buffer_enable"] = True
    pipeline.Stack("Fake", str(store_dir / "fake/libfakestore.so"), fake_config)


def _posix_pipeline_builder(
    config: Dict[str, object], pipeline: ucmpipelinestore.PipelineStore
):
    store_dir = Path(__file__).resolve().parent.parent
    _preload_metrics(store_dir)
    pipeline.Stack("Posix", str(store_dir / "posix/libposixstore.so"), config)


def _cache_fake_pipeline_builder(
    config: Dict[str, object], pipeline: ucmpipelinestore.PipelineStore
):
    store_dir = Path(__file__).resolve().parent.parent
    fake_config = copy.deepcopy(config)
    fake_config["share_buffer_enable"] = True
    pipeline.Stack("Fake", str(store_dir / "fake/libfakestore.so"), fake_config)
    _preload_metrics(store_dir)
    pipeline.Stack("Cache", str(store_dir / "cache/libcachestore.so"), config)


def _mooncake_pipeline_builder(
    config: Dict[str, object], pipeline: ucmpipelinestore.PipelineStore
):
    store_dir = Path(__file__).resolve().parent.parent
    pipeline.Stack(
        "Mooncake", str(store_dir / "mooncakestore/libmooncakestore.so"), config
    )


def _mooncake_posix_pipeline_builder(
    config: Dict[str, object], pipeline: ucmpipelinestore.PipelineStore
):
    store_dir = Path(__file__).resolve().parent.parent
    posix_config = copy.deepcopy(config)
    if config.get("device_id", -1) >= 0:
        posix_config |= {"tensor_size": config["shard_size"]}
    _preload_metrics(store_dir)
    pipeline.Stack("Posix", str(store_dir / "posix/libposixstore.so"), posix_config)
    pipeline.Stack(
        "Mooncake", str(store_dir / "mooncakestore/libmooncakestore.so"), config
    )


UcmPipelineStoreBuilder.register("Cache|Ds3fs", _cache_ds3fs_pipeline_builder)
UcmPipelineStoreBuilder.register("Cache|Empty", _cache_empty_pipeline_builder)
UcmPipelineStoreBuilder.register("Cache|Posix", _cache_posix_pipeline_builder)
UcmPipelineStoreBuilder.register("Empty", _empty_pipeline_builder)
UcmPipelineStoreBuilder.register("Fake", _fake_pipeline_builder)
UcmPipelineStoreBuilder.register("Posix", _posix_pipeline_builder)
UcmPipelineStoreBuilder.register(
    "Cache|Compress|Posix", _build_cache_compress_posix_pipeline
)
UcmPipelineStoreBuilder.register("Cache|Fake", _cache_fake_pipeline_builder)
UcmPipelineStoreBuilder.register("Mooncake", _mooncake_pipeline_builder)
UcmPipelineStoreBuilder.register("Mooncake|Posix", _mooncake_posix_pipeline_builder)
