from __future__ import annotations

import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from ucm.logger import init_logger
from ucm.store.ucmstore_v1 import Task, UcmKVStoreBaseV1

logger = init_logger(__name__)


DEFAULT_METADATA_SERVER = "P2PHANDSHAKE"
DEFAULT_MASTER_SERVER_ADDRESS = "127.0.0.1:50088"
DEFAULT_PROTOCOL = "ascend"
DEFAULT_SCHEDULER_PROTOCOL = "rpc_only"
DEFAULT_DEVICE_NAME = ""
DEFAULT_GLOBAL_SEGMENT_SIZE = 1073741824 * 5
DEFAULT_LOCAL_HOSTNAME = "127.0.0.1"
DEFAULT_LOOKUP_SHARD_INDICES = 0
DEFAULT_MOONCAKE_V1_WORKERS = 4


@dataclass
class MooncakeTaskV1(Task):
    future: Future
    operation: str
    key_count: int


class UcmMooncakeStoreV1(UcmKVStoreBaseV1):

    def __init__(self, config: Dict[str, object]) -> None:
        super().__init__(config)
        try:
            from mooncake.store import MooncakeDistributedStore
        except ImportError as exc:
            raise ImportError(
                "Please install mooncake by following the instructions at "
                "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "
                "to run UcmMooncakeStoreV1."
            ) from exc

        is_scheduler = self._load_config(config)
        self._validate_ascend_runtime()

        self.store = MooncakeDistributedStore()
        self._shutdown = threading.Event()
        self._executor_workers = int(
            config.get("executor_workers", DEFAULT_MOONCAKE_V1_WORKERS)
        )
        self._executor = ThreadPoolExecutor(
            max_workers=self._executor_workers,
            initializer=self._init_worker_context,
        )
        self._registered_buffers: list[int] = []
        if is_scheduler:
            self.device_id = 0
            logger.info(
                f"Mooncake setup detected scheduler path: old Mooncake does not support "
                f"{DEFAULT_SCHEDULER_PROTOCOL}, keep protocol={self.protocol} and set device to npu:{self.device_id}"
            )
            self._set_device_if_needed({"device_id": self.device_id})
        global_segment_size, local_buffer_size = self._build_segment_sizes(
            config, is_scheduler
        )
        ret = self.store.setup(
            self.local_hostname,
            self.metadata_server,
            global_segment_size,
            local_buffer_size,
            self.protocol,
            self.device_name,
            self.master_server_address,
        )
        if ret != 0:
            raise RuntimeError(f"Initialize mooncake failed with ret={ret}.")
        logger.info(
            f"Mooncake setup success: role={'scheduler' if is_scheduler else 'worker'}, "
            f"protocol={self.protocol}, global_segment_size={global_segment_size}, "
            f"local_buffer_size={local_buffer_size}"
        )
        self._register_buffers()

    def _load_config(self, config: Dict[str, object]) -> bool:
        self.device_id = config.get("device_id", None)
        is_scheduler = self.device_id is None
        self.protocol = str(config.get("protocol") or DEFAULT_PROTOCOL)

        self.tensor_size_list = tuple(
            int(size) for size in config.get("tensor_size_list", ())
        )
        self.register_buffer_ptrs = tuple(
            int(ptr) for ptr in config.get("register_buffer_ptrs", ())
        )
        self.register_buffer_sizes = tuple(
            int(size) for size in config.get("register_buffer_sizes", ())
        )
        if len(self.register_buffer_ptrs) != len(self.register_buffer_sizes):
            raise ValueError(
                "register_buffer_ptrs and register_buffer_sizes must have the same length."
            )

        self.local_hostname = str(
            config.get("local_hostname") or DEFAULT_LOCAL_HOSTNAME
        )
        self.metadata_server = str(
            config.get("metadata_server") or DEFAULT_METADATA_SERVER
        )
        self.master_server_address = str(
            config.get("master_server_address") or DEFAULT_MASTER_SERVER_ADDRESS
        )
        self.device_name = str(config.get("device_name") or DEFAULT_DEVICE_NAME)
        self.lookup_shard_indices = config.get(
            "lookup_shard_indices", DEFAULT_LOOKUP_SHARD_INDICES
        )
        return is_scheduler

    def _validate_ascend_runtime(self) -> None:
        if self.protocol != "ascend":
            raise RuntimeError(
                f"UcmMooncakeStoreV1 only supports Ascend protocol, but got protocol='{self.protocol}'."
            )
        if not hasattr(torch, "npu"):
            raise RuntimeError(
                "UcmMooncakeStoreV1 only supports Ascend platform, but torch.npu is not available."
            )

    def _build_segment_sizes(
        self, config: Dict[str, object], is_scheduler: bool
    ) -> tuple[int, int]:
        if is_scheduler:
            return 0, 0
        global_segment_size = self._parse_size_to_bytes(
            config.get("global_segment_size", DEFAULT_GLOBAL_SEGMENT_SIZE),
            "global_segment_size",
        )
        local_buffer_size = self._parse_size_to_bytes(
            config.get("local_buffer_size", DEFAULT_GLOBAL_SEGMENT_SIZE),
            "local_buffer_size",
        )
        return global_segment_size, local_buffer_size

    def _parse_size_to_bytes(self, value: object, field_name: str) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            cleaned = value.strip().upper()
            match = re.fullmatch(r"(\d+)\s*(MB|GB)", cleaned)
            if match:
                number = int(match.group(1))
                unit = match.group(2)
                multiplier = 1024 * 1024 if unit == "MB" else 1024 * 1024 * 1024
                return number * multiplier
            if cleaned.isdigit():
                return int(cleaned)
        raise ValueError(
            f"Invalid {field_name} value '{value}'. "
            "Use integer bytes or strings like '512MB'/'5GB'."
        )

    def cc_store(self) -> int:
        return 0

    def lookup(self, block_ids: List[bytes]) -> List[bool]:
        if not block_ids:
            logger.info("Mooncake lookup skipped: empty block_ids")
            return []

        keys = self._build_lookup_keys(block_ids)
        exists = self.store.batch_is_exist(keys)
        if len(exists) != len(keys):
            logger.error(
                f"Mooncake lookup result size mismatch: results={len(exists)}, keys={len(keys)}"
            )
            raise RuntimeError(
                f"Mooncake lookup returned {len(exists)} results for {len(keys)} keys."
            )

        hits = [int(value) == 1 for value in exists]
        return hits

    def lookup_on_prefix(self, block_ids: List[bytes]) -> int:
        for idx, exists in enumerate(self.lookup(block_ids)):
            if not exists:
                return idx - 1
        return len(block_ids) - 1

    def prefetch(self, block_ids: List[bytes]) -> None:
        del block_ids

    def load(self, block_ids, shard_index, dst_tensor) -> Task:
        addrs = self._tensor_normalize(dst_tensor)
        return self.load_data(block_ids, shard_index, addrs)

    def dump(self, block_ids, shard_index, src_tensor) -> Task:
        addrs = self._tensor_normalize(src_tensor)
        return self.dump_data(block_ids, shard_index, addrs)

    def load_data(self, block_ids, shard_index, dst_addr) -> Task:
        if isinstance(dst_addr, np.ndarray):
            addrs = dst_addr
        else:
            addrs = np.array(dst_addr, dtype=np.uint64)
        keys = self._build_task_keys(block_ids, shard_index)
        future = self._executor.submit(self._execute_load, keys, addrs)
        return MooncakeTaskV1(future=future, operation="load_data", key_count=len(keys))

    def dump_data(
        self, block_ids, shard_index, src_addr, prerequisite_handle=0
    ) -> Task:
        if isinstance(src_addr, np.ndarray):
            addrs = src_addr
        else:
            addrs = np.array(src_addr, dtype=np.uint64)
        keys = self._build_task_keys(block_ids, shard_index)
        future = self._executor.submit(
            self._execute_dump, keys, addrs, prerequisite_handle
        )
        return MooncakeTaskV1(future=future, operation="dump_data", key_count=len(keys))

    def wait(self, task: Task) -> None:
        if not isinstance(task, MooncakeTaskV1):
            raise TypeError(f"Unsupported Mooncake task type: {type(task)}")
        try:
            task.future.result()
        except Exception as exc:
            raise RuntimeError(
                f"Mooncake {task.operation} task failed for {task.key_count} key(s): {exc}"
            ) from exc

    def check(self, task: Task) -> bool:
        if not isinstance(task, MooncakeTaskV1):
            return False
        return task.future.done()

    def shutdown(self) -> None:
        if self._shutdown.is_set():
            return
        self._shutdown.set()
        self._executor.shutdown(wait=True, cancel_futures=False)
        for buffer_ptr in self._registered_buffers:
            try:
                self.store.unregister_buffer(buffer_ptr)
            except Exception as exc:
                logger.warning(
                    f"Mooncake unregister_buffer failed for ptr={buffer_ptr}: {exc}"
                )
        self._registered_buffers.clear()
        self.store.close()

    def _build_lookup_keys(self, block_ids: List[bytes]) -> List[str]:
        keys = []
        for block_id in block_ids:
            block_hex = bytes(block_id).hex()
            keys.append(f"{block_hex}:0")
        return keys

    def _build_task_keys(
        self, block_ids: List[bytes], shard_index: List[int]
    ) -> List[str]:
        return [
            f"{bytes(block_id).hex()}:{int(shard)}"
            for block_id, shard in zip(block_ids, shard_index)
        ]

    def _build_sizes_matrix(self, n_rows: int) -> List[List[int]]:
        return [list(self.tensor_size_list) for _ in range(n_rows)]

    def _tensor_normalize(self, tensors: List[List[torch.Tensor]]) -> np.ndarray:
        if not tensors:
            return np.empty((0, 0), dtype=np.uint64)

        return np.asarray(
            [[tensor.data_ptr() for tensor in row] for row in tensors],
            dtype=np.uint64,
        )

    def _set_device_context(self) -> None:
        if self.protocol == "ascend" and hasattr(torch, "npu"):
            torch.npu.set_device(torch.device(f"npu:{int(self.device_id)}"))

    def _init_worker_context(self) -> None:
        # Set thread-local device context once when worker starts.
        self._set_device_context()

    def _execute_load(self, keys: List[str], addrs: np.ndarray) -> None:
        addrs_list = addrs.tolist()
        sizes = self._build_sizes_matrix(len(addrs))
        self.store.batch_get_into_multi_buffers(keys, addrs_list, sizes)

    def _execute_dump(
        self,
        keys: List[str],
        addrs: np.ndarray,
        prerequisite_handle: int,
    ) -> None:
        if (
            prerequisite_handle != 0
            and self.protocol == "ascend"
            and hasattr(torch, "npu")
        ):
            torch.npu.synchronize()
        addrs_list = addrs.tolist()
        sizes = self._build_sizes_matrix(len(addrs))
        self.store.batch_put_from_multi_buffers(keys, addrs_list, sizes)

    def _set_device_if_needed(self, config: Dict[str, object]) -> None:
        if not hasattr(torch, "npu"):
            return
        if self.protocol != "ascend":
            return

        device_id = config.get("device_id", self.device_id)
        torch.npu.set_device(torch.device(f"npu:{int(device_id)}"))
        logger.info(f"Set NPU device to npu:{int(device_id)} before Mooncake setup")

    def _register_buffers(self) -> None:
        if not self.register_buffer_ptrs:
            return

        seen_ptrs: set[int] = set()
        for buffer_ptr, buffer_size in zip(
            self.register_buffer_ptrs,
            self.register_buffer_sizes,
        ):
            if buffer_ptr in seen_ptrs:
                continue
            seen_ptrs.add(buffer_ptr)
            ret = self.store.register_buffer(buffer_ptr, buffer_size)
            if ret != 0:
                raise RuntimeError(
                    "Mooncake buffer registration failed for "
                    f"ptr={buffer_ptr}, size={buffer_size}, ret={ret}."
                )
            self._registered_buffers.append(buffer_ptr)

        logger.info(
            f"Mooncake registered worker buffers: count={len(self._registered_buffers)}"
        )
