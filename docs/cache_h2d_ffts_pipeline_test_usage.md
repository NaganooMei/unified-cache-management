# CacheStore H2D FFTS Pipeline 测试脚本使用说明

## 目标

本文说明如何使用 CacheStore H2D FFTS pipeline e2e 测试脚本。

脚本位置：

`@ucm/store/test/e2e/cache_h2d_ffts_pipeline_test.py`

脚本用于验证两件事：

1. CacheStore load 在 `cache_h2d_transport: "ffts_pipeline"` 配置下能成功走 FFTS pipeline 路径。
2. FFTS pipeline load 的结果正确，并且性能没有明显异常。

## 前提

需要在 Ascend 运行环境中执行，并且 CacheStore 已经以 FFTS pipeline 能力编译：

```bash
UCM_ENABLE_ASCEND_FFTS_PIPELINE=ON
```

运行环境需要能找到：

- CacheStore、EmptyStore、PipelineStore 的 Python 扩展和动态库。
- Ascend runtime。
- FFTS runtime header 对应的运行时库。
- PyTorch 设备后端。

如果使用 NPU 设备，脚本会尝试导入 `torch_npu`。如果当前环境需要手工配置 `PYTHONPATH` 或 `LD_LIBRARY_PATH`，请先按 UCM e2e 测试环境完成配置。

## 默认测试模型

脚本默认构造一个整存取模型：

```text
block_size == shard_size
```

也就是说，一个 load task 里的每个 shard 对应一个完整 cache block。每个 shard 由多段 tensor fragment 组成：

```text
host cache shard
  -> tensor fragment 0
  -> tensor fragment 1
  -> ...
  -> tensor fragment N
```

默认参数：

```text
fragment_count = 128
fragment_bytes = 32768
block_num = 16
warmup = 2
repeat = 10
pipeline_depth = 2
max_ready_lanes = 8
```

默认每次 load 的数据量是：

```text
block_num * fragment_count * fragment_bytes
```

## pip install 安装方式

如果希望用 `pip install` 的方式安装 CacheStore 和 PipelineStore，并同时打开 FFTS pipeline 编译能力，可以在 Ascend 环境中执行：

```bash
cd unified-cache-management

source /usr/local/Ascend/ascend-toolkit/set_env.sh

PLATFORM=ascend \
ENABLE_SPARSE=false \
UCM_ENABLE_ASCEND_FFTS_PIPELINE=1 \
ASCEND_ROOT=/usr/local/Ascend/ascend-toolkit/latest \
pip install -v -e . --no-build-isolation
```

这里 `PLATFORM=ascend` 会让 `setup.py` 传入：

```text
RUNTIME_ENVIRONMENT=ascend
```

这里 `UCM_ENABLE_ASCEND_FFTS_PIPELINE=1` 会让 `setup.py` 传入：

```text
UCM_ENABLE_ASCEND_FFTS_PIPELINE=ON
```

安装完成后，`PipelineStore`、`CacheStore`、`EmptyStore` 等组件会随 Python 包一起可用，测试脚本可以直接 import `UcmPipelineStore`。

## 先验功能

先跑功能脚本，确认 `ffts_pipeline` 路径能正确 load：

```bash
UC_LOGGER_LEVEL=debug \
UCM_FFTS_TORCH_DEVICE=npu \
UCM_FFTS_DEVICE_ID=0 \
python ucm/store/test/e2e/cache_h2d_ffts_pipeline_function_test.py
```

这个脚本风格和 `cache_on_empty_test.py` 接近：在 `main()` 中显式创建 `worker` 和 `scheduler`，然后把它们传给 `e2e_test()`。

功能脚本只做正确性验证：

1. `scheduler.lookup` 确认初始 cache miss。
2. `worker.dump` 把源 tensor 写入 CacheStore。
3. `scheduler.lookup` 确认 cache hit。
4. `worker.load` 通过 `ffts_pipeline` 加载到目标 tensor。
5. `torch.allclose` 对比源 tensor 和目标 tensor。

如果这个脚本失败，先不要看性能，优先修正编译、运行时配置或数据正确性问题。

## 再比性能

在仓库根目录执行：

```bash
python ucm/store/test/e2e/cache_h2d_ffts_pipeline_test.py
```

如果是 Ascend NPU 环境，建议显式指定：

```bash
UCM_FFTS_TORCH_DEVICE=npu \
UCM_FFTS_DEVICE_ID=0 \
python ucm/store/test/e2e/cache_h2d_ffts_pipeline_test.py
```

如果当前环境沿用 CUDA 设备名或 torch 后端兼容 CUDA 设备名，可以使用默认值或显式指定：

```bash
UCM_FFTS_TORCH_DEVICE=cuda \
UCM_FFTS_DEVICE_ID=0 \
python ucm/store/test/e2e/cache_h2d_ffts_pipeline_test.py
```

## 它怎么验证是否走 pipeline

脚本会创建两组 CacheStore worker：

```text
cache_h2d_transport = "ce"
cache_h2d_transport = "ffts_pipeline"
```

当运行到 `"ffts_pipeline"` 组时，配置会传入 CacheStore：

```text
cache_h2d_transport = "ffts_pipeline"
cache_h2d_ffts_pipeline_depth = UCM_FFTS_PIPELINE_DEPTH
cache_h2d_ffts_max_ready_lanes = UCM_FFTS_MAX_READY_LANES
```

当前 CacheStore 逻辑中，只要编译启用了 FFTS pipeline，并且运行时配置为 `"ffts_pipeline"`，LoadQueue 就会选择 FFTS executor，不再根据 fragment 数回退 CE。

如果二进制没有启用 FFTS pipeline，创建 `"ffts_pipeline"` worker 时会失败。这种失败本身说明当前构建没有走到 FFTS pipeline 能力。

建议运行时打开日志确认：

```bash
UC_LOGGER_LEVEL=info
```

日志里应能看到 CacheStore 的 H2D transport 配置为 `ffts_pipeline`。

## 它怎么验证正确性

每组 transport 都会执行：

1. 在 device 上创建源 tensor。
2. 使用 `dump_data` 把源 tensor 写入 CacheStore。
3. 确认 scheduler 能 lookup 到这些 block。
4. 使用 `load_data` 把 CacheStore 数据加载到目标 tensor。
5. 使用 `torch.allclose` 对比源 tensor 和目标 tensor。

如果 load 数据错误，脚本会打印第一个出错位置和差异值，然后断言失败。

## 它怎么判断性能是否正常

脚本测量的是：

```text
load_data submit + wait 完成
```

这个时间覆盖 CacheStore load 的业务路径，包括队列调度、cache hit 等待、H2D 提交和最终同步。它不是单纯的底层 memcpy submit 时间。

输出示例：

```text
ce: bytes=67108864, avg=2.100ms, median=2.050ms, min=1.980ms, bw=31.956GB/s
ffts_pipeline: bytes=67108864, avg=1.700ms, median=1.660ms, min=1.620ms, bw=39.476GB/s
ffts_vs_ce_slowdown=0.810x
```

默认性能 sanity 规则：

```text
ffts_avg_time / ce_avg_time <= UCM_FFTS_MAX_SLOWDOWN
```

默认 `UCM_FFTS_MAX_SLOWDOWN=5.0`，这个阈值比较宽，只用于发现明显异常。正式性能判断建议根据目标机器基线调小。

也可以设置最低带宽：

```bash
UCM_FFTS_MIN_GBPS=20
```

## 常用参数

指定设备：

```bash
UCM_FFTS_TORCH_DEVICE=npu
UCM_FFTS_DEVICE_ID=0
```

指定 fragment 规模：

```bash
UCM_FFTS_FRAGMENT_COUNT=128
UCM_FFTS_FRAGMENT_BYTES=32768
```

直接指定不等长 tensor size list：

```bash
UCM_FFTS_TENSOR_SIZES=32768,4096,32768,4096
```

指定 block 数和重复次数：

```bash
UCM_FFTS_BLOCK_NUM=16
UCM_FFTS_WARMUP=2
UCM_FFTS_REPEAT=10
```

指定 FFTS pipeline 参数：

```bash
UCM_FFTS_PIPELINE_DEPTH=2
UCM_FFTS_MAX_READY_LANES=8
```

指定 CacheStore 运行参数：

```bash
UCM_FFTS_CACHE_STREAM_NUMBER=4
UCM_FFTS_CACHE_BUFFER_CAPACITY_GB=4
UCM_FFTS_LOAD_EXCLUSIVE_BUFFER_NUMBER=64
UCM_FFTS_WAITING_QUEUE_DEPTH=64
UCM_FFTS_RUNNING_QUEUE_DEPTH=4096
UCM_FFTS_TIMEOUT_MS=30000
```

如果遇到类似错误：

```text
too small buffer(4294967296) on shard(4194304)
```

含义是 `cache_buffer_capacity_gb` 能容纳的 shard 数不满足 CacheStore 校验。脚本默认 shard 大小是 4MB，4GB cache buffer 只能容纳 1024 个 shard；如果 `cache_load_exclusive_buffer_number` 使用 CacheStore 默认值 1024，校验会要求至少 2048 个 shard。测试脚本默认把 `UCM_FFTS_LOAD_EXCLUSIVE_BUFFER_NUMBER` 设为 64，所以 4GB buffer 可以通过校验。也可以选择增大 `UCM_FFTS_CACHE_BUFFER_CAPACITY_GB`。

关闭 CE 对比，只测 FFTS pipeline：

```bash
UCM_FFTS_COMPARE_CE=0
```

调整性能 sanity 阈值：

```bash
UCM_FFTS_MAX_SLOWDOWN=2.0
UCM_FFTS_MIN_GBPS=20
```

## 推荐命令

先跑一个小规模正确性测试：

```bash
UC_LOGGER_LEVEL=info \
UCM_FFTS_TORCH_DEVICE=npu \
UCM_FFTS_DEVICE_ID=0 \
UCM_FFTS_BLOCK_NUM=4 \
UCM_FFTS_FRAGMENT_COUNT=16 \
UCM_FFTS_REPEAT=3 \
python ucm/store/test/e2e/cache_h2d_ffts_pipeline_test.py
```

再跑默认性能 sanity：

```bash
UC_LOGGER_LEVEL=info \
UCM_FFTS_TORCH_DEVICE=npu \
UCM_FFTS_DEVICE_ID=0 \
python ucm/store/test/e2e/cache_h2d_ffts_pipeline_test.py
```

做 lanes sweep 时，可以多次运行：

```bash
for lanes in 1 2 4 8 16 32; do
  UC_LOGGER_LEVEL=info \
  UCM_FFTS_TORCH_DEVICE=npu \
  UCM_FFTS_DEVICE_ID=0 \
  UCM_FFTS_MAX_READY_LANES=${lanes} \
  python ucm/store/test/e2e/cache_h2d_ffts_pipeline_test.py
done
```

## 结果解读

如果脚本成功结束，说明：

- CE baseline 正确。
- FFTS pipeline load 正确。
- FFTS pipeline 的业务路径耗时没有超过设定 sanity 阈值。

如果 `"ffts_pipeline"` worker 创建失败，优先检查：

- 是否使用 `UCM_ENABLE_ASCEND_FFTS_PIPELINE=ON` 编译。
- 是否能找到 FFTS runtime 相关库。
- 是否在 Ascend 运行环境执行。

如果正确性失败，优先检查：

- `tensor_size_list` 是否和目标 tensor 指针列表一一对应。
- `fragment_bytes` 是否能被 bf16 元素大小整除。
- CacheStore 日志中是否有 H2D submit 或 synchronize 错误。

如果性能异常，优先调整：

- `UCM_FFTS_BLOCK_NUM`
- `UCM_FFTS_FRAGMENT_COUNT`
- `UCM_FFTS_FRAGMENT_BYTES`
- `UCM_FFTS_PIPELINE_DEPTH`
- `UCM_FFTS_MAX_READY_LANES`

性能数字要结合机器、驱动、Ascend runtime、fragment 大小和并发规模判断，不建议只看单次运行。
