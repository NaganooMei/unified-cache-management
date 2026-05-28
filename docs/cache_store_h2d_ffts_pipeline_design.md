# CacheStore H2D FFTS Pipeline 设计文档

## 文档范围

本文基于当前默认分支最新提交整理：

```text
commit: afc9d34ca4593d1654571a739db8f532fef5cd13
title:  feat: add CacheStore H2D FFTS pipeline
date:   2026-05-28T21:04:24+08:00
author: naganomei <naganomei@noreply.gitcode.com>
```

该提交为 CacheStore Load 阶段增加 Ascend H2D FFTS pipeline 能力。文档描述的是已落地实现，不再是预研草稿。

主要代码入口：

- `@CMakeLists.txt:16`
- `@ucm/shared/trans/ascend/CMakeLists.txt:15`
- `@ucm/store/cache/cc/cache_store.cc:152`
- `@ucm/store/cache/cc/load_queue.cc:41`
- `@ucm/shared/trans/ascend/ascend_h2d_ffts_pipeline.h:18`
- `@ucm/shared/trans/ascend/ffts_d2d_dispatcher.h:24`

## 背景与目标

CacheStore Load 的输入语义已经完整表达了 H2D 拷贝所需信息：

- `TransBuffer::Handle::Data()` 提供连续 host cache buffer。
- `Detail::Shard.addrs` 提供最终写入的 HBM KV cache device pointer 列表。
- `Config::tensorSizes` 提供每个 device pointer 对应的 tensor slice 大小。

原有路径使用 CE scatter：对每个 tensor slice 发起一次 host-to-device 异步拷贝。

新路径引入 Ascend FFTS pipeline：先用 CE 把一个 shard 对应的连续 host object 拷贝到 device staging buffer，再用 FFTS SDMA 在 device 侧拆分到多个最终 device pointer。这样把多段 H2D scatter 变成“一次 H2D staging + 一组 device-to-device split”。

第一版目标：

- 只覆盖 CacheStore Load 的 host-to-device 阶段。
- 保持默认 CE 路径不变。
- 通过显式编译开关和运行时配置启用 FFTS pipeline。
- 不修改 `TaskDesc`、`Shard`、Python `load_data` 和 pybind 入参语义。

非目标：

- 不改造 Dump 的 device-to-host gather。
- 不把 FFTS pipeline 塞进通用 `Trans::Stream::HostToDeviceAsync`。
- 不引入 sandbox benchmark 的 case 注册、统计、validation、runner 等外围逻辑。

## 总体架构

```text
CacheStore::Load
  -> TransManager
    -> LoadQueue::DispatchStage
      -> 取/填充 host cache buffer
      -> 生成 ShardTask
    -> LoadQueue::TransferStage
      -> H2DTransferExecutor
        -> CeH2DTransferExecutor
        -> FftsPipelineH2DTransferExecutor
          -> AscendH2DFftsPipeline
            -> FftsD2DDispatcher
```

LoadQueue 只依赖窄接口 `H2DTransferExecutor`。底层选择 CE 还是 FFTS pipeline，由配置和编译开关决定。

```text
默认路径:
  host cache buffer
    -> CE H2D slice 0 -> final device[0]
    -> CE H2D slice 1 -> final device[1]
    -> CE H2D slice N -> final device[N]

FFTS pipeline:
  host cache buffer
    -> CE H2D object -> device staging slot
    -> FFTS SDMA slice 0 -> final device[0]
    -> FFTS SDMA slice 1 -> final device[1]
    -> FFTS SDMA slice N -> final device[N]
```

## 编译开关与依赖

顶层新增 CMake 开关，默认关闭：

```text
UCM_ENABLE_ASCEND_FFTS_PIPELINE=OFF
```

对应实现：

- `@CMakeLists.txt:16`
- `@ucm/shared/trans/ascend/CMakeLists.txt:15`

开启后，Ascend trans 组件会：

- 查找 `libruntime`。
- 查找 `runtime/rt_ffts_plus.h` 或 `rt_external_ffts.h`。
- 编译 `ascend_h2d_ffts_pipeline.cc` 和 `ffts_d2d_dispatcher.cc`。
- 向依赖方公开 `UCM_ENABLE_ASCEND_FFTS_PIPELINE=1`。

如果开启编译开关但找不到 FFTS header 或 runtime library，CMake 直接失败：

```text
UCM_ENABLE_ASCEND_FFTS_PIPELINE requires FFTS headers and libruntime.
```

默认关闭时不会编译 FFTS 相关源码，并公开 `UCM_ENABLE_ASCEND_FFTS_PIPELINE=0`，确保 CE 路径不依赖 FFTS。

## 运行时配置

CacheStore `Config` 新增 H2D transport 配置：

```text
h2dTransport            default: "ce"
h2dFftsPipelineDepth    default: 2
h2dFftsMaxReadyLanes    default: 8
h2dFftsMinFragments     default: 2
```

对应代码：

- `@ucm/store/cache/cc/global_config.h:43`
- `@ucm/store/cache/cc/cache_store.cc:152`
- `@ucm/store/cache/cc/cache_store.cc:172`

外部配置 key：

```yaml
cache_h2d_transport: "ffts_pipeline"
cache_h2d_ffts_pipeline_depth: 2
cache_h2d_ffts_max_ready_lanes: 8
cache_h2d_ffts_min_fragments: 2
```

校验规则：

- `cache_h2d_transport` 只能是 `"ce"` 或 `"ffts_pipeline"`。
- 如果运行时配置为 `"ffts_pipeline"`，但编译时未启用 `UCM_ENABLE_ASCEND_FFTS_PIPELINE`，`CacheStore::Setup` 返回错误。
- 如果启用 FFTS pipeline，`depth`、`max_ready_lanes`、`min_fragments` 都必须大于 0。
- 当编译支持 FFTS 且配置为 `"ffts_pipeline"`，但 `tensorSizes.size() < h2dFftsMinFragments` 时，LoadQueue 会回退到 CE executor。

## LoadQueue 改造

LoadQueue 新增内部抽象 `H2DTransferExecutor`：

```cpp
class H2DTransferExecutor {
public:
    virtual ~H2DTransferExecutor() = default;
    virtual Status Setup(const Config& config) = 0;
    virtual Status Submit(void* host, void** device) = 0;
    virtual Status Synchronize() = 0;
};
```

对应代码：

- `@ucm/store/cache/cc/load_queue.cc:41`
- `@ucm/store/cache/cc/load_queue.h:40`

### CE executor

`CeH2DTransferExecutor` 保留原有行为：

- `Setup` 创建 `CopyStream`。
- `Submit` 遍历 `tensorSizes`，按 offset 切分 host buffer。
- 每段调用 `HostToDeviceAsync` 拷贝到对应 device pointer。
- `Synchronize` 调用 `CopyStream::Synchronize()`。

对应代码：

- `@ucm/store/cache/cc/load_queue.cc:49`

### FFTS executor

`FftsPipelineH2DTransferExecutor` 只在 `UCM_ENABLE_ASCEND_FFTS_PIPELINE=1` 时编译：

- 计算 `objectBytes = sum(tensorSizes)`。
- 检查 `h2dFftsMaxReadyLanes` 是否能放入 `uint16_t`。
- 用 `deviceId`、`pipelineDepth`、`maxReadyLanes`、`objectBytes`、`maxFragments` 初始化 `AscendH2DFftsPipeline`。
- `Submit` 调用 `pipeline_.SubmitObject(host, device, tensorSizes_)`。
- `Synchronize` 调用 `pipeline_.Synchronize()`。

对应代码：

- `@ucm/store/cache/cc/load_queue.cc:79`
- `@ucm/store/cache/cc/load_queue.cc:115`

### TransferStage 与 TransferOneTask

`TransferStage` 不再直接持有 `CopyStream`，而是根据配置创建 executor：

- `@ucm/store/cache/cc/load_queue.cc:216`
- `@ucm/store/cache/cc/load_queue.cc:230`

`TransferOneTask` 的业务边界保持不变：

1. 如果 task 已失败，直接结束最后一个 waiter。
2. 等待 backend task 完成，确保 host cache buffer ready。
3. 调用 `executor.Submit(...)` 提交 H2D。
4. 如果不是最后一个 shard，把 `ShardTask` 放入 `holder_`，延长 host buffer 引用生命周期。
5. 如果是最后一个 shard，调用 `executor.Synchronize()`。
6. 同步完成后清理 `holder_`，并调用 waiter `Done()`。
7. 任一步失败都写入 `failureSet_`。

对应代码：

- `@ucm/store/cache/cc/load_queue.cc:241`

## AscendH2DFftsPipeline 设计

`AscendH2DFftsPipeline` 是 Ascend 专用 H2D pipeline transport，负责 stream、event、staging buffer 和 in-flight descriptor 生命周期。

接口定义：

- `@ucm/shared/trans/ascend/ascend_h2d_ffts_pipeline.h:18`
- `@ucm/shared/trans/ascend/ascend_h2d_ffts_pipeline.h:26`

核心配置：

```text
deviceId        Ascend device id
pipelineDepth   staging slot 数量
maxReadyLanes   FFTS ready context lane 上限
objectBytes     每个 shard object 的有效字节数
maxFragments    每个 shard 最多拆分的 tensor fragment 数
```

### Setup

`Setup` 做初始化：

1. 清理旧状态。
2. 校验配置。
3. `aclrtSetDevice(deviceId)`。
4. 创建 `h2dStream_` 和 `fftsStream_`。
5. 为每个 pipeline slot 分配 device staging buffer。
6. 为每个 slot 创建 `slotReady` 和 `slotFree` event。
7. 初始 record `slotFree`，表示所有 slot 可用。

对应代码：

- `@ucm/shared/trans/ascend/ascend_h2d_ffts_pipeline.cc:23`

### SubmitObject

一个 `ShardTask` 对应一个 pipeline object。

输入：

```text
host     = task.bufferHandle.Data()
devices  = task.shard.addrs.data()
sizes    = tensorSizes_
```

提交流程：

```text
slot = nextObjectIndex % pipelineDepth
staging = stagingBuffers[slot]

BuildCopySpecs(staging, devices, sizes)
dispatcher.BuildCopies(specs, maxReadyLanes, readyCount)

h2dStream waits slotFree[slot]
aclrtMemcpyAsync(host -> staging, sum(sizes), h2dStream)
record slotReady[slot] on h2dStream

fftsStream waits slotReady[slot]
dispatcher.Launch(fftsStream, readyCount)
record slotFree[slot] on fftsStream

inFlight_.push_back(object)
nextObjectIndex++
```

对应代码：

- `@ucm/shared/trans/ascend/ascend_h2d_ffts_pipeline.cc:85`
- `@ucm/shared/trans/ascend/ascend_h2d_ffts_pipeline.cc:117`

`BuildCopySpecs` 将 staging buffer 按 `tensorSizes` 切分成 FFTS copy spec：

```text
src = staging + offset
dst = devices[i]
size = sizes[i]
```

这里使用 `sum(tensorSizes)` 作为 H2D 有效 object 大小，不默认搬运完整 `shardSize`。这和原 CE scatter 的有效字节语义一致。

### Synchronize

`Synchronize` 保证所有 H2D staging 和 FFTS D2D split 完成：

1. `aclrtSetDevice(deviceId)`。
2. 让 `h2dStream_` wait 所有 `slotFree` event。
3. 同步 `h2dStream_`。
4. 同步 `fftsStream_`。
5. 清理 `inFlight_`。

对应代码：

- `@ucm/shared/trans/ascend/ascend_h2d_ffts_pipeline.cc:166`

### 生命周期管理

实现里有两个独立生命周期：

- staging slot 生命周期由 `slotFree` event 控制，防止 slot 被覆盖。
- FFTS descriptor 生命周期由 `inFlight_` 控制，防止 runtime 仍可能读取 descriptor 时内存被释放。

`InFlightObject` 保存 copy specs 和 dispatcher：

- `@ucm/shared/trans/ascend/ascend_h2d_ffts_pipeline.h:27`

## FftsD2DDispatcher 设计

`FftsD2DDispatcher` 将一组 device-to-device copy spec 编译成 FFTS Plus context，并提交给 runtime。

接口定义：

- `@ucm/shared/trans/ascend/ffts_d2d_dispatcher.h:24`
- `@ucm/shared/trans/ascend/ffts_d2d_dispatcher.h:30`

### BuildCopies

`BuildCopies(copies, maxReadyLanes, readyContextNum)`：

1. 清空旧 context。
2. 校验 copy spec 和 ready lane。
3. `laneCount = min(copies.size(), maxReadyLanes)`。
4. 为每个 copy 创建一个 SDMA context。
5. 按 `i % laneCount` 把同一 lane 上的 context 串成 dependency chain。
6. 返回 `readyContextNum = laneCount`。

对应代码：

- `@ucm/shared/trans/ascend/ffts_d2d_dispatcher.cc:72`

这种设计让前 `readyContextNum` 个 context 作为初始 ready context，其余 context 由 dependency 驱动。

### Launch

`Launch(stream, readyContextNum)`：

1. 构造 `rtFftsPlusSqe_t`。
2. 设置 `totalContextNum`、`readyContextNum`、`preloadContextNum`。
3. 将 `contexts_.data()` 作为 host descriptor buffer。
4. 调用 `rtFftsPlusTaskLaunchWithFlag` 提交到 `fftsStream_`。

对应代码：

- `@ucm/shared/trans/ascend/ffts_d2d_dispatcher.cc:102`

### SDMA context

`BuildSdmaCtx` 把 `dst`、`src` 和 `size` 写入 128-byte `rtFftsPlusSdmaCtx_t`：

- `@ucm/shared/trans/ascend/ffts_d2d_dispatcher.cc:133`

提交前通过 `static_assert` 确认 FFTS context 结构大小为 128 字节，避免 ABI 假设漂移。

## 错误处理

本实现不沿用 sandbox 式 `ASSERT` 或进程退出，而是统一返回 `Status`：

- ACL 调用失败通过 `AclStatus` 转为 `Status`。
- FFTS copy spec、dependency、ready context 参数不合法返回 `InvalidParam`。
- FFTS launch 失败返回 runtime error code。
- LoadQueue 的 submit/synchronize 失败会写入 `failureSet_`，最后唤醒 waiter。

相关代码：

- `@ucm/shared/trans/ascend/ascend_h2d_ffts_pipeline.cc:12`
- `@ucm/shared/trans/ascend/ffts_d2d_dispatcher.cc:33`
- `@ucm/store/cache/cc/load_queue.cc:254`

## 兼容性与行为边界

默认行为：

- 不打开 CMake 开关时，仍然只编译 CE 路径。
- 不配置 `cache_h2d_transport` 时，默认值是 `"ce"`。
- Dump 路径不变。
- CacheStore 对外 task 语义不变。

显式启用 FFTS pipeline：

- 编译时必须提供 FFTS header 和 `libruntime`。
- 运行时必须设置 `cache_h2d_transport: "ffts_pipeline"`。
- tensor fragment 数小于 `cache_h2d_ffts_min_fragments` 时回退 CE。

保留不变的模块：

- `@ucm/store/detail/type/types.h`
- `@ucm/store/pipeline/cpy/pipeline_store.py.cc`
- `@ucm/integration/vllm/ucm_connector.py`

## 验证建议

功能验证：

- 默认 CE 构建可以正常编译，不要求 FFTS 头文件。
- `UCM_ENABLE_ASCEND_FFTS_PIPELINE=OFF` 且配置 `"ffts_pipeline"` 时，`CacheStore::Setup` 明确失败。
- `UCM_ENABLE_ASCEND_FFTS_PIPELINE=ON` 时，缺少 FFTS header 或 `libruntime` 的构建明确失败。
- 对比 CE 与 FFTS pipeline 的 load 后 HBM KV cache 内容。
- 覆盖不等长 `tensor_size_list`。
- 覆盖多 shard task，确认只有最后一个 shard 同步后才 `Done()`。
- 覆盖 backend load 失败、H2D submit 失败、synchronize 失败后的 `failureSet_` 传播。

性能验证：

- 对比 CE scatter 与 FFTS pipeline 的 load H2D 耗时。
- 分别记录 submit 耗时、stream synchronize 耗时、整体 task wait 耗时。
- 扫描 `cache_h2d_ffts_max_ready_lanes`。
- 扫描 fragment 数和 shard 大小。
- 观察 `cache_h2d_ffts_pipeline_depth` 对重叠度和内存占用的影响。

## 风险与关注点

1. `objectBytes` 必须等于有效 tensor 字节数，而不是默认使用 `shardSize`。否则 staging 会搬运 padding，甚至可能破坏后续 offset 语义。
2. FFTS descriptor buffer 必须活到 stream 任务完成之后。当前通过 `inFlight_` 在 `Synchronize()` 后统一清理。
3. staging slot 复用必须依赖 `slotFree` event。不能仅依赖 host 侧 `nextObjectIndex` 轮转。
4. `h2dFftsMaxReadyLanes` 需要控制在 runtime 支持和 `uint16_t` 表达范围内。
5. 当配置 `"ffts_pipeline"` 但 fragment 数小于 `h2dFftsMinFragments` 时，实际会走 CE，这一点需要在日志或调试文档中明确，避免误判性能结果。

## 一句话总结

最新提交把 FFTS pipeline 落成了 CacheStore Load 阶段的可选 Ascend H2D transport：LoadQueue 仍按 shard 组织任务和生命周期，默认 CE 行为不变；启用后每个 shard 先 H2D 到 device staging buffer，再由 FFTS SDMA 按 `tensorSizes` 拆分到最终 KV cache device pointer。
