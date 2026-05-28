# CacheStore H2D FFTS Pipeline 对接方案

## 目标

本文给出一个更干净的 CacheStore 对接 H2D FFTS pipeline 方案。

核心目标不是把 sandbox 的 benchmark 代码搬进 UCM，而是把 FFTS pipeline 的核心能力抽成 CacheStore Load 阶段可选的 H2D transport：

```text
CacheStore Load shard
  host cache buffer
  final device pointer list
  tensor size list

H2D FFTS pipeline transport
  H2D CE: host cache buffer -> device staging buffer
  FFTS SDMA: device staging buffer -> final device pointer list
```

第一版只接 CacheStore Load，也就是 host cache buffer 到 HBM KV cache 的 H2D。Dump 仍然走现有 CE gather。

## 非目标

不要照搬 sandbox 这些部分：

- benchmark case 注册、命令行参数、结果统计。
- `CopyBuffer` / `FragmentedDeviceCopyBuffer` 抽象。
- 多卡 fork runner。
- benchmark validation 和 sweep 脚本。
- host-direct FFTS 实验路径。
- Dump 的 D2H 反向 pipeline。

sandbox 在这里仅作为核心接口参考：怎样用两个 stream、slot event、device staging buffer 和 FFTS dispatcher 完成 H2D staging + D2D split。

## 当前 CacheStore 可以复用的语义

CacheStore 的 Load 已经把业务语义表达完整：

```text
Detail::Shard.owner
  block id，用来定位 cache/backend 中的 block

Detail::Shard.index
  shard index，direct 模式通常是 0，layerwise 模式是 layer id

Detail::Shard.addrs
  这个 shard 要写入的最终 HBM device pointer 列表

LoadQueue::tensorSizes_
  每个 device pointer 对应的 tensor slice 大小

TransBuffer::Handle::Data()
  这个 shard 的连续 host cache buffer 起始地址
```

当前 CE 路径是：

```text
host = bufferHandle.Data()
device = shard.addrs.data()

for i in tensorSizes_:
  pHost = host + offset
  pDevice = device[i]
  aclrtMemcpyAsync(pDevice, pHost, tensorSizes_[i], H2D)
  offset += tensorSizes_[i]
```

FFTS pipeline 不需要改 `TaskDesc`、`Shard`、Python `load_data`、pybind `MakeTaskDesc`。要改的是 LoadQueue 的 H2D 执行方式。

相关 CacheStore 文件：

- `@unified-cache-management/ucm/store/cache/cc/load_queue.h`
- `@unified-cache-management/ucm/store/cache/cc/load_queue.cc`
- `@unified-cache-management/ucm/store/cache/cc/trans_buffer.h`
- `@unified-cache-management/ucm/store/cache/cc/trans_buffer.cc`
- `@unified-cache-management/ucm/store/cache/cc/global_config.h`
- `@unified-cache-management/ucm/shared/trans/stream.h`
- `@unified-cache-management/ucm/shared/trans/ascend/ascend_stream.cc`

## sandbox 只参考哪些核心接口

参考 sandbox 的 H2D FFTS pipeline 只看这两层。

第一层是 pipeline 编排：

- `@dev-sandbox-upstream-yuanrong/module/copy/ascend/h2d_ffts_pipeline/copy_instance_h2d_ffts_pipeline_ascend.h`

核心调用关系是：

```text
SubmitObject
  slot = objectIndex % pipelineDepth
  h2dStream waits slotFree[slot]
  aclrtMemcpyAsync(host object -> device staging slot, H2D, h2dStream)
  record slotReady[slot] on h2dStream
  fftsStream waits slotReady[slot]
  BuildObjectCopies(staging slot -> final fragments)
  dispatcher.BuildCopies(copySpecs)
  dispatcher.Launch(fftsStream, readyCount)
  record slotFree[slot] on fftsStream
```

第二层是 FFTS dispatcher：

- `@dev-sandbox-upstream-yuanrong/module/copy/ascend/h2d_ffts_pipeline/ffts_d2d_dispatcher_ascend.h`

核心接口是：

```text
AscendFftsCopySpec {
  dst
  src
  size
}

FftsD2DDispatcher::BuildCopies(copySpecs)
  -> 为每个 copy spec 构造 128-byte FFTS SDMA context
  -> 按 ready lanes 建 dependency
  -> 返回 readyContextNum

FftsD2DDispatcher::Launch(stream, readyContextNum)
  -> rtFftsPlusTaskLaunchWithFlag(task, stream, 0)
```

UCM 里应该复用这个“接口形状”，但要重写成 UCM 风格：

- 返回 `Status`，不要 `ASSERT` / `exit`。
- 配置来自 CacheStore config，不依赖 benchmark env。
- descriptor 和 staging buffer 生命周期由 transport 对象管理。
- 不依赖 sandbox `CopyBuffer`。

## 推荐总体设计

新增一个 Ascend 专用 H2D pipeline transport，放在 UCM 的 trans/ascend 层：

```text
@unified-cache-management/ucm/shared/trans/ascend/ascend_h2d_ffts_pipeline.h
@unified-cache-management/ucm/shared/trans/ascend/ascend_h2d_ffts_pipeline.cc
@unified-cache-management/ucm/shared/trans/ascend/ffts_d2d_dispatcher.h
@unified-cache-management/ucm/shared/trans/ascend/ffts_d2d_dispatcher.cc
```

然后在 CacheStore LoadQueue 的 H2D 阶段按配置选择：

```text
默认:
  CE scatter

配置启用:
  H2D FFTS pipeline scatter
```

不要把 FFTS pipeline 做成 `Trans::Stream::HostToDeviceAsync` 的隐藏实现。原因是 `HostToDeviceAsync` 的粒度是一段 host 到一个 device pointer，而 FFTS pipeline 的收益来自“一段 host object 先 H2D 到 staging，再一次性 FFTS split 到多个 device fragments”。它需要 object、slot、event、staging buffer、dispatcher descriptor 这些概念，超出了普通 stream copy 抽象。

## 新接口建议

### H2D transport 抽象

可以先在 CacheStore 内部定义一个很窄的 H2D executor，而不是改全局 `Trans::Stream`：

```cpp
class H2DTransferExecutor {
public:
    virtual ~H2DTransferExecutor() = default;
    virtual Status Submit(void* host, void** devices,
                          const std::vector<size_t>& sizes) = 0;
    virtual Status Synchronize() = 0;
};
```

CE 实现：

```text
CeScatterExecutor
  owns CopyStream
  Submit(host, devices, sizes)
    for each size:
      stream.NextStream()->HostToDeviceAsync(host + offset, devices[i], size)
  Synchronize()
    CopyStream::Synchronize()
```

FFTS 实现：

```text
FftsPipelineExecutor
  owns AscendH2DFftsPipeline
  Submit(host, devices, sizes)
    pipeline.SubmitObject(host, devices, sizes)
  Synchronize()
    pipeline.Synchronize()
```

这样 `LoadQueue::TransferOneTask` 的主体逻辑不用关心底层是 CE 还是 FFTS：

```text
WaitBackendTaskReady(task)
executor.Submit(task.bufferHandle.Data(), task.shard.addrs.data(), tensorSizes_)

if not last shard:
  holder_.push_back(task)
  return

executor.Synchronize()
holder_.clear()
waiter->Done()
```

### AscendH2DFftsPipeline 接口

建议核心接口：

```cpp
struct H2DFftsPipelineConfig {
    int32_t deviceId;
    size_t pipelineDepth;
    size_t maxReadyLanes;
    size_t objectBytes;
    size_t maxFragments;
};

class AscendH2DFftsPipeline {
public:
    Status Setup(const H2DFftsPipelineConfig& config);
    Status SubmitObject(void* host, void** devices,
                        const std::vector<size_t>& sizes);
    Status Synchronize();
};
```

其中：

```text
objectBytes = sum(tensorSizes_)
maxFragments = tensorSizes_.size()
```

这里不要用 `shardSize` 直接当 H2D staging size。当前 CE scatter 只拷贝 `tensorSizes_` 累加出来的有效 tensor bytes。FFTS staging 也应该拷贝 `sum(tensorSizes_)`，避免把 shard padding 也搬到 staging buffer。

## FFTS pipeline 内部流程

### Setup

`AscendH2DFftsPipeline::Setup` 做这些事：

```text
aclrtSetDevice(deviceId)
aclrtCreateStream(h2dStream)
aclrtCreateStream(fftsStream)

for slot in pipelineDepth:
  aclrtMalloc staging device buffer，大小 objectBytes
  aclrtCreateEvent(slotReady[slot])
  aclrtCreateEvent(slotFree[slot])
  aclrtRecordEvent(slotFree[slot], h2dStream)
```

`slotFree` 初始化为已完成，这样前几个 object 可以直接使用空 slot。

### SubmitObject

一个 CacheStore `ShardTask` 就是第一版的一个 pipeline object。

输入：

```text
host
  task.bufferHandle.Data()

devices
  task.shard.addrs.data()

sizes
  tensorSizes_
```

提交流程：

```text
slot = nextObjectIndex % pipelineDepth
staging = stagingBuffers[slot]

h2dStream waits slotFree[slot]

aclrtMemcpyAsync(
  staging,
  objectBytes,
  host,
  objectBytes,
  ACL_MEMCPY_HOST_TO_DEVICE,
  h2dStream
)

record slotReady[slot] on h2dStream
fftsStream waits slotReady[slot]

offset = 0
for each i in sizes:
  copySpec.src = staging + offset
  copySpec.dst = devices[i]
  copySpec.size = sizes[i]
  offset += sizes[i]

readyCount = dispatcher.BuildCopies(copySpecs)
dispatcher.Launch(fftsStream, readyCount)

record slotFree[slot] on fftsStream
```

这相当于把当前 CE scatter：

```text
host + offset -> device[i]
```

替换为：

```text
host -> staging
staging + offset -> device[i] by FFTS SDMA
```

### Synchronize

`Synchronize` 要保证两件事：

```text
所有 H2D staging 完成
所有 FFTS D2D split 完成
```

可以让 `h2dStream` wait 所有 `slotFree`，再 synchronize `h2dStream`：

```text
for slot:
  aclrtStreamWaitEvent(h2dStream, slotFree[slot])
aclrtSynchronizeStream(h2dStream)
```

也可以直接同步 `h2dStream` 和 `fftsStream`。第一版建议保守一些，同步两个 stream 或者用一个明确的 end event，避免误判完成边界。

同步完成后再清理 in-flight descriptor holder。

## descriptor 生命周期要单独处理

FFTS dispatcher 的 `Launch` 会把 `contexts_.data()` 作为 `descBuf` 传给 runtime。不要把 dispatcher 做成 `SubmitObject` 里的栈对象。

建议每次 submit 创建一个 in-flight 记录：

```cpp
struct InFlightObject {
    size_t slot;
    std::vector<AscendFftsCopySpec> specs;
    FftsD2DDispatcher dispatcher;
};
```

这些 in-flight objects 保存在 `AscendH2DFftsPipeline` 内部，直到 `Synchronize()` 完成后统一清理。

这样可以保证：

```text
FFTS descriptor buffer 在 device/runtime 可能读取期间仍然活着
```

staging slot 的复用由 `slotFree` event 保证；descriptor 内存的复用由 in-flight holder 保证。两者不要混在一起。

## LoadQueue 改造点

### Config

在 CacheStore `Config` 增加：

```text
h2dTransport
  默认 "ce"
  可选 "ffts_pipeline"

h2dFftsPipelineDepth
  默认 2

h2dFftsMaxReadyLanes
  默认 8

h2dFftsMinFragments
  默认 2，可选；fragment 太少时回退 CE
```

对应 Python 配置可以是：

```yaml
cache_h2d_transport: "ffts_pipeline"
cache_h2d_ffts_pipeline_depth: 2
cache_h2d_ffts_max_ready_lanes: 8
cache_h2d_ffts_min_fragments: 2
```

如果显式启用 `ffts_pipeline` 但编译时没有 FFTS runtime/header，应在 `Setup` 返回错误，不要静默降级。默认 `ce` 则不依赖 FFTS。

### LoadQueue::TransferStage

当前 `TransferStage` 创建 `CopyStream`。

建议改成：

```text
CreateH2DTransferExecutor(config)
  if config.h2dTransport == ce:
    CeScatterExecutor
  if config.h2dTransport == ffts_pipeline:
    FftsPipelineExecutor

running_.ConsumerLoop(stop_, &LoadQueue::TransferOneTask, this, executor)
```

### LoadQueue::TransferOneTask

当前核心逻辑保留：

```text
failureSet_ 检查
WaitBackendTaskReady
提交 H2D
非最后 shard 放入 holder_
最后 shard synchronize
失败写 failureSet_
最后 shard waiter->Done()
```

只把：

```text
HostToDeviceScatterAsync(...)
stream.Synchronize()
```

替换成：

```text
executor.Submit(...)
executor.Synchronize()
```

`holder_` 继续保留，FFTS pipeline 同样需要它。原因是 H2D staging 是异步的，`bufferHandle.Data()` 在 pipeline 完成前不能释放引用。

## 和现有 CE 路径的对应关系

```text
现有 CE:
  host cache buffer
    -> H2D slice 0 -> device[0]
    -> H2D slice 1 -> device[1]
    -> H2D slice 2 -> device[2]

FFTS pipeline:
  host cache buffer
    -> one H2D object -> staging buffer
    -> FFTS slice 0 -> device[0]
    -> FFTS slice 1 -> device[1]
    -> FFTS slice 2 -> device[2]
```

第一版 object 粒度：

```text
一个 ShardTask = 一个 pipeline object
```

后续如果要提高吞吐，可以再考虑把多个 shard 合并成一个更大的 pipeline object，但这会改变 holder、失败边界和 task wait 语义，不建议第一版做。

## 编译和依赖

FFTS 支持应保持可选：

```text
默认构建:
  不要求 FFTS headers
  CE 路径正常编译

启用 FFTS pipeline 构建:
  检测 runtime/rt_ffts_plus.h 或对应外部头
  检测 libruntime
  编译 ascend_h2d_ffts_pipeline 和 ffts_d2d_dispatcher
```

如果没有 FFTS 能力：

```text
cache_h2d_transport 未配置或为 ce:
  正常运行

cache_h2d_transport = ffts_pipeline:
  CacheStore::Setup 返回 InvalidParam 或 NotSupported
```

## 错误处理要求

sandbox 的 `ASSERT` / `ASCEND_ASSERT` 适合 benchmark，不适合 CacheStore。

UCM 实现要改成：

```text
acl/rt 调用失败 -> Status{ret, ...}
FFTS BuildCopies 失败 -> Status::InvalidParam 或 Status::Error
Launch 失败 -> Status
SubmitObject 失败 -> Status
Synchronize 失败 -> Status
```

LoadQueue 收到失败后：

```text
failureSet_->Insert(taskHandle)
如果当前 shard 挂 waiter，则 waiter->Done()
```

保持和当前 CE 路径一致。

## 验证计划

第一阶段只验证功能等价：

```text
CE 默认路径不变
未启用 FFTS 时不引入新依赖
启用 FFTS 但缺少 runtime 时 Setup 明确失败
Cache|Fake 或 Cache|Empty 下做小尺寸 load
对比 HBM 目标 KV block 内容
覆盖 tensorSizes_ 不等长场景
覆盖多 shard task，确认最后一个 shard 才 Done
覆盖 timeout/failureSet 路径
```

第二阶段再验证性能：

```text
记录 CE scatter 的 load H2D 时间
记录 FFTS pipeline 的 load H2D 时间
区分 submit time、stream synchronize time、业务 load_data 到 wait time
扫描 maxReadyLanes
扫描 tensor fragment 数
扫描 block/shard 大小
```

性能验证不要放进第一版功能 patch。

## 主要风险

### 1. objectBytes 和 shardSize 混淆

CE 路径按 `tensorSizes_` 累加拷贝。FFTS staging 也应该按 `sum(tensorSizes_)` 拷贝。不要默认搬完整 `shardSize`，除非确认 `shardSize == sum(tensorSizes_)`。

### 2. descriptor 生命周期

FFTS `descBuf` 不能指向短生命周期栈内存。每个 in-flight object 的 dispatcher/context buffer 要活到 synchronize 之后。

### 3. staging slot 复用

slot 复用只能由 `slotFree` 保证。不要在 host 侧仅凭 objectIndex 轮转就覆盖 staging buffer。

### 4. raw stream 抽象

现有 `Trans::Stream` 不暴露 raw `aclrtStream`，也没有 grouped object copy 语义。第一版不要强行塞进 `HostToDeviceAsync`。

### 5. Dump 不对称

Load 是 host -> device，适合 H2D staging + FFTS D2D split。Dump 是 device -> host gather，方向不同，第一版保持 CE。

## 推荐落地步骤

第一步：增加配置和编译开关，但默认 CE。

第二步：新增 UCM 风格 `FftsD2DDispatcher`，接口返回 `Status`。

第三步：新增 `AscendH2DFftsPipeline`，完成 stream、event、staging buffer、SubmitObject、Synchronize。

第四步：给 LoadQueue 增加 `H2DTransferExecutor`，先实现 CE executor，确保重构后 CE 行为不变。

第五步：接入 FFTS executor，并只在 `cache_h2d_transport=ffts_pipeline` 时启用。

第六步：做小尺寸正确性验证，再做性能验证。

## 最小代码改动范围

预计新增：

- `@unified-cache-management/ucm/shared/trans/ascend/ffts_d2d_dispatcher.h`
- `@unified-cache-management/ucm/shared/trans/ascend/ffts_d2d_dispatcher.cc`
- `@unified-cache-management/ucm/shared/trans/ascend/ascend_h2d_ffts_pipeline.h`
- `@unified-cache-management/ucm/shared/trans/ascend/ascend_h2d_ffts_pipeline.cc`

预计修改：

- `@unified-cache-management/ucm/store/cache/cc/global_config.h`
- `@unified-cache-management/ucm/store/cache/cc/cache_store.cc`
- `@unified-cache-management/ucm/store/cache/cc/load_queue.h`
- `@unified-cache-management/ucm/store/cache/cc/load_queue.cc`
- `@unified-cache-management/ucm/store/cache/CMakeLists.txt`
- `@unified-cache-management/ucm/shared/trans/ascend/CMakeLists.txt`

不建议修改：

- `@unified-cache-management/ucm/store/detail/type/types.h`
- `@unified-cache-management/ucm/store/pipeline/cpy/pipeline_store.py.cc`
- `@unified-cache-management/ucm/integration/vllm/ucm_connector.py`

因为现有 `TaskDesc`、`Shard.addrs` 和 `tensor_size_list` 已经足够表达 FFTS pipeline 需要的输入。

## 一句话方案

把 FFTS pipeline 做成 CacheStore Load 阶段的 Ascend 专用 H2D transport：输入仍然是 `handle.Data() + shard.addrs + tensorSizes_`，内部先用 CE 把一个 shard 搬到 device staging buffer，再用 FFTS SDMA split 到最终 KV cache device pointers；LoadQueue 的任务、wait、failure、holder 生命周期保持不变。
