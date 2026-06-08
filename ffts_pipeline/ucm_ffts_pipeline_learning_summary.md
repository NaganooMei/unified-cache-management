# UCM CacheStore 与 H2D FFTS Pipeline 学习总结

## 范围

这份笔记基于当前本地 UCM 代码仓的实读结果，重点覆盖 vLLM 侧 `ucm_connector.py`、Python/C++ `PipelineStore` 边界、CacheStore 的 Load/Dump 数据路径，以及已经落地的 Ascend H2D FFTS pipeline transport。

本次没有在 Ascend 机器上做实机编译或运行验证，所以这里总结的是代码结构、调用链和实现语义，不包含性能结论。

主要阅读文件：

`@ucm/integration/vllm/ucm_connector.py`

`@ucm/store/pipeline/connector.py`

`@ucm/store/pipeline/cpy/pipeline_store.py.cc`

`@ucm/store/cache/cc/cache_store.cc`

`@ucm/store/cache/cc/load_queue.cc`

`@ucm/store/cache/cc/dump_queue.cc`

`@ucm/store/cache/cc/copy_stream.h`

`@ucm/store/cache/cc/trans_buffer.h`

`@ucm/shared/trans/ascend/ascend_h2d_ffts_pipeline.h`

`@ucm/shared/trans/ascend/ascend_h2d_ffts_pipeline.cc`

`@ucm/shared/trans/ascend/ffts_d2d_dispatcher.h`

`@ucm/shared/trans/ascend/ffts_d2d_dispatcher.cc`

`@ucm/shared/trans/ascend/CMakeLists.txt`

`@ucm/store/test/e2e/cache_h2d_ffts_pipeline_function_test.py`

`@ucm/store/test/e2e/cache_h2d_ffts_pipeline_test.py`

## 一句话结论

当前 UCM 已经把 H2D FFTS pipeline 落成了 CacheStore Load 阶段的可选 Ascend transport。它不是新的 StoreV1 语义，也不是 Python connector 里的分支，而是在 CacheStore 的 H2D transfer 层把一个 shard 从“多段 CE H2D scatter”切换成“一次 CE H2D staging + 多段 FFTS SDMA D2D split”。

默认编译已包含 FFTS pipeline 支持，但默认运行路径仍然是 CE。只有运行时配置 `cache_h2d_transport` 为 `ffts_pipeline`，LoadQueue 才会选择 FFTS executor。

## 入口层：ucm_connector.py 做什么

vLLM 集成层的核心职责不是执行拷贝，而是把 vLLM 的 KV cache 结构翻译成 UCM 可以消费的 block、shard 和 device pointer 矩阵。

顶层 `UCMConnector` 会根据配置和 KV cache 类型选择具体 connector，包括 direct、layerwise、CP、lite、mock、hybrid linear attention 等路径。无论具体 connector 是哪一种，最后都会通过 `UcmPipelineStore` 提交 `load_data` 或 `dump_data`。

`KVCacheLayout` 是这里最重要的数据整理层。它把每层 KV cache 的 base pointer、每个 block 的 stride、每个 tensor slice 的大小整理成数组：

- `tensor_size_list` 会传给 CacheStore，表示一个 shard 里每个目标 device pointer 对应的 byte 数。
- `shard_size` 是一个 shard 的连续 host cache buffer 大小。
- `block_size` 是 CacheStore 后端存储一个 block 需要理解的总大小。
- `extract_block_addrs` 根据 vLLM block id 算出目标 KV cache device pointer 矩阵。

direct 模式下，`start_load_kv` 会按 request 组织要加载的 UCM block id 和 vLLM block id，提取一行一 block 的 device pointer 矩阵，然后调用 `self.store.load_data(...)`。`wait_for_save` 则把要持久化的 block 聚合后调用 `self.store.dump_data(...)`。

layerwise 模式下，connector 会按层提交和等待 load，让 load、forward、save 形成层级重叠。但对 CacheStore 来说，输入仍然是 block id、shard index 和 device pointer 矩阵。

## Python 到 C++ 的边界

Python 的 `UcmPipelineStore` 负责把 tensor 或裸地址归一化成 numpy/array buffer，再传给 pybind 暴露的 C++ `PipelineStore`。

C++ `PipelineStore` 做两件关键事：

- 根据 `store_pipeline` 逐层 `Stack` store，比如 `Cache|Empty` 会先 stack EmptyStore，再 stack CacheStore。
- 把 Python 入参转换成 `Detail::TaskDesc`。

`TaskDesc` 里每个元素是一个 `Detail::Shard`。在当前 H2D Load 语义下，可以把一个 shard 理解为：

```text
owner: cache block id
index: shard index
addrs: 这个 shard 最终要写入的 device pointer 列表
```

也就是说，FFTS pipeline 需要的“一个连续 host object 拆到多个 device fragment”这个形状，在进入 CacheStore 前已经由 `TaskDesc` 表达好了，不需要改 Python API。

## CacheStore 的任务主线

CacheStore setup 会读取运行时配置，初始化 BufferManager 和 TransManager。只要 worker 侧有有效 `device_id`，`transEnable_` 就会打开，Load/Dump 会提交给 TransManager。

主调用链可以概括成：

```text
UcmPipelineStore.load_data
  -> PipelineStore.Load
  -> CacheStore.Load
  -> TransManager.Submit
  -> LoadQueue

UcmPipelineStore.dump_data
  -> PipelineStore.Dump
  -> CacheStore.Dump
  -> TransManager.Submit
  -> DumpQueue
```

TransManager 只是任务分发层。真正的 Load H2D 行为在 LoadQueue，Dump D2H 行为在 DumpQueue。

## CacheStore Load：从 backend 到 HBM

LoadQueue 分成两个线程阶段。

Dispatch 阶段从 waiting queue 取 task，逐 shard 申请或查找 `TransBuffer::Handle`。这个 handle 背后是 host cache buffer。若当前 worker 是该 buffer owner 且 buffer 尚未 ready，就先向后端 store 提交一次 backend load，把数据读到 `handle.Data()`。随后把 shard、buffer handle、backend task handle 等信息组成 `ShardTask` 推入 running queue。

Transfer 阶段先根据配置创建 H2D executor，然后消费 running queue。每个 shard 的处理顺序是：

```text
wait backend ready
submit H2D transfer
if not last shard:
  hold ShardTask to keep host buffer alive
else:
  synchronize executor
  clear held shard handles
  mark task waiter done
```

这里的 `holder_` 很关键。它保证最后一次 synchronize 完成前，前面 shard 对应的 host buffer handle 不会被释放。换成 FFTS pipeline 后，这个生命周期要求仍然成立，因为 host-to-staging 和 staging-to-destination 都必须完成后才能释放 host cache buffer 引用。

## CE H2D 路径

默认 `cache_h2d_transport` 是 `ce`。CE executor 使用 `CopyStream` 创建 stream pool，stream 数来自 `cache_stream_number`。提交一个 shard 时，它按 `tensorSizes_` 累加 host offset，然后逐个 slice 调用 `HostToDeviceAsync`：

```text
host buffer + offset0 -> device[0]
host buffer + offset1 -> device[1]
host buffer + offsetN -> device[N]
```

最后一个 shard 到来时，LoadQueue 调用 executor synchronize，CE executor 会同步所有 CopyStream stream。

## FFTS Pipeline H2D 路径

FFTS executor 的输入仍然是一个 shard 的 host pointer、device pointer 列表和 `tensorSizes_`。不同的是它不会逐 fragment 发 H2D，而是把整个 shard 当成一个 pipeline object：

```text
host cache buffer
  -> CE H2D copy to device staging slot
  -> FFTS SDMA D2D split to final device pointers
```

在 UCM 当前实现中，一个 `ShardTask` 就是一个 FFTS pipeline object。object 大小等于 `tensorSizes_` 的总和，fragment 数等于 `tensorSizes_` 的元素个数。

`FftsPipelineH2DTransferExecutor` 会用这些配置初始化 `AscendH2DFftsPipeline`：

- `deviceId`
- `pipelineDepth`
- `maxReadyLanes`
- `objectBytes`
- `maxFragments`

当前 FFTS executor 不使用 CE executor 的 stream pool。也就是说，Load 侧配置为 `ffts_pipeline` 后，`cache_stream_number` 仍会被 CacheStore 校验和打印，也仍会影响 DumpQueue，但 FFTS Load executor 内部是一个 H2D stream、一个 FFTS stream，以及由 `cache_h2d_ffts_pipeline_depth` 决定的 staging slot 数。

## AscendH2DFftsPipeline 内部

`AscendH2DFftsPipeline::Setup` 会做这些事情：

- 设置 device。
- 创建 `h2dStream_` 和 `fftsStream_`。
- 按 `pipelineDepth` 分配 device staging buffer。
- 为每个 slot 创建 `slotReady_` 和 `slotFree_` event。
- 初始化时在 H2D stream 上记录每个 slot 的 free event，让第一个 object 可以直接使用。

提交一个 object 时，流程是：

```text
slot = nextObjectIndex % pipelineDepth
build copy specs from staging offsets to final device pointers
build FFTS D2D descriptors
h2dStream waits slotFree[slot]
aclrtMemcpyAsync host -> staging on h2dStream
record slotReady[slot] on h2dStream
fftsStream waits slotReady[slot]
launch FFTS D2D split on fftsStream
record slotFree[slot] on fftsStream
keep descriptor object alive in inFlight_
```

这个模型的重叠来自 slot 复用约束：同一个 slot 必须等 FFTS split 完成后才能再次被 H2D 覆盖，但不同 slot 之间可以让 H2D staging 和 FFTS split 交错推进。

Synchronize 会让 H2D stream 等所有 slot free event，然后同步 H2D stream 和 FFTS stream，最后清理 `inFlight_`。`inFlight_` 的存在是为了让 FFTS descriptor buffer 在任务完成前保持有效。

## FftsD2DDispatcher 做什么

`FftsD2DDispatcher` 不理解 CacheStore，也不理解 H2D。它只负责把一组 device-to-device copy spec 转成 FFTS Plus SDMA context，并通过 runtime 提交。

它的核心逻辑是：

- 每个 copy spec 生成一个 128 字节的 FFTS Plus communication context。
- SDMA context 里写入 source/destination address、copy size 和 SDMA header。
- `maxReadyLanes` 决定初始 ready context 数。
- fragment 会按 lane 轮转分配，同一 lane 上后一个 context 依赖前一个 context。
- launch 时填充 `rtFftsPlusSqe_t` 和 `rtFftsPlusTaskInfo_t`，然后调用 `rtFftsPlusTaskLaunchWithFlag`。

所以，FFTS pipeline 的真实执行边界在 runtime launch 之后。UCM 侧代码负责构造 descriptor 和提交任务，实际 D2D copy 由 Ascend runtime/FFTS Plus/SDMA 侧执行。

## 编译与运行时开关

顶层 CMake 选项默认开启：

```text
UCM_ENABLE_ASCEND_IO_AGGREGATION=ON
```

开启时，Ascend trans 组件会查找 FFTS header 和 `libruntime`，并把 IO 聚合相关源码加入编译。如果缺少依赖，CMake 会直接失败。需要关闭时可以显式设置 `UCM_ENABLE_ASCEND_IO_AGGREGATION=OFF`。

运行时配置有三个关键项：

```text
cache_h2d_transport: "ce" | "ffts_pipeline"
cache_h2d_ffts_pipeline_depth: default 2
cache_h2d_ffts_max_ready_lanes: default 8
```

CacheStore 会校验：

- `cache_h2d_transport` 只能是 `ce` 或 `ffts_pipeline`。
- 未编译 FFTS pipeline 时配置 `ffts_pipeline` 会 setup 失败。
- FFTS pipeline depth 和 max ready lanes 必须大于 0。
- stream number 仍然必须在合法范围内。

当前实现中，只要编译支持并配置为 `ffts_pipeline`，LoadQueue 就直接选择 FFTS executor，不再根据 fragment 数自动回退 CE。

## Dump 路径保持 CE

DumpQueue 当前仍是 device-to-host gather：

```text
final device pointers
  -> CE D2H gather into host cache buffer
  -> backend dump
```

它支持等待上游 event handle，D2H 后同步 stream，标记 host buffer ready，再提交 backend dump。当前 FFTS pipeline 只覆盖 Load 的 H2D scatter，不覆盖 Dump 的 D2H gather。

## 测试脚本说明

当前有两个 e2e 脚本覆盖 H2D FFTS pipeline。

`@ucm/store/test/e2e/cache_h2d_ffts_pipeline_function_test.py`

这个脚本用 `Cache|Empty` 构造 worker 和 scheduler，先 dump 源 tensor，再通过 `ffts_pipeline` load 到目标 tensor，最后逐 tensor 比对内容。它主要验证功能正确性。

`@ucm/store/test/e2e/cache_h2d_ffts_pipeline_test.py`

这个脚本可以分别跑 CE 和 FFTS pipeline，完成 dump、load、数据校验、warmup、repeat，并打印平均耗时、中位数、最小值和 GB/s。它的计时范围是 `load_data` submit 加 `wait` 完成，包含 CacheStore 业务路径、队列调度、backend/cache ready 等开销，不是裸 H2D memcpy 或 FFTS launch 的纯耗时。

常用环境变量包括：

```text
UCM_FFTS_TORCH_DEVICE
UCM_FFTS_DEVICE_ID
UCM_FFTS_FRAGMENT_COUNT
UCM_FFTS_FRAGMENT_BYTES
UCM_FFTS_TENSOR_SIZES
UCM_FFTS_BLOCK_NUM
UCM_FFTS_WARMUP
UCM_FFTS_REPEAT
UCM_FFTS_PIPELINE_DEPTH
UCM_FFTS_MAX_READY_LANES
UCM_FFTS_CACHE_STREAM_NUMBER
UCM_FFTS_COMPARE_CE
```

## 和 sandbox / Yuanrong 技术模型的关系

当前 UCM 实现吸收的是 Yuanrong-style H2D FFTS pipeline 的核心传输模型，而不是照搬 benchmark case。

相同点是：

- FFTS 不直接做 host-to-device。
- 第一阶段仍是普通 CE H2D，把一个连续 host object 搬到 device staging。
- 第二阶段才用 FFTS Plus SDMA 做 device-to-device split。
- 使用 staging slot、双 stream、event 和 ready lanes 组织 overlap。

不同点是：

- UCM 的 object 来自 CacheStore shard，不来自 benchmark CopyBuffer。
- UCM 的 fragment size 来自 `tensor_size_list`，可以是非等长 slice。
- UCM 的 destination pointer 来自 vLLM KV cache layout 提取出的真实 HBM 地址。
- UCM 的错误处理走 `Status` 和 task failure set，不使用 benchmark 里的断言或直接退出。
- UCM 只接单 worker / 单 device 内的 Load H2D transport，不搬多进程或多卡 benchmark runner 逻辑。

## 后续继续排查或优化时的关注点

第一，区分测量范围。Python e2e 的性能数字是 CacheStore load 业务路径耗时，不等同于 CE H2D、FFTS launch 或 SDMA D2D 的单段耗时。如果要判断 FFTS pipeline 真正收益，需要在 C++ 内部分段打点：backend wait、H2D staging、FFTS launch、FFTS synchronize、queue wait。

第二，关注 `cache_stream_number` 与 FFTS Load 的关系。CE Load 用它控制 stream pool；FFTS Load 当前不使用它，而是靠 pipeline depth 控制 staging slot。对比 CE 时不要把两个参数当成完全同类控制变量。

第三，关注 object 形状。UCM 当前一个 shard 是一个 pipeline object。如果 `tensor_size_list` 很短，FFTS descriptor overhead 可能吃掉收益；如果 fragment 很多，`maxReadyLanes` 和 dependency 链会影响并发形状。

第四，关注内存成本。FFTS pipeline 额外 device staging memory 近似为 `pipelineDepth * sum(tensorSizes)`，这是 Load transfer executor 内部持有的 staging buffer。

第五，关注依赖边界。FFTS pipeline 需要 Ascend FFTS header 和 `libruntime`，未启用编译开关时运行时配置不会静默降级，而是 setup 失败。

第六，Dump 方向仍是 CE。如果后续要做 D2H 侧 pipeline，需要单独设计，不应该把 H2D split 的实现直接反向套用。

## 我的理解模型

可以把当前代码的核心抽象压成三句话：

1. Python connector 负责把 vLLM KV cache 变成 UCM block id、shard index 和 device pointer 矩阵。
2. CacheStore Load 负责把 backend/cache 中的连续 host shard 搬回 HBM。
3. FFTS pipeline 是 Load 阶段的一种 Ascend H2D scatter transport：先 CE H2D 到 staging，再 FFTS SDMA D2D 拆分到最终 KV cache 地址。

这也是当前实现最重要的边界：FFTS pipeline 是 transport，不是新的 cache 语义；是 Load H2D 的可选实现，不是 Dump、backend、Python connector 或 StoreV1 API 的重写。
