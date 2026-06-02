# CacheStore H2D FFTS Pipeline Shard 拆分方案设计

## 一句话结论

当前最值得主攻的方向不是 layerwise 跨 shard 聚合，而是整存整取场景下的单 shard 内部拆分。

也就是保持 CacheStore 对外语义不变，仍然让一个 CacheStore shard 对应一个连续 host cache buffer，但在 H2D FFTS pipeline transport 内部，把这个大 shard 按 1M 到 2M 左右的目标粒度拆成多个 pipeline object：

```text
one CacheStore shard
  -> object 0 around target bytes
  -> object 1 around target bytes
  -> object 2 around target bytes
  -> ...
```

每个 object 仍然走同一套路径：

```text
host shard subrange
  -> one CE H2D staging copy
  -> one FFTS D2D split launch
  -> final device tensor slices
```

这样可以同时解决两个问题：GQA full-cache 的 4M/8M object 偏大，以及 DSV4 WA 只有一个 row 时 pipeline object 数不足的问题。layerwise 因为不同 block id 的 host buffer 不保证连续，暂时不做跨 shard 聚合。

## 背景与当前问题

当前 UCM CacheStore 的 H2D FFTS pipeline 已经落在 Load 阶段，核心边界是 transport，而不是 StoreV1 API 或 Python connector。

当前路径可以概括为：

```text
PipelineStore.Load
  -> CacheStore.Load
  -> TransManager.Submit
  -> LoadQueue
  -> H2DTransferExecutor
```

默认 CE executor 会按 `tensor_size_list` 逐个 fragment 发起 H2D：

```text
host + offset0 -> device[0]
host + offset1 -> device[1]
...
```

FFTS pipeline executor 当前把整份 `tensor_size_list` 合成一个 object：

```text
host shard
  -> device staging slot
  -> FFTS split to device[0..N]
```

也就是说，现在的映射关系是：

```text
one ShardTask = one FFTS pipeline object
```

这个映射简单稳定，但 object 粒度完全被 CacheStore shard size 绑死。前期 sandbox 验证显示，GQA 场景中 1M 到 2M 的 IO 聚合粒度更理想；而真实 Qwen32B full-cache 的 shard 是：

```text
TP8: 32K * 128 = 4M
TP4: 64K * 128 = 8M
```

4M/8M 整 shard object 虽然已经明显优于原生 CE scatter，但不一定是最优粒度。shard 拆分的目标就是让 transport object 粒度从 CacheStore row 粒度中解耦出来。

## 设计目标

第一，保持 UCM 对外接口不变。

不改 `StoreV1`，不改 `PipelineStore.Load` 的 Python/C++ 绑定，不改 vLLM 侧 `ucm_connector.py` 的 block/shard/device pointer 组织方式。

第二，只在 H2D FFTS pipeline transport 内部改变 object 粒度。

CacheStore 仍然认为自己在 load 一个 shard。backend 仍然把一个 shard 读到一个连续 host cache buffer。变化只发生在 `FftsPipelineH2DTransferExecutor` 内部：它收到一个 host shard 后，不再一次性提交整 shard object，而是按拆分计划提交多个 sub-object。

第三，只做单 shard 内部拆分，不做跨 shard 聚合。

单 shard 内部的 host buffer 是连续的，可以直接把 `host + offset` 作为 sub-object 的 H2D source。跨 shard 聚合需要假设多个 shard 的 host buffer 连续，这在 layerwise 的不同 block id 场景下不成立；如果额外做 host-side packing，又会引入 CPU memcpy、staging buffer 生命周期和异步调度复杂度。

第四，Dump 方向保持不变。

当前课题只覆盖 Load H2D scatter 优化。Dump 的 D2H gather 继续走原生 CE 路径。

## 核心方案

### 新增拆分策略

建议新增一个运行时配置：

```text
cache_h2d_ffts_object_target_bytes
```

语义：

```text
0:
  关闭 shard 内部拆分，保持当前 one shard one object 行为。

1048576:
  尽量按 1M object 拆分。

2097152:
  尽量按 2M object 拆分。
```

为了保持兼容，默认值建议为 `0`。这样合入后不会改变现有 `ffts_pipeline` 行为，只有显式打开 target bytes 才会启用 shard split。

### 拆分粒度

第一阶段建议只按 fragment 边界拆分，不切开单个 tensor fragment。

输入：

```text
tensor_sizes = [s0, s1, s2, ...]
device_addrs = [d0, d1, d2, ...]
host_base    = handle.Data()
```

输出：

```text
object plan:
  object 0: host_offset, first_fragment, fragment_count, object_bytes
  object 1: host_offset, first_fragment, fragment_count, object_bytes
  ...
```

拆分算法建议用贪心策略：

```text
current object starts empty
for each fragment:
  if current object is not empty and adding this fragment would exceed target:
    close current object
    start a new object
  add fragment to current object
close final object
```

边界规则：

- 每个 object 至少包含一个 fragment。
- 如果单个 fragment 本身大于 target，第一阶段直接让它单独成为一个 object。
- object bytes 只是尽量接近 target，不强求精确等于 target。
- `tensor_sizes` 是 CacheStore store 级固定配置，所以 object plan 可以在 executor setup 时预计算。

### Submit 行为

拆分前：

```text
Submit(host, devices):
  pipeline.SubmitObject(host, devices, tensor_sizes)
```

拆分后：

```text
Submit(host, devices):
  for object in object_plan:
    sub_host = host + object.host_offset
    sub_devices = devices + object.first_fragment
    sub_sizes = tensor_sizes[object.first_fragment : object.end_fragment]
    pipeline.SubmitObject(sub_host, sub_devices, sub_sizes)
```

这个改造仍然复用现有 `AscendH2DFftsPipeline`：

- 每个 sub-object 都是一段连续 host source。
- 每个 sub-object 都对应一段 device pointer 子列表。
- 每个 sub-object 内部继续由 FFTS 做 D2D split。
- 多个 sub-object 顺序提交给同一个 pipeline，依靠已有 slot/event 机制形成 overlap。

### Pipeline setup 行为

当前 pipeline setup 使用整 shard 的 `sum(tensor_sizes)` 作为 `objectBytes`，并按整 shard fragment 数设置 `maxFragments`。

启用拆分后，应改为按拆分计划设置：

```text
pipeline.objectBytes = max(object.object_bytes for object in object_plan)
pipeline.maxFragments = max(object.fragment_count for object in object_plan)
```

这样有两个好处：

- staging buffer 从 `pipelineDepth * shard_bytes` 降到 `pipelineDepth * max_object_bytes`。
- FFTS dispatcher 的单次 max fragment 上限更贴近真实 sub-object，而不是整 shard。

以 Qwen32B 为例：

```text
TP8 full:
  no split:  pipeline slot objectBytes = 4M
  target 2M: pipeline slot objectBytes = 2M
  target 1M: pipeline slot objectBytes = 1M

TP4 full:
  no split:  pipeline slot objectBytes = 8M
  target 2M: pipeline slot objectBytes = 2M
  target 1M: pipeline slot objectBytes = 1M
```

### 生命周期与同步

当前 `LoadQueue` 的 host buffer 生命周期由 `holder_` 保证。非最后一个 shard 提交 H2D 后会被放进 holder，直到最后一个 shard 触发 executor synchronize 后再统一释放。

shard 内部拆分不改变这个约束：

- 同一个 shard 的多个 sub-object 都在 `TransferOneTask` 内提交。
- 对非最后一个 shard，提交完成后 shard task 仍会进入 holder，host buffer 不会提前释放。
- 对最后一个 shard，提交完成后立刻 synchronize，host buffer 在同步完成前仍然活在当前栈对象中。

因此第一阶段不需要改 LoadQueue 的 task waiter 和 holder 语义。

## 为什么不做 layerwise 跨 shard 聚合

layerwise 场景下，一个 task 里通常是同一层的多个 block id：

```text
(block_id_0, layer_id)
(block_id_1, layer_id)
(block_id_2, layer_id)
...
```

这些 shard 对应不同 block id。CacheStore 的 host buffer 来自 TransBuffer 的 block/shard 查询和分配，不同 block id 在 host buffer 池里不保证相邻。

所以不能把它们直接合成一次大 H2D：

```text
host shard A + host shard B + host shard C
```

除非额外做 host-side pack：

```text
non-contiguous host shards
  -> CPU memcpy pack into contiguous host staging
  -> one large H2D
  -> D2D split
```

但这样会引入新的 CPU 拷贝、host staging buffer 管理、pack 调度和错误恢复逻辑，而且对 64K/128K layerwise object 来说收益不确定，复杂度明显偏高。

因此当前结论是：layerwise 暂时只保留为负向对照和边界 case，不作为这轮接口改造目标。

## 对 GQA 的预期

Qwen32B TP8：

```text
full shard = 128 * 32K = 4M

target 2M:
  64 fragments per object
  2 objects per shard

target 1M:
  32 fragments per object
  4 objects per shard
```

Qwen32B TP4：

```text
full shard = 128 * 64K = 8M

target 2M:
  32 fragments per object
  4 objects per shard

target 1M:
  16 fragments per object
  8 objects per shard
```

预期收益点：

- 比整 shard object 更接近 sandbox 里 1M 到 2M 的较优聚合区间。
- 单 shard 内部也能形成 pipeline object 序列，`block_num=1` 时也能观察到 H2D staging 和 FFTS split 的重叠。
- staging memory 明显下降，尤其是 TP4 从 8M per slot 降到 1M/2M per slot。

需要警惕的点：

- object 变多会增加 FFTS launch 次数。
- target 过小会让 descriptor 构建和 launch 开销重新变重。
- TP8/TP4 的最优 target 不一定相同，不能只用一个 case 下结论。

## 对 DSV4 的预期

DSV4 的 FA/WA row 本身已经比较大：

```text
FA row: about 3.18M effective payload
WA row: about 6.5M effective payload
```

因此整 row object 的收益不明显是符合预期的。shard 拆分后可以验证两个问题：

第一，FA/WA 大 row 拆到 1M/2M 后，是否能比整 row object 更好。

第二，WA store 每次只 load 一个 row，之前无法跨 row 形成 pipeline；拆分后，一个 WA row 内部会形成多个 sub-object，理论上可以在 row 内部建立 pipeline overlap。

但 DSV4 的预期收益要比 GQA 谨慎：

- DSV4 有不少 128K fragment，原生 CE 的 per-copy 固定开销占比没有 32K/64K GQA 那么极端。
- DSV4 fragment size 混合，按 fragment 边界贪心拆分后，object bytes 不一定非常均匀。
- WA 虽然能在 row 内部拆出多个 object，但 FFTS launch 次数也会增加，需要实测判断。

## 单测脚本验证路径

建议继续先在 e2e 脚本验证，而不是马上改主接口。

当前重点脚本：

`@ucm/store/test/e2e/cache_h2d_ffts_pipeline_test.py`

### 第一阶段：用现有脚本验证 object 粒度

可以先通过自定义 `tensor_size_list` 或新增 model case preset，模拟 1M/2M object 的表现。

建议补齐这些 case：

```text
qwen32b_tp8_full:
  128 * 32K = 4M

qwen32b_tp8_2m:
  64 * 32K = 2M

qwen32b_tp8_1m:
  32 * 32K = 1M

qwen32b_tp4_full:
  128 * 64K = 8M

qwen32b_tp4_2m:
  32 * 64K = 2M

qwen32b_tp4_1m:
  16 * 64K = 1M
```

这一步不等价于真实 full shard split，因为 CacheStore shard size 也变小了；但它可以先回答一个问题：在 UCM e2e 路径里，1M/2M object 是否仍然是较优区域。

### 第二阶段：在脚本里验证真实 shard split

主代码支持 `cache_h2d_ffts_object_target_bytes` 后，脚本应增加对应环境变量：

```text
UCM_FFTS_OBJECT_TARGET_BYTES
```

脚本把它透传到 CacheStore 配置：

```text
cache_h2d_ffts_object_target_bytes = UCM_FFTS_OBJECT_TARGET_BYTES
```

然后用 full shard case 直接验证真实拆分：

```text
qwen32b_tp8_full + target 0
qwen32b_tp8_full + target 1M
qwen32b_tp8_full + target 2M

qwen32b_tp4_full + target 0
qwen32b_tp4_full + target 1M
qwen32b_tp4_full + target 2M
```

这一步才是真正回答：同一个 4M/8M CacheStore shard，在保持 backend/cache row 语义不变的情况下，transport 内部拆分是否比整 shard object 更好。

### 建议输出字段

脚本 summary 建议增加这些字段：

```text
object_target
objects_per_shard
max_object_bytes
max_object_fragments
pipeline_depth
max_ready_lanes
```

这样一眼能看出实际跑的是整 shard object，还是 1M/2M split object。

### 推荐实验矩阵

先单卡，少变量：

```text
case:
  qwen32b_tp8_full
  qwen32b_tp4_full

transport:
  ce
  ffts_pipeline target 0
  ffts_pipeline target 1M
  ffts_pipeline target 2M

block_num:
  1
  16

pipeline_depth:
  2

max_ready_lanes:
  8
```

`block_num=1` 用来观察单 shard 内部拆分是否能建立 overlap；`block_num=16` 用来对齐当前 baseline 的真实批量 load 形态。

等单卡结论稳定后，再做多卡同时读：

```text
device_count:
  2
  4
  8

case:
  qwen32b_tp8_full target 1M/2M
  qwen32b_tp4_full target 1M/2M
```

多卡阶段重点看总带宽是否被 PCIe/host/cache backend 侧打满，以及 target 从 1M 到 2M 是否仍然稳定。

## 主代码改造建议

### 配置层

在 CacheStore config 中增加：

```text
cache_h2d_ffts_object_target_bytes
```

内部字段：

```text
h2dFftsObjectTargetBytes
```

校验建议：

- 只有 `cache_h2d_transport=ffts_pipeline` 时生效。
- `0` 表示关闭拆分。
- 非 0 时必须大于等于一个保守下限，例如 64K 或 128K。
- 非 0 时不要求大于最大 fragment；如果最大 fragment 超过 target，该 fragment 单独成为 object。

### Executor 层

在 `FftsPipelineH2DTransferExecutor` 中新增 object plan。

推荐内部结构：

```text
FftsObjectPlanItem:
  hostOffset
  firstFragment
  fragmentCount
  objectBytes
```

setup 时：

```text
tensorSizes_ = config.tensorSizes
objectPlan_ = BuildObjectPlan(tensorSizes_, config.h2dFftsObjectTargetBytes)
pipelineConfig.objectBytes = max object bytes in plan
pipelineConfig.maxFragments = max fragment count in plan
pipeline.Setup(pipelineConfig)
```

submit 时：

```text
for each item in objectPlan_:
  pipeline.SubmitObject(host + item.hostOffset,
                        device + item.firstFragment,
                        sizes for this item)
```

为了避免每个 object 都分配新的 `std::vector<size_t>`，可以考虑给 `AscendH2DFftsPipeline::SubmitObject` 增加一个 view 形态的入参；如果当前 C++ 标准不方便使用 `std::span`，也可以先用指针加长度的轻量接口。

### Pipeline 层

`AscendH2DFftsPipeline` 现有语义基本够用，不需要理解 CacheStore shard。

需要确认两点：

- `objectBytes_` 是 staging slot 的最大容量，不要求每次 submit 的 object bytes 都等于它。
- `maxFragments_` 是单次 submit 的 fragment 上限，不需要等于整 shard fragment 数。

因此 pipeline 层可以只做很小的接口适配，甚至第一版不改 pipeline 层，只在 executor 中构造 sub vector 后复用现有接口。

### 日志与观测

建议在 setup 阶段打印拆分计划摘要：

```text
H2D FFTS object target bytes
objects per shard
max object bytes
max object fragments
```

不要每个 shard 都打印 object plan，避免性能测试日志污染。需要详细排查时再加 debug 级别日志。

## 自动 fallback 策略

第一版建议只做显式 target，不做复杂自动选择。

后续可以再加策略：

```text
if transport != ffts_pipeline:
  use CE
elif target == 0:
  use whole-shard FFTS pipeline
elif shard_bytes < min_pipeline_bytes:
  use CE
else:
  use split FFTS pipeline
```

`min_pipeline_bytes` 不应该只看总 bytes，也要结合 fragment 数和 fragment 分布。例如：

```text
64K object with 2 fragments:
  likely CE is better

1M object with 128 tiny fragments:
  likely FFTS pipeline is better
```

这部分建议等 Qwen32B TP8/TP4 和 DSV4 的 target sweep 数据稳定后再做。

## 风险点

第一，FFTS launch 次数增加。

拆分让 H2D object 更接近最优聚合粒度，但每个 object 都有 descriptor 构建和 FFTS launch 成本。target 过小会重新放大固定开销。

第二，fragment 边界拆分只是近似 target。

对于 GQA 等长 32K/64K fragment，1M/2M 很规整；对于 DSV4 混合 fragment，object bytes 可能不均匀。

第三，in-flight descriptor 生命周期会变长或变多。

一个 shard 被拆成多个 object 后，`inFlight_` 里的 dispatcher object 数会增加，直到 synchronize 才释放。需要关注 repeat 压测下 descriptor 内存是否稳定。

第四，测试计时范围不是裸 copy。

当前 e2e 脚本统计的是 `load_data + wait` 的端到端 CacheStore load 耗时，包括 queue、backend/cache ready、Python/C++ 绑定等开销。解释带宽时要避免把它等同于单段 H2D memcpy 带宽。

第五，多卡同时读可能暴露新的瓶颈。

单卡 target 最优不一定等于多卡最优。多卡阶段要同时观察单卡带宽和总带宽，防止局部优化把瓶颈推到 host/backend/PCIe 拓扑上。

## 推荐推进顺序

第一步，补测试脚本 preset 和输出字段。

先确认 1M/2M object 粒度在 UCM e2e 里仍然表现好，避免把 sandbox 结论直接搬进主代码。

第二步，实现显式 target 的单 shard 内部拆分。

只改 CacheStore config 和 FFTS executor，不动 Python connector、TaskDesc、backend、Dump。

第三步，用 full shard case 做真实 A/B。

重点比较：

```text
CE
whole-shard FFTS pipeline
split FFTS pipeline target 1M
split FFTS pipeline target 2M
```

第四步，再扩展到 DSV4 FA/WA。

DSV4 重点看 FA/WA 大 row 拆分是否有收益，以及 WA 单 row 是否能通过 row 内 sub-object pipeline 获得额外 overlap。

第五步，最后才考虑自动阈值和 CE fallback。

自动策略应该由数据驱动，而不是提前把阈值写死。

## 当前倾向

我当前倾向的第一版实现是：

```text
cache_h2d_ffts_object_target_bytes = 0 by default

target == 0:
  one shard one object, preserve current behavior

target > 0:
  split one shard by tensor fragment boundary
  submit multiple objects to the existing AscendH2DFftsPipeline
  setup staging slot by max sub-object bytes
```

这版改造足够小，验证路径清晰，并且直接对准当前最有收益希望的整存整取 GQA full-cache 场景。layerwise 暂时不碰，DSV4 作为第二优先级验证目标。

## 参考代码

`@ucm/store/cache/cc/load_queue.cc`

`@ucm/store/cache/cc/load_queue.h`

`@ucm/store/cache/cc/cache_store.cc`

`@ucm/store/cache/cc/global_config.h`

`@ucm/shared/trans/ascend/ascend_h2d_ffts_pipeline.cc`

`@ucm/shared/trans/ascend/ascend_h2d_ffts_pipeline.h`

`@ucm/store/test/e2e/cache_h2d_ffts_pipeline_test.py`

