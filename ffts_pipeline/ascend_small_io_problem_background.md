# Ascend 小 IO 拷贝问题背景

## 课题背景

当前研究的核心问题是 Ascend 上 UCM CacheStore 在批量小 IO 场景下的 H2D 拷贝效率。这里的小 IO 主要指单个 tensor slice 小于或等于 64K 的场景，尤其是一次 load 里包含大量小 tensor、碎片化 device pointer 的情况。

UCM 原生 CacheStore 的 H2D 路径是按 tensor slice 逐个调用 `aclrtMemcpyAsync`。这种方式在大 IO 下比较直接，但在 64K 以下的小 IO 批量场景里，每次 copy 都有一份相对固定的提交和调度开销。payload 本身很小，固定开销占比就会被放大，导致 PCIe 链路很难被打满。

换句话说，瓶颈不是单个 byte 的搬运能力，而是大量小 copy 的 per-copy overhead。小包太多时，host 侧提交、runtime 调度、stream 任务管理等固定成本会压过 PCIe 的有效吞吐。

## 原生 CacheStore 的问题形态

CacheStore Load 的输入天然是一个连续 host shard 和多个最终 device pointer：

```text
host cache shard
  -> device tensor slice 0
  -> device tensor slice 1
  -> device tensor slice 2
  -> ...
```

原生 CE scatter 路径会按 `tensor_size_list` 拆分这个 host shard，对每个 tensor slice 单独发起一次 H2D copy：

```text
copy 0: host offset 0      -> device[0]
copy 1: host offset size0  -> device[1]
copy 2: host offset size01 -> device[2]
...
```

当每个 slice 都很小时，整体效果就是很多次小 H2D。即使总数据量不小，单次 copy 的粒度太细，也会让固定开销不断重复，最终表现为总耗时偏高、PCIe 带宽利用率偏低。

相关代码背景：

`@ucm/store/cache/cc/load_queue.cc`

`@ucm/store/cache/cc/copy_stream.h`

`@ucm/shared/trans/ascend/ascend_stream.cc`

## 当前重点收益模型

当前最重点关注的收益模型有两类：GQA 模型的 32K/64K IO，以及 DeepSeek-V4/DSV4 的混合 IO。它们共同的特点是：一次 CacheStore load 的总数据量不小，但会被 `tensor_size_list` 展开成很多 fragment；在 CE 模式下，每个 fragment 都对应一次 `aclrtMemcpyAsync` H2D copy。

### GQA / Qwen32B

GQA 模型的 CacheStore 输入形态相对简单。以 Qwen32B 为例，当前重点关注 TP8 和 TP4 两种并行配置，对应 32K IO 和 64K IO 两档。

Qwen32B 有 64 层，每层有 2 个 KV tensor slice。CacheStore 注册时会把它拍平成一维 `tensor_size_list`：

```text
64 layers * 2 tensors per layer = 128 fragments
```

因此，一个 CacheStore shard 会变成 128 个小 IO：

```text
Qwen32B TP8:  128 fragments * 32K
Qwen32B TP4:  128 fragments * 64K
```

如果走原生 CE scatter，每个 row/shard 就是 128 次 async H2D copy。TP8 场景单个 fragment 更小，固定开销占比更高；TP4 场景 fragment 到 64K，但仍然属于这个课题重点关注的小 IO 边界。对 FFTS pipeline 来说，它们都是很典型的收益模型：把 128 次小 H2D 聚合成 1 次较大的 H2D staging，再用 FFTS 做 128 个 D2D split。

这个模型的好处是结构规则，适合作为第一类基准：

- fragment size 固定。
- fragment 数固定为 128。
- row object 形状稳定。
- CE 与 FFTS pipeline 的 copy 次数差异非常直观。

### DeepSeek-V4 / DSV4

DSV4 的 CacheStore 形态更复杂。每个 rank 会创建两个 store，分别对应 FA 和 WA。可以参考现有报告：

`@docs/ascend_dsv4_io_hit_load_report.md`

FA store 负责 full/classical KV 相关 row，WA store 负责 window/compressor state 相关 row。命中 `N` 个 external blocks 时，典型 load 形态是：

```text
FA store: load N rows
WA store: load 1 boundary row
total rows: N + 1
```

如果启用 FFTS pipeline，每个 row 就是一个 pipeline object，所以 DSV4 命中 `N` 个 blocks 时，对应 `N + 1` 个 FFTS pipeline objects。

参考当前报告里的拆分，FA row 和 WA row 的形态是：

```text
FA row:
  fragments = 83
  effective H2D bytes = 3,183,872
  CacheStore shard bytes = 3,186,688

WA row:
  fragments = 127
  effective H2D bytes = 6,496,256
  CacheStore shard bytes = 6,496,256
```

fragment 分布按报告完整口径可以理解为：

```text
FA row:
  21 * 128K
  21 * 16K
  20 * 4K
  21 * 256B

WA row:
  43 * 128K
  42 * 16K
  42 * 4K
```

你补充的 DSV4 重点 IO 口径是 128K、4K、128K、256B。这里可以把它理解成：DSV4 既有相对较大的 128K fragment，也有大量 4K 和 256B 这种更小的 fragment；报告中还展开出 16K 这一档，后续算 copy 数和 payload 时也应该保留，避免漏掉真实 row 形态。

在 CE 模式下，命中 `N` 个 external blocks 时，DSV4 的 H2D copy 数大致是：

```text
128K copy: 21N + 43
16K copy:  21N + 42
4K copy:   20N + 42
256B copy: 21N
total async H2D copy: 83N + 127
```

这就是 DSV4 成为重点收益模型的原因：命中 blocks 越多，FA row 会线性增加；即使命中很少，只要 `N > 0`，WA row 也会带来一整行固定 load。CE 路径会把这些 row 继续拆成大量 fragment copy，而 FFTS pipeline 可以把每个 row 作为 object 聚合，减少跨 PCIe 的小 H2D 次数。

### 两类模型的差异

GQA / Qwen32B 更适合验证“规则小 IO 聚合”的收益：

```text
one row = 128 fragments with same size
```

DSV4 更适合验证“真实复杂 row 形态”的收益：

```text
FA row = 83 mixed fragments
WA row = 127 mixed fragments
one load = N FA rows + 1 WA row
```

因此，Qwen32B TP8/TP4 可以作为 32K/64K 小 IO 的稳定收益模型；DSV4 可以作为 128K、16K、4K、256B 混合 IO 和 FA/WA 双 store 的复杂收益模型。

## 阶段性验证结论与当前矛盾

前期 sandbox 里已经针对 GQA 场景做过更细的验证，包括单卡不同 IO 聚合力度下的带宽，以及多卡同时读时的带宽。当前比较明确的结论是：GQA 场景下，IO 聚合粒度在 1M 到 2M 区间效果比较好。

这个结论和当前 CacheStore 的真实 row/object 形态之间存在几个矛盾。

### GQA 整存整取 object 偏大但仍有收益

在非 layerwise 的 GQA CacheStore 路径里，Qwen32B 的一个 row 通常是完整 64 层拍平后的 `tensor_size_list`：

```text
TP8: 32K * 128 = 4M
TP4: 64K * 128 = 8M
```

这意味着当前 FFTS pipeline 的 object 粒度已经大于 sandbox 验证里效果较好的 1M 到 2M 区间。按直觉看，4M/8M object 可能不是最优聚合粒度，但测试结果仍然明显优于原生 CE copy。这说明收益的第一来源仍然成立：把 128 次小 H2D copy 合成一次大 H2D staging，能显著摊薄 per-copy 固定开销。

但这也提示当前接口不够灵活：object 粒度被 CacheStore shard 绑定了，无法把一个 4M/8M shard 按 1M/2M 的目标粒度拆成多个 pipeline objects。

### GQA layerwise object 太小导致劣化

layerwise 场景的问题相反。layerwise 下一个 row 往往只包含单层的 2 个 KV tensor：

```text
TP8 layerwise: 32K * 2 = 64K
TP4 layerwise: 64K * 2 = 128K
```

这个 object 又太小了。对 FFTS pipeline 来说，64K/128K object 需要先 H2D staging，再构建 FFTS descriptor，再做 D2D split；如果 object 本身太小，这些额外开销很容易超过聚合收益，表现为明显劣化。

这说明 layerwise 路径如果仍然按“一层一个 shard/object”接入 FFTS pipeline，很难得到理想效果。它需要跨 layer、跨 row 或跨 task 做更高层的 object 聚合，否则聚合粒度永远达不到 1M 到 2M。

### DSV4 object 大且收益不明显

DSV4 的 FA/WA row 本身已经比较大：

```text
FA row effective payload: 3,183,872 bytes
WA row effective payload: 6,496,256 bytes
```

这两个 object 都大于 1M 到 2M 的经验较优区间。并且 DSV4 的 fragment 里有不少 128K，相比 32K/64K GQA 小 IO，原生 CE 的 per-copy 固定开销占比没有那么极端。所以 DSV4 上 FFTS pipeline 的提升不明显，是比较符合当前观察的。

另外，WA store 每次只 load 一个 boundary row。即使 WA row 内部很大，当前 pipeline 的 row object 数也只有 1 个，无法形成“前一个 object 的 FFTS split 和后一个 object 的 H2D staging 重叠”。这会让 pipeline depth 的价值很有限。

### 当前核心问题

当前实现把 CacheStore row/shard 直接映射成 FFTS pipeline object：

```text
one ShardTask = one FFTS pipeline object
```

这个映射简单、稳定、侵入小，但它把 object 粒度固定成了 CacheStore 的 row 粒度。实际收益模型却说明，最优 object 粒度可能应该是一个独立的 transport 参数，而不是直接等于 shard size。

## 接口改造思考

我的判断是，当前接口最大的问题不是 FFTS dispatcher，而是缺少一个“transport object builder”。这个 builder 应该负责把 CacheStore 输入的 row/shard 重新组织成适合 H2D staging + FFTS split 的 object。

### 第一类改造：单 shard 内拆分 object

对 GQA 整存整取和 DSV4 FA/WA 这种 shard 偏大的场景，可以先支持在单个 shard 内按目标字节数拆分 object。

新增一个运行时配置，例如：

```text
cache_h2d_ffts_object_target_bytes = 1048576 or 2097152
```

LoadQueue 仍然接收一个 `ShardTask`，但 FFTS executor 不再把整个 shard 一次性提交给 `AscendH2DFftsPipeline::SubmitObject`，而是按 `tensorSizes_` 的累计字节切成多个 object：

```text
shard tensor sizes
  -> object 0 around 1M/2M
  -> object 1 around 1M/2M
  -> object 2 around 1M/2M
```

每个 object 的 host source 是同一个 host shard 内的连续 offset，device pointers 是对应的一段 fragment 子列表。这样不需要改 Python connector、TaskDesc 或 TransBuffer 生命周期，改造边界相对收敛。

这个方向可以验证：

- GQA 4M/8M shard 拆成 1M/2M object 后是否优于当前整 shard object。
- DSV4 FA 3M row 拆成 2 个左右 object 是否有收益。
- DSV4 WA 6.5M row 拆成多个 object 后，是否能在同一个 row 内形成更有效的 pipeline。

### 第二类改造：layerwise 多 shard 聚合暂不推进

对 GQA layerwise 这种 shard 太小的场景，表面上看需要跨 shard 聚合到 1M/2M。但代码路径确认后，这个方向不适合作为当前主攻点。

layerwise 的一个 load task 里，通常是同一层的多个 block：

```text
(block_id_0, layer_id)
(block_id_1, layer_id)
(block_id_2, layer_id)
...
```

也就是说，一个 task 内的不同 shard 对应不同 block id。CacheStore 会按 `(block_id, shard_index)` 去 `TransBuffer` 查找或分配 host cache buffer。一个 block 内部的 shard 可以理解为同一个 block 的不同分片语义，但同一层的不同 block 是不同 cache entry，它们的 host buffer 没有连续性保证。

底层 `TransBuffer` 的数据池本身是连续大 buffer，`DataAt(iNode)` 是按 `nodeSize * iNode` 计算地址；但不同 block id 拿到的 `iNode` 由 hash 查找、已有缓存状态、freeHead 分配、引用计数和 shared buffer 复用共同决定。task 中相邻的 block 不等于 buffer 中相邻的 node，这不是接口语义能保证的事情。

因此，layerwise 不能在现有 CacheStore buffer 接口上直接做零拷贝跨 shard 聚合：

- 不 pack 的情况下，多个 shard 的 host source 不保证连续，不能合成一次大 H2D。
- 如果先在 host 侧新增 pack buffer，把多个 shard 拷到连续 host staging，再做大 H2D，会引入额外 CPU memcpy、host buffer 生命周期、异步 pack 调度和错误处理。
- 如果不做 host pack，而是把多个 shard 多次 H2D 到同一个 device staging object，不会减少跨 PCIe 的小 H2D 次数，只是改变 D2D split 的组织方式，无法解决核心瓶颈。

所以，layerwise 暂时不进行进一步优化。它可以保留为负向对照或边界 case，用来说明 object 太小时 FFTS pipeline 会劣化；但当前不建议为了 layerwise 去改主接口或引入 host pack 方案。

当前主攻方向收敛为：整存整取场景下的单 shard 内拆分，也就是把 GQA full 的 4M/8M shard 或 DSV4 的大 row，按 1M/2M 目标粒度拆成多个 pipeline objects。这个方向不需要跨 shard 聚合，也不依赖不同 block 的 host buffer 连续性。

### 第三类改造：保留 CE fallback 或阈值策略

不是所有 object 都适合 FFTS pipeline。当前观察已经说明，过小 object 会劣化，过大 object 也未必是最优。因此后续接口最好保留阈值策略：

```text
if object bytes < min_pipeline_bytes:
  use CE
elif object bytes > target bytes:
  split into target-sized pipeline objects
else:
  use one FFTS pipeline object
```

这个策略不能只看总 bytes，还要看 fragment 数和 fragment size 分布。比如一个 1M object 如果只有 8 个 128K fragment，收益可能和 128 个 8K fragment 完全不同。

## 先在 e2e 脚本里验证

我赞同先在单测脚本里验证，而不是马上改主接口。当前脚本已经能模拟很多 object 形态：

`@ucm/store/test/e2e/cache_h2d_ffts_pipeline_test.py`

现有能力包括：

- `UCM_FFTS_FRAGMENT_COUNT`
- `UCM_FFTS_FRAGMENT_BYTES`
- `UCM_FFTS_TENSOR_SIZES`
- `UCM_FFTS_BLOCK_NUM`
- `UCM_FFTS_PIPELINE_DEPTH`
- `UCM_FFTS_MAX_READY_LANES`
- `UCM_FFTS_COMPARE_CE`

### 不改脚本即可先跑的矩阵

GQA 当前整存整取形态：

```text
qwen32b_tp8_full:  128 fragments * 32K = 4M
qwen32b_tp4_full:  128 fragments * 64K = 8M
```

GQA 目标 object 粒度模拟：

```text
qwen32b_tp8_1m: 32 fragments * 32K = 1M
qwen32b_tp8_2m: 64 fragments * 32K = 2M
qwen32b_tp4_1m: 16 fragments * 64K = 1M
qwen32b_tp4_2m: 32 fragments * 64K = 2M
```

GQA layerwise 当前形态：

```text
qwen32b_tp8_layerwise: 2 fragments * 32K = 64K
qwen32b_tp4_layerwise: 2 fragments * 64K = 128K
```

这两个 layerwise case 当前只建议作为负向对照：验证 object 太小时 FFTS pipeline 的额外 staging、descriptor 和 D2D split 开销会压过收益，而不是作为下一步接口优化目标。

DSV4 FA / WA 单 row 形态可以用 `UCM_FFTS_TENSOR_SIZES` 显式传入：

```text
dsv4_fa:
  21 * 128K
  21 * 16K
  20 * 4K
  21 * 256B

dsv4_wa:
  43 * 128K
  42 * 16K
  42 * 4K
```

这组矩阵可以先回答两个问题：

- 1M/2M object 在 UCM e2e 业务路径里是否仍然是 GQA 最优区间。
- DSV4 的 FA/WA row 如果按当前整 row object 跑，为什么收益不明显。

### 建议对脚本做的最小增强

为了让验证更稳定，建议先给脚本加几个轻量能力，而不是直接改 CacheStore 接口。

第一，增加 model case preset：

```text
UCM_FFTS_MODEL_CASE=qwen32b_tp8_full
UCM_FFTS_MODEL_CASE=qwen32b_tp4_full
UCM_FFTS_MODEL_CASE=qwen32b_tp8_layerwise
UCM_FFTS_MODEL_CASE=qwen32b_tp4_layerwise
UCM_FFTS_MODEL_CASE=qwen32b_tp8_1m
UCM_FFTS_MODEL_CASE=qwen32b_tp8_2m
UCM_FFTS_MODEL_CASE=qwen32b_tp4_1m
UCM_FFTS_MODEL_CASE=qwen32b_tp4_2m
UCM_FFTS_MODEL_CASE=dsv4_fa
UCM_FFTS_MODEL_CASE=dsv4_wa
```

这样可以避免每次手写很长的 `UCM_FFTS_TENSOR_SIZES`，也能减少实验矩阵出错。

第二，增加 shard size override：

```text
UCM_FFTS_SHARD_SIZE
```

这个主要用于模拟 DSV4 FA row 的 CacheStore 4KB 对齐 padding。H2D effective payload 仍然来自 `tensor_size_list`，但 CacheStore row/backend 口径可以更贴近真实 DSV4。

第三，输出更明确的 object 信息：

```text
case name
fragment count
fragment size histogram
shard bytes
effective tensor bytes
pipeline object bytes
pipeline depth
max ready lanes
```

当前脚本只打印 fragments 和 shard bytes，不足以直接判断是不是跑到了预期 object 形态。

第四，增加结果表模式。单次脚本现在只跑一个 tensor shape。后续可以让脚本按 preset list 连续跑多组 case，并打印一张终端表，字段包括：

```text
case
transport
bytes
avg_ms
median_ms
min_ms
gbps
ffts_vs_ce
```

第五，多卡同时读建议先做 wrapper，而不是塞进单进程测试。每个 device 一个进程，更接近真实多 rank 行为；wrapper 负责同时启动多个 `cache_h2d_ffts_pipeline_test.py`，最后汇总每卡带宽和总带宽。

### 单测验证后的接口决策

如果单测确认 1M/2M 在 UCM e2e 路径里仍然明显更优，那么接口改造优先级可以是：

1. 主攻单 shard 内 object splitting，解决 GQA full 和 DSV4 大 row 的 object 过大问题。
2. 暂停 layerwise 多 shard 聚合方向，因为它需要 host source 连续性或 host-side packing，复杂度和额外开销都比较高。
3. 保留 layerwise 小 object 作为负向对照，用来确认过小 object 不适合 FFTS pipeline。
4. 最后再做自动阈值策略，让 CE、整 row FFTS、split FFTS 可以按 object size 和 fragment 分布选择。

## 解决思路：IO 聚合

我们的核心办法是把“小 IO 批量 scatter”改写成“先聚合成一次较大的 H2D，再在 device 内部拆分”。

目标路径是：

```text
host cache shard
  -> one large H2D copy to device staging buffer
  -> multiple D2D copies to final device tensor slices
```

这样做的关键收益是：跨 PCIe 的 H2D 阶段从很多次小 copy 变成一次较大的 copy。大 copy 的 payload 更充足，固定提交开销被摊薄，更容易接近 PCIe 的有效带宽。

D2D 阶段虽然仍然有多段 split，但它发生在 device 侧，不再占用 PCIe H2D 链路；并且可以用 FFTS 把多段 D2D descriptor 批量提交，进一步减少 host 侧逐个 D2D copy 的开销。

## 为什么需要流水线

如果只做“先大 H2D，再 D2D 拆分”，单个 object 的执行仍然是串行两阶段：

```text
H2D staging
FFTS D2D split
```

这能减少小 H2D 数量，但 H2D 和 D2D 之间仍然可能互相等待。为了缩短总耗时，需要引入 pipeline slot，让不同 object 的阶段重叠。

典型流水线形态是：

```text
slot 0: H2D object 0  -> FFTS split object 0
slot 1:               H2D object 1  -> FFTS split object 1
slot 0:                              H2D object 2  -> FFTS split object 2
```

也就是当 `slot 0` 正在做 FFTS D2D split 时，`slot 1` 可以开始下一批 H2D staging。slot 的复用由 event 保证：同一个 slot 只有在 D2D split 完成后才能再次被 H2D 覆盖。

流水线的意义是把 H2D staging 和 D2D split 的等待关系局部化，让不同 object 之间尽量重叠执行，而不是每个 object 都完整串行。

## 为什么需要 FFTS

IO 聚合后，D2D split 仍然可能包含很多 fragment。如果继续用普通 `aclrtMemcpyAsync` 逐个做 D2D，问题会从“小 H2D 多次提交”部分转移成“小 D2D 多次提交”。

FFTS 的作用是把多个 D2D copy spec 编成一组 SDMA context，通过一次 FFTS Plus task launch 提交给 runtime。这样可以把 host 侧逐个提交 D2D copy 的开销进一步压缩，并利用 ready lanes 和依赖关系组织 device 侧执行。

因此，这里的 FFTS 不是直接做 H2D。它负责第二阶段：

```text
device staging buffer
  -> final device tensor slice 0
  -> final device tensor slice 1
  -> final device tensor slice 2
  -> ...
```

第一阶段 H2D staging 仍然是普通 CE H2D copy，FFTS 负责 device 内部的 D2D split。

相关实现：

`@ucm/shared/trans/ascend/ascend_h2d_ffts_pipeline.cc`

`@ucm/shared/trans/ascend/ffts_d2d_dispatcher.cc`

## 三个技术点的分工

IO 聚合解决的是 PCIe H2D 小包太多的问题：

```text
many small H2D copies -> one large H2D staging copy
```

Pipeline 解决的是阶段间等待和 slot 复用问题：

```text
H2D staging of next object overlaps with FFTS split of previous object
```

FFTS 解决的是多段 D2D split 的提交和调度问题：

```text
many D2D copy intents -> one FFTS Plus launch with many SDMA contexts
```

三者合起来，目标是减少 H2D per-copy 固定开销、提高 PCIe 有效带宽，并用流水线和 FFTS 缩短端到端 load 总耗时。

## 对 UCM CacheStore 的定位

在 UCM 里，这个优化应该被理解成 CacheStore Load 阶段的一种 H2D transport，而不是新的缓存语义。

CacheStore 的 block id、shard index、backend、TransBuffer、wait/check 机制都不需要因为这个课题改变。变化点集中在 LoadQueue 的 H2D transfer executor：默认 CE 路径保持原样，开启 `ffts_pipeline` 后，H2D 阶段换成聚合 staging + FFTS D2D split。

当前方向只覆盖 Load 的 H2D scatter。Dump 的 D2H gather 仍然保持原生 CE 路径；如果后续要优化 D2H，也应该作为另一个方向单独设计。

## 预期收益与需要验证的点

预期收益主要来自三部分：

- H2D copy 次数减少，64K 以下小 IO 的固定开销被摊薄。
- H2D 大 copy 更容易打满 PCIe 带宽。
- D2D split 通过 FFTS 批量提交，并与下一批 H2D staging 流水重叠。

需要重点验证的点包括：

- 64K 以下不同 tensor size、fragment count、block 数下 CE 与 FFTS pipeline 的 load 总耗时。
- `cache_h2d_ffts_pipeline_depth` 对重叠效果和额外 device staging memory 的影响。
- `cache_h2d_ffts_max_ready_lanes` 对 D2D split 并发形态的影响。
- 小 fragment 数场景下 FFTS descriptor 和 staging 是否反而引入额外开销。
- Python e2e 计时与 C++ 内部分段计时的差异，避免把 queue/backend 等开销误判成底层 copy 性能。

## 简化理解

这个课题可以简化成一句话：

```text
把很多 64K 以下的小 H2D，聚合成一次较大的 H2D，再用 FFTS 在 device 内部拆回原来的多个 tensor 地址，并通过流水线让 H2D 和 D2D 尽量重叠。
```

最终目标不是改变 CacheStore 对外行为，而是在不改变 UCM task 语义的前提下，让 Ascend 小 IO 批量 load 更接近硬件链路的有效吞吐。
