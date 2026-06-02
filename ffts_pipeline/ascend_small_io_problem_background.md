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
