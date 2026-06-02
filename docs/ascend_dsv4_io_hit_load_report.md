# Ascend DSV4 KV Cache 分组、命中与 Load IO 分析

## 结论摘要

这份文档解释 vLLM-Ascend + UCM FAWA 路径下，DeepSeek-V4/DSV4 的 KV cache group 如何映射成 FA/WA store，以及命中 `N` 个 512-token canonical blocks 时，CacheStore 一次 load 会产生多少 H2D IO。

核心结论：

| 主题 | 结论 |
| --- | --- |
| 命中 block 粒度 | UCM canonical hash block 是 512 original tokens。命中统计里的 block 指这个 512-token canonical block。 |
| `token_block_size` | 某个 KV group 的一个 logical block 在 original-token 坐标上覆盖的跨度。它不是 CacheStore byte block，也不是一次 load 的 token 数。 |
| `tensor_block_size` | 该 KV group 注册到 device 上的 tensor view，在 token 维度上的物理 slot 数。 |
| slot 换算 | `slots_to_copy = tensor_block_size * logical_tokens_to_copy / token_block_size`。 |
| FA load | 外部命中 `N` 个 blocks 时，FA store load `N` 行。 |
| WA load | 外部命中 `N > 0` 时，WA store 只 load 最后一个 boundary row。 |
| CE copy 数 | 当前日志为 `H2DTransport to ce`，所以 async H2D copy 数为 `83N + 127`。 |
| 有效 H2D bytes | `N * 3,183,872 + 6,496,256`。 |
| Cache row bytes | `N * 3,186,688 + 6,496,256`，FA row 有 4KB 对齐 padding。 |

如果 `external_hit_blocks = 0`，FA/WA load 都不会提交。

## 代码入口

| 代码入口 | 作用 |
| --- | --- |
| `@ucm/integration/vllm/hma_connector.py` | FAWA group 分类、`token_block_size` 计算、tensor layout、FA/WA load/dump 指针生成。 |
| `@ucm/integration/vllm/ucm_connector.py` | 在 vLLM connector 中选择 FAWA connector。 |
| `@ucm/store/pipeline/connector.py` | Python 侧 `load_data`/`dump_data` 包装。 |
| `@ucm/store/pipeline/cpy/pipeline_store.py.cc` | 把 Python 的 keys/indexes/addrs 转成 C++ `TaskDesc`。 |
| `@ucm/store/cache/cc/cache_store.cc` | CacheStore `Load`/`Dump` 入口。 |
| `@ucm/store/cache/cc/trans_manager.h` | 把 load/dump task 分发到 LoadQueue/DumpQueue。 |
| `@ucm/store/cache/cc/load_queue.cc` | CacheStore H2D load 队列；CE 模式下按 fragment 提交 async H2D copy。 |

## 概念表

| 概念 | 含义 | 在本文里的值/用法 |
| --- | --- | --- |
| original token | 原始请求 token 坐标。 | `token_block_size` 的坐标系。 |
| canonical block | UCM 外部缓存命中的 hash block。 | DSV4 Ascend 路径下是 512 original tokens。 |
| KV group | vLLM-Ascend 暴露的一组 KV/state tensor。 | 当前日志中 group 0/3/8 是 FA，group 1/2/4/5/6/7/9/10 是 WA。 |
| FA row | FA store 中一条 canonical block 对应的 full/classical KV tensor slices。 | 命中 `N` 个 canonical blocks 就 load `N` 行 FA row。 |
| WA row | WA store 中一个 boundary 对应的 window/compressor state tensor slices。 | 命中 `N > 0` 时只 load 最后 1 行 WA row。 |
| CacheStore row/shard | CacheStore 传输任务里的一个 `Shard`。 | 一个 row 有一个 key、一个 shard index 和一组 device addresses。 |
| fragment | 一个 row 内的单个 tensor slice。 | CE 模式下每个 fragment 对应一次 async H2D copy。 |
| `tensor_size_list` | 一个 CacheStore row 内所有 fragment 的 byte size 列表。 | 决定 row 会拆成多少次 H2D copy。 |

## `token_block_size` 和 `tensor_block_size`

### 两把尺子

| 字段 | 回答的问题 | 常见误解 |
| --- | --- | --- |
| `token_block_size` | 这个 group 的一个 logical block 覆盖多少 original tokens？ | 误以为它是 CacheStore block size，或者这次 load 的 token 数。 |
| `tensor_block_size` | 这个 group 的一个 tensor block 在 device tensor 里有多少 physical slots？ | 误以为它也是 original-token 数。 |

换算公式：

```text
slots_to_copy =
    tensor_block_size * logical_tokens_to_copy / token_block_size

physical_offset =
    logical_token_offset * tensor_block_size / token_block_size
```

这里的 `logical_tokens_to_copy` 是这次要恢复的 original-token span。对 FA row 来说，一行对应 1 个 canonical block，所以通常是 512 original tokens。

### FA group 的 slot 和 byte 计算

| group | `token_block_size` | `tensor_block_size` | 1 个 512-token hit 对应 slots | 每 slot bytes | 每 view bytes | view 数 | row 贡献 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 512 | 128 | `128 * 512 / 512 = 128` | 1024 | `128 * 1024 = 131072` | 21 | `21 x 131072` |
| 3 | 4096 | 1024 | `1024 * 512 / 4096 = 128` | 128 | `128 * 128 = 16384` | 21 | `21 x 16384` |
| 3 | 4096 | 1024 | `1024 * 512 / 4096 = 128` | 2 | `128 * 2 = 256` | 21 | `21 x 256` |
| 8 | 16384 | 128 | `128 * 512 / 16384 = 4` | 1024 | `4 * 1024 = 4096` | 20 | `20 x 4096` |

因此一个 FA row 的有效 H2D payload 是：

```text
group 0: 21 * 131072 = 2,752,512
group 3: 21 * 16384 + 21 * 256 = 349,440
group 8: 20 * 4096 = 81,920

FA_row_h2d_bytes = 3,183,872
```

重点解释：

| group | 怎么理解 |
| --- | --- |
| group 0 | 512 original tokens 对应完整 128 slots，所以一次 hit 搬完整 tensor block。 |
| group 3 | 4096 original tokens 对应 1024 slots；一次 512-token hit 只占其中 1/8，也就是 128 slots。 |
| group 8 | 16384 original tokens 对应 128 slots；一次 512-token hit 只占其中 1/32，也就是 4 slots。 |

group 3 的 16KB 和 256B 都来自同一个 `128 slots`，差别在每个 tensor view 的 `bytes_per_slot` 不同：

```text
128 slots * 128 bytes/slot = 16,384 bytes
128 slots * 2 bytes/slot   = 256 bytes
```

group 8 可以理解为 16K original tokens 映射到 128 physical slots，压缩比例是 `16384 / 128 = 128`。一次 512-token canonical block 只取 4 slots，不是取 128 slots，也不是放到 512 block 的 1/4；它占的是 group8 tensor block 的 `4 / 128 = 1/32`。

### group 内 offset

当要恢复第 `k` 个 512-token canonical block 时：

```text
token_start = k * 512
token_offset = token_start % token_block_size
physical_offset = token_offset * tensor_block_size / token_block_size
```

以 group 3 为例：

| canonical block | `token_start` | group3 physical offset |
| --- | --- | --- |
| 0 | 0 | 0 |
| 1 | 512 | 128 |
| 2 | 1024 | 256 |
| 7 | 3584 | 896 |

8 个 512-token canonical blocks 正好填满 group3 的一个 4096-token logical span，也正好覆盖 group3 的 1024 physical slots。

## WA group 的 row 贡献

WA 不是为每个 external-hit block 都 load 一行。它只恢复最终 boundary 需要的 window/compressor state，所以命中 `N > 0` 时只 load 1 行 WA row。

| group | 语义 | `token_block_size` | `tail_tokens` | slots_to_copy | 每 view bytes | view 数 | row 贡献 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | SWA/window tail | 128 | 128 | `128 * 128 / 128 = 128` | 131072 | 22 | `22 x 131072` |
| 2 | SWA/window tail | 128 | 128 | `128 * 128 / 128 = 128` | 131072 | 21 | `21 x 131072` |
| 4 | compressor tail | 32 | 4 | `32 * 4 / 32 = 4` | 16384 | 21 | `21 x 16384` |
| 5 | compressor tail | 32 | 4 | `32 * 4 / 32 = 4` | 16384 | 21 | `21 x 16384` |
| 6 | compressor tail | 128 | 4 | `128 * 4 / 128 = 4` | 4096 | 21 | `21 x 4096` |
| 7 | compressor tail | 128 | 4 | `128 * 4 / 128 = 4` | 4096 | 21 | `21 x 4096` |
| 9 | indexer/辅助 state | 64 | 0 | 0 | 0 | 20 | 不贡献 IO |
| 10 | indexer/辅助 state | 64 | 0 | 0 | 0 | 20 | 不贡献 IO |

WA row 的有效 H2D payload 是：

```text
group 1 + group 2: 43 * 131072 = 5,636,096
group 4 + group 5: 42 * 16384  =   688,128
group 6 + group 7: 42 * 4096   =   172,032
group 9 + group 10: tail_tokens=0, no IO

WA_row_h2d_bytes = 6,496,256
```

## Row Size 和 Fragment 分布

### FA/WA row size

| row 类型 | row 数 | fragment 数 | 有效 H2D bytes | CacheStore shard bytes | padding |
| --- | --- | --- | --- | --- | --- |
| FA row | 每个 external-hit block 1 行 | 83 | 3,183,872 | 3,186,688 | 2,816 |
| WA row | 每个 request 最多 1 行 | 127 | 6,496,256 | 6,496,256 | 0 |

FA 的 `ShardSize` 比有效 H2D payload 大，是因为 CacheStore row 管理按 4KB 对齐：

```text
round_up(3,183,872, 4096) = 3,186,688
```

padding 属于 CacheStore row/backend 管理口径，不会作为 tensor payload scatter 到 device。

### fragment 桶

| row 类型 | 128KB fragment | 16KB fragment | 4KB fragment | 256B fragment | fragment 总数 |
| --- | --- | --- | --- | --- | --- |
| FA row | 21 | 21 | 20 | 21 | 83 |
| WA row | 43 | 42 | 42 | 0 | 127 |

当前日志显示：

```text
H2DTransport to ce
```

所以 LoadQueue 在 CE 模式下会按 `tensor_size_list` 循环提交普通 async H2D copy。也就是说，表里的一个 fragment 就是一条 CE async H2D copy。

## 一次 CacheStore Load IO 如何生成

### 从命中到 load task

| 阶段 | 输入 | 输出 |
| --- | --- | --- |
| scheduler 命中判断 | request tokens、HBM hit、FA store prefix、WA boundary | `external_hit_blocks = N` |
| worker FA load | `request.load_keys`，长度为 `N` | 1 个 FA `load_data` task，里面有 `N` 行 FA row |
| worker WA load | `request.load_keys[-1:]` | 1 个 WA `load_data` task，里面有 1 行 WA row |
| PipelineStore | keys、shard indexes、device ptr rows | C++ `TaskDesc`，每个 row 变成一个 `Shard` |
| CacheStore | `CacheStore::Load(TaskDesc)` | 进入 `TransManager` 和 `LoadQueue` |
| LoadQueue | host-side row buffer、device addresses、`tensor_size_list` | 按 fragment 做 H2D scatter |

### 命中 `N` 个 blocks 时的 task/row 数

| 口径 | 数量 |
| --- | --- |
| CacheStore load task 数 | 2 个：1 个 FA load task + 1 个 WA load task |
| FA rows | `N` |
| WA rows | 1 |
| total rows | `N + 1` |
| FFTS object 数 | 如果启用 `ffts_pipeline`，是 `N + 1` 个 row objects |

如果 `N = 0`，FA/WA load task 都不会提交。

### 命中 `N` 个 blocks 时的 H2D bytes

| 口径 | 公式 |
| --- | --- |
| 有效 H2D payload | `N * 3,183,872 + 6,496,256` |
| CacheStore row/backend bytes | `N * 3,186,688 + 6,496,256` |

### 命中 `N` 个 blocks 时的 CE copy 数

| fragment 类型 | copy 数 |
| --- | --- |
| 128KB copy | `21N + 43` |
| 16KB copy | `21N + 42` |
| 4KB copy | `20N + 42` |
| 256B copy | `21N` |
| 总 async H2D copy | `83N + 127` |

如果把 128KB fragment 叫“大 IO”，把 16KB、4KB、256B 都归为“小 IO”：

| IO 类型 | copy 数 |
| --- | --- |
| 大 IO | `21N + 43` |
| 小 IO | `62N + 84` |

### 示例

| external_hit_blocks | load task | rows | 128KB 大 IO | 小 IO | 总 async H2D copy | 有效 H2D bytes |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | FA 1 个 + WA 1 个 | 2 | 64 | 146 | 210 | 9,680,128 |
| 10 | FA 1 个 + WA 1 个 | 11 | 253 | 704 | 957 | 38,334,976 |
| 100 | FA 1 个 + WA 1 个 | 101 | 2,143 | 6,284 | 8,427 | 324,883,456 |

## CE 和 FFTS Pipeline 的差异

| H2D transport | 提交粒度 | 当前日志是否启用 | 对 IO 形态的影响 |
| --- | --- | --- | --- |
| `ce` | 每个 fragment 一次 async H2D copy | 是，日志显示 `H2DTransport to ce` | copy 数按 `83N + 127` 计算。 |
| `ffts_pipeline` | 每个 row 作为一个 object，object 内带多个 fragments | 否，只有配置项打印不代表启用 | object 数是 `N + 1`；FA object `objectBytes=3,183,872, maxFragments=83`，WA object `objectBytes=6,496,256, maxFragments=127`。 |

日志里看到：

```text
H2DFftsPipelineDepth
H2DFftsMaxReadyLanes
```

只说明配置项被打印。是否真正启用，要看：

```text
H2DTransport == "ffts_pipeline"
```

当前日志是：

```text
H2DTransport to ce
```

所以实际仍然是普通 CE H2D。

## FA/WA 命中语义

| store | 命中要求 | load 行为 |
| --- | --- | --- |
| FA store | 必须从候选 external key 开始连续 prefix 命中。 | 命中多少个 canonical blocks，就 load 多少行 FA row。 |
| WA store | 在 FA 连续命中范围内，必须存在最终 boundary key。 | 只 load 最后一个 boundary row。 |

因此：

```text
external_hit_blocks =
    min(FA 连续 prefix 命中范围内，WA 有最终 boundary 的最大 block 数)
```

例子：

| FA prefix hit | WA boundary hit | final external hit |
| --- | --- | --- |
| 19 | 第 19 个 boundary 存在 | 19 |
| 19 | 只在第 12 个 boundary 存在 | 12 |
| 19 | WA 全 miss | 0 |

## 当前日志判断点

| 判断项 | 期望/解释 |
| --- | --- |
| 是否进入 FAWA | 日志出现 `FAWA KV group config` 和 `Init UCM FAWA connector`。 |
| group 分类 | `fa_groups=[0, 3, 8]`，`window_groups=[1, 2, 4, 5, 6, 7, 9, 10]`。 |
| hash block | `hash_block_size = 512`。 |
| FA store row | `tensor_count=83`，`tensor_bytes=3,183,872`，`ShardSize=3,186,688`。 |
| WA store row | `tensor_count=127`，`tensor_bytes=6,496,256`，`ShardSize=6,496,256`。 |
| scheduler store | `device_id=-1`、`TensorSizes=[]`、`ShardSize=0` 是 lookup-only store，不负责 H2D，不是错误。 |
| worker store | `device_id=0..7`，带真实 tensor sizes，负责本 rank H2D。 |
| `Cache|Empty` | EmptyStore 不提供持久化 payload；external hit 只能来自同一 `unique_id` 下可见的 CacheStore shared buffer。 |
| 实际 H2D transport | 当前是 `H2DTransport to ce`，不是 FFTS pipeline。 |

## 性能分析时看什么

| 问题 | 应该看 |
| --- | --- |
| 为什么命中 token 多但 TTFT 仍高？ | 用 `N * 3,183,872 + 6,496,256` 估算有效 H2D payload。 |
| 为什么小拷贝很多？ | 看 CE fragment 数：`83N + 127`。 |
| 为什么低命中时固定成本明显？ | WA row 固定为 `6,496,256` bytes，只要 `N > 0` 就要 load 一行。 |
| 为什么 FA hit 多但 external hit 少？ | 查 WA boundary 是否存在；WA miss 会把最终 external hit 压低。 |
| FFTS 参数打印了是否代表启用？ | 不代表；必须看到 `H2DTransport to ffts_pipeline`。 |
