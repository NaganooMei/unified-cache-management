# Ascend DSV4 IO 模式与命中 Load 模式报告

## 结论摘要

Ascend DSV4 在 UCM 中走 FAWA/HMA 路径时，一个 rank 内会创建两个逻辑 store：FA store 和 WA store。它们不是重复缓存，而是分别承载不同 KV 语义、不同 row size、不同命中约束和不同 load 粒度。

- FA store 保存 full-attention prefix row，按 canonical prefix block 连续命中和逐 block load。
- WA store 保存 window/compressor tail row，只要求命中最终边界，load 时只加载最后一个 boundary row。
- 最终外部命中长度不是只看 FA，而是 FA 连续前缀命中和 WA boundary 命中的交集。
- H2D load 的主公式是 `external_hit_blocks * FA_row_bytes + 1 * WA_row_bytes`。
- Storage/cache row IO 以 `ShardSize` 为准；H2D scatter 或 FFTS pipeline object IO 以 `tensor_size_list` 的 sum 为准。

## 背景

DSV4 不是普通 dense KV cache 布局。它在 vLLM-Ascend 侧会暴露多类 KV cache group，例如 full-attention 相关 group、sliding-window/window-tail 相关 group，以及 compressor/indexer 状态。UCM 的 FAWA connector 会根据 `kv_cache_config` 把这些 group 分成两类：

- FA group：没有 sliding window 的 full-attention group。
- WA group：带 sliding window 或 compressor tail 语义的 window group。

Ascend DSV4 的 hash block 粒度在 UCM FAWA 路径中为 512 token。请求 token 会按 512 token 生成 canonical hashes，每一个 hash 对应一个可复用 prefix boundary。

相关代码：

`@unified-cache-management/ucm/integration/vllm/hma_connector.py`

`@unified-cache-management/ucm/integration/vllm/ucm_connector.py`

## 为什么要拆成两个 store

拆分的根本原因是 FA 和 WA 的缓存语义不同。

FA store 的 value 是某个 canonical prefix block 对应的 full-attention 数据。prefix 复用需要连续块，所以 FA store 用 `lookup_on_prefix` 找最长连续前缀。load 时，外部命中了多少个 canonical block，就需要加载多少行 FA。

WA store 的 value 是某个 prefix boundary 对应的 window tail 或 compressor state。恢复推理时不需要把历史上每个 boundary 的 WA 都加载回来，只需要最终命中边界对应的 tail 状态。因此 WA store 不要求每个 block 都连续命中，load 时也只取最后一个 boundary key。

如果强行合并到一个 store，会遇到几个问题：

| 维度 | FA store | WA store | 合并问题 |
| --- | --- | --- | --- |
| 命中语义 | 连续前缀 | 最终 boundary | 一个 lookup 策略表达不了 |
| Load 粒度 | 每个外部命中 block 都 load | 只 load 最后一个 boundary | 会多读 WA 或漏读 FA |
| Row size | 由 FA group layout 决定 | 由 WA group layout 决定 | CacheStore 单实例要求固定 tensor list |
| Tensor list | FA fragment 列表 | WA fragment 列表 | 大小和列数不一致 |
| 存储 namespace | `_fawa_fa` | `_fawa_wa` | 需要隔离 key/value 语义 |

因此 UCM 在 FAWA 路径里会创建两个 CacheStore 实例，unique id 分别带 `_fawa_fa` 和 `_fawa_wa` 后缀；如果配置了 storage backend，也会把 FA/WA 放到不同的 backend 子目录。

## Store 创建模式

Worker 注册 KV cache 后，会为每个 group 构造 `KVCacheGroupLayout`，再分别创建 FA store 和 WA store。

整体过程如下：

```text
register_kv_caches
  -> 按 kv_cache_config 拆分 group layout
  -> _create_fa_store(group_layouts)
       -> _store_tensor_size_list(fa_group_ids)
       -> unique_id = engine_id + "_fawa_fa"
       -> shard_size = round_up(sum(tensor_size_list), 4096)
  -> _create_wa_store(group_layouts)
       -> _store_tensor_size_list(window_group_ids)
       -> unique_id = engine_id + "_fawa_wa"
       -> shard_size = round_up(sum(tensor_size_list), 4096)
```

`tensor_size_list` 的含义是：一个 store row 里包含多少个连续 tensor fragment，以及每个 fragment 的字节数。它不是 token 数，也不是 layer 数。

CacheStore 的 H2D 阶段会按这个 list 逐片段拷贝：

```text
host row buffer:
  [fragment_0][fragment_1][fragment_2]...[fragment_n]

device dst ptr list:
  [dst_0, dst_1, dst_2, ..., dst_n]

copy:
  fragment_i -> dst_i, size=tensor_size_list[i]
```

如果 H2D transport 是 `ce`，就是按 fragment 逐个下发 CE copy；如果 H2D transport 是 `ffts_pipeline`，则把整个 row 看成一个 object，再由 FFTS pipeline 将 object 拆到多个 device fragment。

相关代码：

`@unified-cache-management/ucm/store/cache/cc/cache_store.cc`

`@unified-cache-management/ucm/store/cache/cc/load_queue.cc`

## 命中判断模式

命中判断发生在 scheduler 侧。它先区分 HBM 已经命中的 block，再对剩余 block 查外部 cache。

流程如下：

```text
输入:
  request.all_token_ids
  num_computed_tokens

1. 计算 HBM 已命中 block 数
   hbm_hit_block_num = num_computed_tokens / hash_block_size

2. 对请求 token 生成 canonical hashes
   canonical_hashes = generate_hash(hash_block_size, request.all_token_ids)

3. 只查 HBM 之后的外部候选 key
   external_keys = canonical_hashes[hbm_hit_block_num:]

4. 查 FA store 的最长连续前缀
   fa_hit_blocks = fa_store.lookup_on_prefix(external_keys) + 1

5. 在 FA 命中范围内倒序查 WA boundary
   for hit_blocks in range(fa_hit_blocks, -1, -1):
       key = external_keys[hit_blocks - 1]
       if wa_store.lookup([key])[0]:
           external_hit_blocks = hit_blocks
           break

6. 最终命中
   total_hit_block_num = hbm_hit_block_num + external_hit_blocks
   external_hit_tokens = external_hit_blocks * hash_block_size
```

这里最关键的是第 5 步。FA 命中很多并不代表最终可用命中很多，因为恢复 WA 需要最终 boundary 的 window tail 状态。如果 FA 有 100 个连续 block，但 WA 只在第 80 个 boundary 存在，那么最终 external hit 只能算 80 个 block。

这也是为什么两个 store 都必须命中：FA 决定可复用的 full-attention prefix，WA 决定这个 prefix boundary 能不能恢复 window/compressor state。

## Load 模式

Worker 侧 `start_load_kv` 收到 scheduler 下发的 dispatch metadata 后，会分别提交 FA load 和 WA load。

对于同一个请求：

```text
load range:
  [hbm_hit_block_num, total_hit_block_num)

request.load_keys:
  这一段外部命中的 canonical hashes

FA load:
  keys = request.load_keys
  rows = external_hit_blocks
  ptrs = _extract_fa_ptr(...)
  store = fa_store

WA load:
  keys = request.load_keys[-1:]
  rows = 1
  ptrs = _extract_wa_ptr(...)
  store = wa_store
```

也就是说，FA load 是逐命中 block 加载；WA load 只加载最后一个 boundary。这个模式和保存模式也是对应的：FA 可以按 canonical block 保存多行，WA 只需要保存每个请求完成边界的最终 tail row。

## IO 计算模式

### Row IO

一个 CacheStore row 的有效 H2D 字节数为：

```text
row_h2d_bytes = sum(tensor_size_list)
```

CacheStore 的 storage/cache row 管理通常以 `ShardSize` 为准：

```text
row_storage_bytes = shard_size
```

两者大多数情况下相同；如果 `tensor_size_list` 的 sum 不是 4KB 对齐，或者日志摘录不完整，则 `ShardSize` 会以 4KB 对齐后的值为准。

### 外部命中 Load IO

假设：

```text
N = external_hit_blocks
FA = FA_row_h2d_bytes
WA = WA_row_h2d_bytes
```

则 H2D IO 为：

```text
h2d_bytes = N * FA + WA
```

如果考虑 storage/cache row 读取，则近似为：

```text
storage_or_cache_row_bytes = N * FA_shard_size + WA_shard_size
```

如果命中在 CacheStore DRAM 中，主要就是 host cache -> device 的 H2D；如果只在 PosixStore 等后端命中，则会先 backend -> CacheStore host buffer，再 CacheStore host buffer -> device。

### 基于当前完整日志的解析

这份启动日志可以分成三类信息看：FAWA group 配置、worker/rank 侧 store 创建、scheduler 侧 lookup store 创建。

#### 1. 已经进入 Ascend FAWA 路径

日志中反复出现：

```text
FAWA KV group config: fa_groups=[0, 3, 8], window_groups=[1, 2, 4, 5, 6, 7, 9, 10], is_ascend_layout=True
Init UCM FAWA connector.
```

这说明当前没有走普通 direct connector，也没有走 layerwise connector。虽然配置里有 `use_layerwise: True`，但 UCMConnector 会先检查 `UCMFAWAConnector.can_handle_kv_cache_config(kv_cache_config)`；只要 DSV4 Ascend KV layout 匹配，就优先进入 FAWA 路径，后面的 layerwise 分支不会被选中。

当前配置里还有：

```text
store_pipeline: Cache|Empty
share_buffer_enable: True
persist_token_threshold: 0
```

这表示后端是 CacheStore 加 EmptyStore。没有 PosixStore 这类持久化 backend 时，所谓 external hit 只能来自同一个 `unique_id` 下的 CacheStore 共享 host buffer；EmptyStore 本身不会提供持久化命中。

#### 2. 一个 engine 下有 8 个 worker rank，共用同一组 FA/WA namespace

worker 侧可以看到 device id 从 0 到 7 都创建了同样的 FA store 和 WA store：

```text
unique_id = ..._fawa_fa
device_id = 0..7
local_rank_size = 8

unique_id = ..._fawa_wa
device_id = 0..7
local_rank_size = 8
```

这符合 DSV4 MLA 的设计。`local_rank_size=8` 表示这些 rank 属于同一个本地 TP/MLA rank group；`share_buffer_enable=True` 且 `unique_id` 相同，说明这些进程打开的是同一组逻辑共享 CacheStore namespace。每个 worker 仍然有自己的 `device_id`，用于把共享 host buffer 中的数据 H2D 到本 rank 的 device。

同一份日志里还出现了 engine/scheduler 进程创建 store：

```text
device_id = -1
TensorSizes = []
ShardSize = 0
BlockSize = 0
```

这是正常的 scheduler/lookup 侧实例。CacheStore 在 `device_id=-1` 时不会启用传输管理，也不会要求 tensor size、shard size、block size 有效；它主要用于 lookup，不负责 H2D load。因此不要把这两条空 TensorSizes 的日志理解成 worker store 配置错误。

#### 3. group_metas 的含义

当前 DSV4 Ascend layout 被分成 11 个 group：

```text
FA groups: 0, 3, 8
WA groups: 1, 2, 4, 5, 6, 7, 9, 10
```

`token_block_size` 是这个 group 的逻辑 block span，`tail_tokens` 是该 store row 需要保存或恢复的 token 范围。FA group 的 `tail_tokens` 固定是 UCM canonical hash block，也就是 512 token；WA group 的 `tail_tokens` 来自 sliding window 或 compressor state 的 tail。

根据日志中的 `KV cache group layout` 和 store summary，可以把 row 贡献拆成下面这样：

| group | 类型 | group meta | layout 日志 | row 贡献 |
| --- | --- | --- | --- | --- |
| 0 | FA | token_block_size=512, tail_tokens=512 | views=21, ptrs=21, tensor_block_sizes=[128] | 21 x 131072 |
| 3 | FA | token_block_size=4096, tail_tokens=512 | views=21, ptrs=42, tensor_block_sizes=[1024] | 21 x 16384 + 21 x 256 |
| 8 | FA | token_block_size=16384, tail_tokens=512 | views=20, ptrs=20, tensor_block_sizes=[128] | 20 x 4096 |
| 1 | WA | token_block_size=128, tail_tokens=128 | views=22, ptrs=22, tensor_block_sizes=[128] | 22 x 131072 |
| 2 | WA | token_block_size=128, tail_tokens=128 | views=21, ptrs=21, tensor_block_sizes=[128] | 21 x 131072 |
| 4 | WA | token_block_size=32, tail_tokens=4 | views=21, ptrs=21, tensor_block_sizes=[32] | 21 x 16384 |
| 5 | WA | token_block_size=32, tail_tokens=4 | views=21, ptrs=21, tensor_block_sizes=[32] | 21 x 16384 |
| 6 | WA | token_block_size=128, tail_tokens=4 | views=21, ptrs=21, tensor_block_sizes=[128] | 21 x 4096 |
| 7 | WA | token_block_size=128, tail_tokens=4 | views=21, ptrs=21, tensor_block_sizes=[128] | 21 x 4096 |
| 9 | WA | token_block_size=64, tail_tokens=0 | views=20, ptrs=20, tensor_block_sizes=[64] | 不贡献 IO |
| 10 | WA | token_block_size=64, tail_tokens=0 | views=20, ptrs=20, tensor_block_sizes=[64] | 不贡献 IO |

这里 group 9 和 group 10 虽然出现在 `window_groups` 里，但 `tail_tokens=0`，构造 `tensor_size_list` 和提取 WA ptr 时都会跳过，所以不贡献 WA row IO。

#### 4. 当前 FA store row size

当前 worker 侧 FA store summary：

```text
tensor_count = 83
tensor_bytes = 3,183,872 bytes
ShardSize = 3,186,688 bytes
```

FA 的 tensor list 形态为：

```text
131072 x 21
16384  x 21
256    x 21
4096   x 20
```

也就是：

```text
group 0: 21 x 131072 = 2,752,512
group 3: 21 x (16384 + 256) = 349,440
group 8: 20 x 4096 = 81,920
FA tensor_bytes = 3,183,872
FA ShardSize = round_up(3,183,872, 4096) = 3,186,688
```

所以 FA 的有效 H2D bytes 是 `3,183,872`，storage/cache row 管理 bytes 是 `3,186,688`。

#### 5. 当前 WA store row size

当前 worker 侧 WA store summary：

```text
tensor_count = 127
tensor_bytes = 6,496,256 bytes
ShardSize = 6,496,256 bytes
约 6.20 MiB
```

WA 的 tensor list 形态为：

```text
131072 x 43
16384  x 42
4096   x 42
```

```text
group 1 + group 2: 43 x 131072 = 5,636,096
group 4 + group 5: 42 x 16384 = 688,128
group 6 + group 7: 42 x 4096 = 172,032
group 9 + group 10: tail_tokens=0, no IO
WA tensor_bytes = 6,496,256
WA ShardSize = 6,496,256
```

#### 6. 当前日志下的 load IO 公式

按当前日志，单请求外部命中 `N` 个 512-token canonical block 时：

```text
H2D bytes = N * 3,183,872 + 6,496,256
Cache row bytes = N * 3,186,688 + 6,496,256
```

示例：

| 外部命中 block 数 | FA 有效 H2D | WA 有效 H2D | 总有效 H2D |
| --- | --- | --- | --- |
| 1 | 约 3.04 MiB | 约 6.20 MiB | 约 9.23 MiB |
| 10 | 约 30.36 MiB | 约 6.20 MiB | 约 36.56 MiB |
| 100 | 约 303.64 MiB | 约 6.20 MiB | 约 309.83 MiB |

这里的 block 是 512 token canonical block，不是 vLLM 物理 KV block。

#### 7. 当前日志仍然是 CE H2D

worker FA store 和 WA store 都显示：

```text
H2DTransport to ce
H2DFftsPipelineDepth to 2
H2DFftsMaxReadyLanes to 8
```

这说明当前实际 H2D transport 仍然是 CE。`H2DFftsPipelineDepth` 和 `H2DFftsMaxReadyLanes` 是配置项打印，不代表已经走 FFTS pipeline。要走 FFTS pipeline，需要显式设置 `cache_h2d_transport: ffts_pipeline`，同时运行的 UCM 必须是带 FFTS pipeline 编译开关的版本。

#### 8. shm broadcast warning 的位置

日志中有一条：

```text
No available shared memory broadcast block found in 60 seconds...
init engine (profile, create kv cache, warmup model) took 124.16 seconds
```

它出现在 worker 创建 KV cache / warmup 阶段附近，日志本身也提示常见原因是编译、权重量化、KV cache 初始化等耗时操作。只要后续 engine init 完成并继续创建 scheduler connector，这条更像启动期等待告警，不是 FA/WA store tensor list 或 H2D transport 的配置错误。

## H2D transport 对 IO 的影响

`tensor_size_list` 决定一个 row 会被拆成多少个 fragment。Ascend DSV4 的 FA/WA row 里通常会出现 128KB、16KB、4KB，甚至 256B 这类混合小 fragment。

当 `H2DTransport=ce` 时，LoadQueue 会按 fragment 循环调用普通 H2D async copy。fragment 数越多、fragment 越小，下发开销越明显。

当 `H2DTransport=ffts_pipeline` 时，LoadQueue 会把一个 row 作为 object 提交给 FFTS pipeline：

```text
objectBytes = sum(tensor_size_list)
maxFragments = len(tensor_size_list)
```

这样可以把 host 侧大对象 staging 和 device 侧 fragment 拆分组织成 pipeline，目标是降低小 IO scatter 场景的下发开销。

需要注意：日志里打印 `H2DFftsPipelineDepth` 和 `H2DFftsMaxReadyLanes` 不代表一定启用了 FFTS pipeline。是否启用要看：

```text
H2DTransport == "ffts_pipeline"
```

同时编译时需要打开 Ascend FFTS pipeline 相关开关。若日志显示 `H2DTransport to ce`，实际仍是普通 CE H2D。

## Save/Dump 模式补充

保存路径和 load 路径保持语义对称。

FA dump：

- 保存可复用 canonical block 对应的 FA row。
- TP 多卡场景下，会按 canonical block index 切分给不同 TP rank 保存，避免每个 rank 都重复 dump 全部 FA row。

WA dump：

- 保存请求最终 boundary 的 WA tail row。
- TP 多卡场景下，会按 request boundary round-robin 分配给 rank 保存。

这也是为什么 load 时 WA 只取最后一个 key：WA row 表示某个 boundary 的 tail 状态，不需要把中间每个 boundary 都恢复到 device。

## 对性能分析的含义

对 Ascend DSV4 来说，命中率不能只看“命中了多少 token”。需要同时看：

- HBM 已有多少 block。
- FA store 外部连续命中多少 block。
- WA store 在 FA 命中范围内能不能找到最终 boundary。
- 每个 external hit block 对应的 FA row size。
- 每个请求固定多出来的 WA row size。
- H2D transport 是 CE 还是 FFTS pipeline。

在长 prompt、高命中场景下，FA IO 随 external hit blocks 线性增长；WA IO 是每个请求一次固定成本。命中越多，`N * FA` 成为主要部分；命中很少时，单次 WA row 的固定成本会更显眼。

## 排查建议

看日志时建议按下面顺序确认：

```text
1. 是否进入 FAWA:
   Init UCM FAWA connector.
   FAWA KV group config: ... is_ascend_layout=True ...

2. FA/WA group 是否合理:
   fa_groups=...
   window_groups=...
   group_metas=...

3. 两个 store 是否都创建:
   unique_id 后缀包含 _fawa_fa
   unique_id 后缀包含 _fawa_wa

4. row size 是否符合预期:
   FA store TensorSizes / ShardSize
   WA store TensorSizes / ShardSize

5. 实际 H2D transport:
   H2DTransport to ce
   或 H2DTransport to ffts_pipeline

6. 命中日志:
   hit hbm: ...
   hit external: ...
```

如果 FA hit 很多但 external hit 偏低，要优先检查 WA store 是否缺少对应 boundary key。
如果 external hit 正常但 TTFT 仍高，要优先估算 `N * FA + WA` 的 H2D IO，并确认当前是否仍在 CE 小 fragment 路径上。

## 参考代码

`@unified-cache-management/ucm/integration/vllm/hma_connector.py`

`@unified-cache-management/ucm/integration/vllm/ucm_connector.py`

`@unified-cache-management/ucm/store/cache/cc/cache_store.cc`

`@unified-cache-management/ucm/store/cache/cc/load_queue.cc`

`@unified-cache-management/ucm/store/pipeline/connector.py`
