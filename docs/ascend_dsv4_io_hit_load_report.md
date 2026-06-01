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

### 基于当前日志的示例

当前观察到 WA store：

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

当前观察到另一组 FA store 日志大约为 3 MiB 级别，`ShardSize` 显示为：

```text
ShardSize = 3,186,688 bytes
约 3.04 MiB
```

因此可以用下面的近似估算外部命中 load IO：

| 外部命中 block 数 | FA H2D | WA H2D | 总 H2D |
| --- | --- | --- | --- |
| 1 | 约 3.04 MiB | 约 6.20 MiB | 约 9.24 MiB |
| 10 | 约 30.4 MiB | 约 6.20 MiB | 约 36.6 MiB |
| 100 | 约 304 MiB | 约 6.20 MiB | 约 310 MiB |

这里的 block 是 512 token canonical block，不是 vLLM 物理 KV block。

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
