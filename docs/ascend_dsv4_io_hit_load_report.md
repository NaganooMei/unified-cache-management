# Ascend DSV4 KV Cache 分组、命中与 Load IO 分析

## 结论摘要

这份文档解释 DeepSeek-V4/DSV4 在 vLLM-Ascend + UCM 中的 KV cache 组织方式，以及 UCM FAWA connector 为什么把它拆成 FA store 和 WA store 两套缓存。

核心结论：

- DeepSeek-V4 的长上下文 KV cache 不是普通 dense KV cache，而是由 classical KV cache 和 state cache 共同组成。
- 论文里的 classical KV cache 主要保存 CSA/HCA 压缩后的历史表示；state cache 保存 SWA 最近窗口以及 CSA/HCA 尚未压缩完成的 tail state。
- UCM FAWA connector 把 vLLM-Ascend 暴露出来的 KV cache groups 分成两类：FA groups 和 WA groups。
- FA store 对应可按 prefix 复用的 full/classical attention rows，要求从外部候选 key 开始连续命中。
- WA store 对应恢复最终 boundary 所需的 window/compressor state，只要求最终 boundary 命中，load 时只加载最后一个 boundary row。
- Ascend DSV4 的 UCM canonical hash block 是 512 original tokens。命中统计里的 block 指这个 512-token canonical block。
- `token_block_size` 不是 CacheStore 的字节 block，也不是一次命中的 token 数。它表示某个 KV cache group 的一个 logical block 在 original-token 坐标上覆盖的跨度。
- 对压缩 KV group，`token_block_size = kv_cache_spec.block_size * compress_ratio`。它必须和 tensor layout 里的 `tensor_block_size` 一起看，才能知道实际要从 KV tensor 里取多少物理槽位。
- CacheStore 视角下，单请求外部命中 `N` 个 blocks 时，会提交 1 个 FA load task 和 1 个 WA load task；当前 CE H2D 路径下，实际 async copy 数是 `83N + 127`。
- 当前日志下单请求外部命中 `N` 个 512-token canonical blocks 时，H2D load 公式是：

```text
H2D bytes = N * FA_row_h2d_bytes + WA_row_h2d_bytes
          = N * 3,183,872 + 6,496,256
```

如果看 CacheStore row 管理或后端读写，FA row 要按 4KB 对齐后的 `ShardSize=3,186,688` 估算。

## 资料来源与代码入口

论文和公开资料：

- DeepSeek-V4 technical report: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/DeepSeek_V4.pdf?download=true
- Hugging Face Transformers DeepSeek-V4 model docs: https://huggingface.co/docs/transformers/model_doc/deepseek_v4
- NVIDIA Megatron Bridge DeepSeek-V4 notes: https://docs.nvidia.com/nemo/megatron-bridge/nightly/models/deepseek/deepseek-v4.html

本地代码入口：

`@ucm/integration/vllm/hma_connector.py`

`@ucm/integration/vllm/ucm_connector.py`

`@ucm/store/cache/cc/cache_store.cc`

`@ucm/store/cache/cc/trans_manager.h`

`@ucm/store/cache/cc/load_queue.cc`

`@ucm/store/pipeline/connector.py`

`@ucm/store/pipeline/cpy/pipeline_store.py.cc`

## DeepSeek-V4 论文里的 KV Cache 结构

DeepSeek-V4 的长上下文能力主要依赖 NSA-based hybrid attention。对 KV cache 来说，最重要的是三类 attention/cache 组件：

1. CSA: Compressed Sparse Attention
2. HCA: Highly Compressed Attention
3. SWA: Sliding Window Attention

论文里可以把推理态 KV cache 分成两大类：

### Classical KV Cache

Classical KV cache 保存已经压缩成稳定历史表示的 KV entries。

这里的 entries 不是普通 dense attention 中每个 original token 一份 K/V，而是 CSA/HCA 压缩后的 entries。

典型语义：

- CSA 以较低压缩率保存历史上下文，再通过 sparse/top-k 选择相关 compressed entries。
- HCA 以更高压缩率保存更长历史，用较小 KV cache 成本覆盖百万 token 级上下文。
- 这些 compressed entries 是 prefix 可复用的：只要同一个 prefix boundary 已经保存，下次请求可以按 prefix block load 回来。

这部分和 UCM FA store 的语义最接近。

### State Cache

State cache 保存恢复继续推理所必需的边界状态。

它主要包括：

- SWA 的最近窗口 KV。
- CSA/HCA 还没达到压缩粒度时暂存的 uncompressed tail state。
- 压缩或索引相关的局部状态。

这部分不是“每个历史 boundary 都要全部 load 回来”。恢复一个 prefix 时，关键是最终命中 boundary 的 state 是否存在。只要最终 boundary 的 window/compressor tail state 能恢复，模型就可以从这个边界继续推理。

这部分和 UCM WA store 的语义最接近。

### 论文语义到 UCM 语义的映射

| 论文概念 | 作用 | UCM FAWA 中的近似对应 | 命中/Load 特点 |
| --- | --- | --- | --- |
| CSA compressed KV | 低压缩历史表示，支持稀疏选择 | FA store 的一部分 | 按 prefix block 连续命中，命中多少 load 多少行 |
| HCA compressed KV | 高压缩长历史表示 | FA store 的一部分 | 按 prefix block 连续命中，单个 512-token block 对应更少 tensor 槽位 |
| SWA recent window | 最近窗口的未压缩局部上下文 | WA store 的一部分 | 只需要最终 boundary 的 window tail |
| CSA/HCA tail state | 尚未完成压缩块的未压缩尾部 | WA store 的一部分 | 只需要最终 boundary 的 compressor tail |
| Indexer/辅助状态 | sparse/compressed attention 的辅助状态 | 部分 WA group，但可能不贡献 IO | `tail_tokens=0` 时跳过 load/dump |

注意：UCM 不直接用论文里的 CSA/HCA 名称分组。UCM 看到的是 vLLM-Ascend 提供的 `kv_cache_config.kv_cache_groups`、spec 类型、block size、compress ratio、sliding window 和真实注册的 tensor layout。FA/WA 是 UCM 为了 prefix cache 复用而建立的存储语义。

## 当前 Ascend DSV4 的 KV Cache Groups

当前启动日志里的 FAWA group config 是：

```text
FAWA KV group config:
  fa_groups=[0, 3, 8]
  window_groups=[1, 2, 4, 5, 6, 7, 9, 10]
  is_ascend_layout=True
```

文档下文使用的 group 信息来自同一份日志中的 `KV cache group layout` 和 store summary。

### Group 总览

| group | UCM 分类 | 语义解释 | token_block_size | tail_tokens | tensor layout 摘要 | row 贡献 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | FA | 接近未压缩或低层 full/classical KV 表示 | 512 | 512 | views=21, ptrs=21, tensor_block_sizes=[128] | 21 x 131072 |
| 3 | FA | 压缩历史表示，logical span 放大 | 4096 | 512 | views=21, ptrs=42, tensor_block_sizes=[1024] | 21 x 16384 + 21 x 256 |
| 8 | FA | 高压缩历史表示，logical span 更大 | 16384 | 512 | views=20, ptrs=20, tensor_block_sizes=[128] | 20 x 4096 |
| 1 | WA | SWA/window tail 类 state | 128 | 128 | views=22, ptrs=22, tensor_block_sizes=[128] | 22 x 131072 |
| 2 | WA | SWA/window tail 类 state | 128 | 128 | views=21, ptrs=21, tensor_block_sizes=[128] | 21 x 131072 |
| 4 | WA | compressor tail state | 32 | 4 | views=21, ptrs=21, tensor_block_sizes=[32] | 21 x 16384 |
| 5 | WA | compressor tail state | 32 | 4 | views=21, ptrs=21, tensor_block_sizes=[32] | 21 x 16384 |
| 6 | WA | compressor tail state | 128 | 4 | views=21, ptrs=21, tensor_block_sizes=[128] | 21 x 4096 |
| 7 | WA | compressor tail state | 128 | 4 | views=21, ptrs=21, tensor_block_sizes=[128] | 21 x 4096 |
| 9 | WA | indexer/辅助 state，当前不搬运 | 64 | 0 | views=20, ptrs=20, tensor_block_sizes=[64] | 不贡献 IO |
| 10 | WA | indexer/辅助 state，当前不搬运 | 64 | 0 | views=20, ptrs=20, tensor_block_sizes=[64] | 不贡献 IO |

### 为什么 group 9/10 是 WA 但不贡献 IO

group 9 和 group 10 出现在 `window_groups` 中，但 `tail_tokens=0`。

UCM 在构造 `tensor_size_list` 时会跳过 `tail_tokens=0` 的 group：

```text
if not meta.tail_tokens:
    continue
```

因此它们不会进入 WA store row 的 tensor fragment 列表，也不会在 `_extract_wa_ptr` 里生成 load/dump 指针。

这类 group 更像 DSV4 layout 中暴露的辅助/indexer 状态。它在 `kv_cache_config` 中存在，但对当前 UCM 外部命中恢复路径不产生 H2D IO。

相关代码：

`@ucm/integration/vllm/hma_connector.py`

`@ucm/integration/vllm/hma_connector.py`

## token_block_size 到底是什么

这是最容易误解的概念。

不要把 `token_block_size` 理解成“这次命中要搬多少 token 的 KV”，也不要把它等同于 CacheStore 的 `block_size`。

在 UCM FAWA connector 里，它的准确作用是：

```text
token_block_size = 这个 KV cache group 的一个 logical block 在 original-token 坐标上覆盖的跨度
```

对 Ascend compressed groups，代码会做：

```text
token_block_size = kv_cache_spec.block_size * compress_ratio
```

也就是说，压缩 group 的 `token_block_size` 被放大，是因为它的一个 KV tensor block 对应多个 original tokens。

相关代码：

`@ucm/integration/vllm/hma_connector.py`

### token_block_size 必须和 tensor_block_size 一起看

UCM 真实搬运的是 tensor 里的物理槽位，不是 original-token 本身。

一个 group layout 里还有 `tensor_block_size`：

```text
tensor_block_size = 当前注册 tensor 在 token 维度上的物理槽位数
```

UCM 用下面两个换算来处理 logical token range 和 physical tensor range：

```text
physical_token_offsets =
    logical_token_offsets * tensor_block_size / token_block_size

tensor_tokens_to_copy =
    tensor_block_size * logical_tokens_to_copy / token_block_size
```

相关代码：

`@ucm/integration/vllm/hma_connector.py`

`@ucm/integration/vllm/hma_connector.py`

### 用 FA group 0/3/8 举例

UCM canonical hash block 是 512 original tokens。FA store 每行保存一个 canonical block 对应的 FA row。

但不同 FA group 的 tensor layout 不同：

| group | token_block_size | tensor_block_size | 一个 512-token canonical block 对应的 tensor 槽位 |
| --- | --- | --- | --- |
| 0 | 512 | 128 | `128 * 512 / 512 = 128` |
| 3 | 4096 | 1024 | `1024 * 512 / 4096 = 128` |
| 8 | 16384 | 128 | `128 * 512 / 16384 = 4` |

因此：

- group 0 的 512-token canonical block 对应完整的 128 个 tensor slots。
- group 3 的 512-token canonical block 对应一个 larger logical block 里的 128 个 tensor slots。
- group 8 的 512-token canonical block 只对应一个 high-compression logical block 里的 4 个 tensor slots。

这解释了为什么 group 8 的 row 贡献很小：

```text
group 8 row contribution = 20 x 4096 bytes
```

它不是没有覆盖这 512 original tokens，而是这些 tokens 在高压缩 group 中被映射到很少的 physical tensor slots。

### token_block_size 与 offset

当 UCM 要恢复第 `k` 个 512-token canonical block 时，它先计算 original-token 起点：

```text
token_start = k * 512
```

如果某个 group 的 `token_block_size` 不是 512，就要计算它在该 group logical block 内的偏移：

```text
token_offset = token_start % token_block_size
```

再换算成 physical tensor offset：

```text
physical_offset = token_offset * tensor_block_size / token_block_size
```

例子：group 3 的 `token_block_size=4096`，`tensor_block_size=1024`。

```text
canonical block 0: token_start=0     -> physical_offset=0
canonical block 1: token_start=512   -> physical_offset=128
canonical block 2: token_start=1024  -> physical_offset=256
canonical block 7: token_start=3584  -> physical_offset=896
```

所以 8 个 512-token canonical blocks 共同落在 group 3 的一个 4096-token logical span 中，但它们对应的 physical tensor slice 不同。

相关代码：

`@ucm/integration/vllm/hma_connector.py`

## UCM FAWA Connector 如何识别 DSV4

UCMFAWAConnector 会先判断当前 KV cache config 是否适合 FAWA 路径。

GPU 侧目前用 DeepSeek-V4 style MLA+SWA layout 作为判断线索：

```text
DS_V4_REQUIRED_SPECS = {"SlidingWindowMLASpec"}
```

Ascend 侧要求 vLLM-Ascend 暴露的 spec 中包含：

```text
Compress4AttentionSpec
C4IndexerSpec
Compress128AttentionSpec
```

并且第一个 group 类型名以 `Ascend` 开头。

进入 Ascend DSV4 路径后：

```text
is_ascend_layout = True
hash_block_size = 512
```

相关代码：

`@ucm/integration/vllm/hma_connector.py`

`@ucm/integration/vllm/hma_connector.py`

`@ucm/integration/vllm/hma_connector.py`

## UCM 如何把 Groups 分成 FA 和 WA

UCM 遍历 `kv_cache_config.kv_cache_groups`，读取每个 group 的 representative spec：

```text
window_size = getattr(spec, "sliding_window", None)
compress_ratio = getattr(spec, "compress_ratio", 1)
token_block_size = kv_cache_spec.block_size

if is_ascend_layout:
    token_block_size = kv_cache_spec.block_size * compress_ratio
```

分类规则：

- `window_size is None`：归为 FA group。
- `window_size is not None`：归为 WA group。

FA group 的 `tail_tokens` 固定为 UCM canonical hash block，也就是 512 original tokens：

```text
tail_tokens = hash_block_size
```

WA group 的 `tail_tokens` 分两种：

```text
如果是 SWA cache:
    tail_tokens = sliding_window

否则认为是 compressor state:
    tail_tokens = sliding_window - layer_compress_ratios[layer_index]
```

最后计算：

```text
tail_blocks = max(tail_tokens // token_block_size, 1)
```

相关代码：

`@ucm/integration/vllm/hma_connector.py`

`@ucm/integration/vllm/hma_connector.py`

`@ucm/integration/vllm/hma_connector.py`

## Store 创建模式

Worker 注册 KV cache 后，会为每个 group 构造 `KVCacheGroupLayout`，再分别创建 FA store 和 WA store。

整体流程：

```text
register_kv_caches
  -> 为每个 kv_cache_group 生成 KVCacheGroupLayout
  -> _create_fa_store(group_layouts)
       -> _store_tensor_size_list(fa_group_ids)
       -> unique_id = engine_id + "_fawa_fa"
       -> shard_size = round_up(sum(tensor_size_list), 4096)
       -> block_size = shard_size
  -> _create_wa_store(group_layouts)
       -> _store_tensor_size_list(window_group_ids)
       -> unique_id = engine_id + "_fawa_wa"
       -> shard_size = round_up(sum(tensor_size_list), 4096)
       -> block_size = shard_size
```

FA/WA store 的配置隔离体现在：

```text
unique_id = {engine_id}_fawa_fa
unique_id = {engine_id}_fawa_wa
```

如果配置了 `storage_backends`，UCM 还会把 FA/WA 的后端目录拆成：

```text
fawa_fa
fawa_wa
```

这很重要，因为 FA 和 WA 的 key 语义、row size、tensor list 都不同，不能混在同一个 CacheStore namespace。

相关代码：

`@ucm/integration/vllm/hma_connector.py`

`@ucm/integration/vllm/hma_connector.py`

`@ucm/integration/vllm/hma_connector.py`

`@ucm/integration/vllm/hma_connector.py`

## tensor_size_list 如何生成

`tensor_size_list` 是一个 store row 中每个 tensor fragment 的字节大小列表。

它不是 token 数、不是 layer 数，也不是 CacheStore row 总大小。

生成逻辑：

```text
for group_id in group_ids:
    meta = group_metas[group_id]
    if tail_tokens == 0:
        continue

    segment_tokens = tail_tokens / tail_blocks

    for each tail block:
        segment_sizes = layout.segment_tensor_size_list(
            segment_tokens,
            meta.token_block_size,
        )
        tensor_size_list.extend(segment_sizes)
```

`segment_tensor_size_list` 内部用：

```text
tensor_tokens = tensor_block_sizes * logical_tokens / group_token_block_size
fragment_bytes = tensor_sizes_per_token * tensor_tokens
```

所以同样是 512 original tokens，不同 group 因为 `token_block_size` 和 `tensor_block_size` 不同，会贡献不同 fragment bytes。

相关代码：

`@ucm/integration/vllm/hma_connector.py`

`@ucm/integration/vllm/hma_connector.py`

## 命中判断模式

命中判断发生在 scheduler 侧。

DSV4 FAWA 路径中的命中单位是 512-token canonical hash block。

流程：

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

4. 查 FA store 的最长连续 prefix
   fa_hit_blocks = fa_store.lookup_on_prefix(external_keys) + 1

5. 在 FA 命中范围内倒序查 WA boundary
   for hit_blocks in range(fa_hit_blocks, -1, -1):
       key = external_keys[hit_blocks - 1]
       if wa_store.lookup([key])[0]:
           external_hit_blocks = hit_blocks
           break

6. 最终命中
   total_hit_block_num = hbm_hit_block_num + external_hit_blocks
   external_hit_tokens = external_hit_blocks * 512
```

关键点：

- FA 命中很多，不代表最终 external hit 就很多。
- WA store 必须在 FA 命中范围内找到某个最终 boundary。
- 如果 FA 连续命中 100 个 block，但 WA 只在第 80 个 boundary 存在，最终 external hit 只能算 80 个 block。
- 如果 WA 完全 miss，最终 external hit 是 0。

相关代码：

`@ucm/integration/vllm/hma_connector.py`

`@ucm/integration/vllm/hma_connector.py`

## Load 模式

Worker 侧 `start_load_kv` 收到 scheduler 下发的 dispatch metadata 后，会分别提交 FA load 和 WA load。

同一个请求：

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

FA load 是逐 external-hit canonical block 加载；WA load 只加载最后一个 boundary。

这正好对应论文中的两类 cache：

- classical/compressed KV prefix 可以按连续 prefix rows 复用。
- state cache 只需要恢复最终 boundary state。

相关代码：

`@ucm/integration/vllm/hma_connector.py`

## Dump/Save 模式

保存路径和 load 路径保持语义对称。

FA dump：

- 保存每个完整 canonical block 对应的 FA row。
- TP 多卡时，按 canonical block index 分片给不同 TP rank dump，避免每个 rank 重复 dump 全部 FA row。

WA dump：

- 只保存请求最终 boundary 的 WA tail row。
- TP 多卡时，按 request boundary round-robin 分配给 rank 保存。

也就是说：

```text
FA: 一个请求完成了多个 512-token canonical blocks，就可能 dump 多行。
WA: 一个请求只 dump 最终 boundary 的一行。
```

相关代码：

`@ucm/integration/vllm/hma_connector.py`

## 当前日志下的 Row Size 和 IO

### FA Store Row

当前 worker 侧 FA store summary：

```text
tensor_count = 83
tensor_bytes = 3,183,872 bytes
ShardSize = 3,186,688 bytes
```

FA 的 tensor list 形态：

```text
131072 x 21
16384  x 21
256    x 21
4096   x 20
```

拆分：

```text
group 0: 21 x 131072 = 2,752,512
group 3: 21 x (16384 + 256) = 349,440
group 8: 20 x 4096 = 81,920
FA tensor_bytes = 3,183,872
FA ShardSize = round_up(3,183,872, 4096) = 3,186,688
```

所以：

```text
FA_row_h2d_bytes = 3,183,872
FA_row_storage_bytes = 3,186,688
```

### WA Store Row

当前 worker 侧 WA store summary：

```text
tensor_count = 127
tensor_bytes = 6,496,256 bytes
ShardSize = 6,496,256 bytes
```

WA 的 tensor list 形态：

```text
131072 x 43
16384  x 42
4096   x 42
```

拆分：

```text
group 1 + group 2: 43 x 131072 = 5,636,096
group 4 + group 5: 42 x 16384 = 688,128
group 6 + group 7: 42 x 4096 = 172,032
group 9 + group 10: tail_tokens=0, no IO
WA tensor_bytes = 6,496,256
WA ShardSize = 6,496,256
```

所以：

```text
WA_row_h2d_bytes = 6,496,256
WA_row_storage_bytes = 6,496,256
```

### Load IO 公式

假设：

```text
N = external_hit_blocks
FA = 3,183,872
WA = 6,496,256
```

则有效 H2D IO 为：

```text
H2D bytes = N * FA + WA
```

这里默认 `N > 0`。如果 `external_hit_blocks = 0`，说明外部缓存没有可复用 boundary，FA/WA load 都不会提交。

如果看 CacheStore row 管理或 backend row IO：

```text
Cache row bytes = N * 3,186,688 + 6,496,256
```

示例：

| 外部命中 block 数 | FA 有效 H2D | WA 有效 H2D | 总有效 H2D |
| --- | --- | --- | --- |
| 1 | 约 3.04 MiB | 约 6.20 MiB | 约 9.23 MiB |
| 10 | 约 30.36 MiB | 约 6.20 MiB | 约 36.56 MiB |
| 100 | 约 303.64 MiB | 约 6.20 MiB | 约 309.83 MiB |

这里的 block 是 512-token canonical block，不是 vLLM 物理 KV block，也不是 CacheStore byte block。

### CacheStore 一次 load 的 IO 形态

当单个 request 外部命中 `N` 个 512-token canonical blocks 时，FAWA connector 会提交两个 CacheStore load task：

| load task | key/row 数 | 每 row 的 device pointer 数 | 每 row 的有效 H2D bytes | 每 row 的 CacheStore shard bytes |
| --- | --- | --- | --- | --- |
| FA load | `N` | 83 | 3,183,872 | 3,186,688 |
| WA load | 1 | 127 | 6,496,256 | 6,496,256 |

这里的 `key/row 数` 到 CacheStore C++ 里就是 `TaskDesc` 里的 shard 数。一个 shard 对应一个 CacheStore row：它有一个 block key、一个 shard index，以及一组 device-side addresses。LoadQueue 会逐 shard 取 host-side row buffer，再把这个 row buffer 按 `tensor_size_list` scatter 到该 row 的 device addresses。

所以命中 `N` 个 blocks 时，CacheStore 层面的 row 数是：

```text
FA rows = N
WA rows = 1
total rows = N + 1
```

当前日志是 `H2DTransport to ce`，所以真实 H2D 不是“一次 3MB/6MB 大 memcpy”，而是按 fragment 循环提交 async copy。每个 row 的 fragment 桶如下：

| row 类型 | 128KB fragment | 16KB fragment | 4KB fragment | 256B fragment | 总 fragment/copy 数 |
| --- | --- | --- | --- | --- | --- |
| FA row | 21 | 21 | 20 | 21 | 83 |
| WA row | 43 | 42 | 42 | 0 | 127 |

因此命中 `N` 个 blocks 时，CE H2D copy 数为：

```text
128KB copy 数 = 21N + 43
16KB copy 数  = 21N + 42
4KB copy 数   = 20N + 42
256B copy 数  = 21N

总 async H2D copy 数 = 83N + 127
```

如果把 128KB fragment 叫“大 IO”，把 16KB、4KB、256B 都归为“小 IO”，则：

```text
大 IO 数 = 21N + 43
小 IO 数 = 62N + 84
```

例子：

| 外部命中 block 数 | CacheStore load task | row 数 | 128KB 大 IO | 小 IO | 总 async H2D copy | 总有效 H2D |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | FA 1 个 + WA 1 个 | 2 | 64 | 146 | 210 | 9,680,128 bytes |
| 10 | FA 1 个 + WA 1 个 | 11 | 253 | 704 | 957 | 38,334,976 bytes |
| 100 | FA 1 个 + WA 1 个 | 101 | 2,143 | 6,284 | 8,427 | 324,883,456 bytes |

注意这里的“大 IO/小 IO”是 CE H2D fragment 口径；CacheStore row/backend 口径又是另一层：

```text
FA backend/cache row bytes = 3,186,688
FA effective H2D bytes     = 3,183,872
FA row padding             = 2,816

WA backend/cache row bytes = 6,496,256
WA effective H2D bytes     = 6,496,256
```

也就是说，CacheStore 管理 FA row 时按 4KB 对齐后的 `ShardSize` 走；但 CE H2D 真正 scatter 到 device 的只有 `tensor_size_list` 里列出的 tensor payload，不会把 FA row 的 padding 搬到 device。

如果 CacheStore host buffer 中该 row 已经 ready，LoadQueue 直接从 cache buffer 做 H2D。若 row 不 ready 且有 backend，LoadQueue 会先按 row 向 backend 提交一次 load，把 backend 数据回填到 cache buffer，再做 H2D。当前日志的 `Cache|Empty` 配置下，EmptyStore 不提供持久化 payload；能命中的 external row 必须来自同一 `unique_id` 下已经可见的 CacheStore shared buffer。

相关代码：

`@ucm/integration/vllm/hma_connector.py`

`@ucm/store/pipeline/connector.py`

`@ucm/store/pipeline/cpy/pipeline_store.py.cc`

`@ucm/store/cache/cc/cache_store.cc`

`@ucm/store/cache/cc/trans_manager.h`

`@ucm/store/cache/cc/load_queue.cc`

## H2D Transport 对 IO 的影响

`tensor_size_list` 决定一个 row 会拆成多少 fragment。Ascend DSV4 的 FA/WA row 中会出现 128KB、16KB、4KB、256B 等混合 fragment。

当 `H2DTransport=ce` 时，LoadQueue 按 fragment 循环提交普通 H2D async copy。上面的 `83N + 127` 就是当前日志下的实际 CE async copy 数。

当 `H2DTransport=ffts_pipeline` 时，LoadQueue 将整个 row 作为 object 提交给 FFTS pipeline：

```text
objectBytes = sum(tensor_size_list)
maxFragments = len(tensor_size_list)
```

这时 CacheStore 的提交粒度变成 row object：命中 `N` 个 blocks 时是 `N` 个 FA objects 加 1 个 WA object。每个 FA object 的 `objectBytes=3,183,872`、`maxFragments=83`；WA object 的 `objectBytes=6,496,256`、`maxFragments=127`。这样可以把 host 侧大 object staging 和 device 侧 fragment scatter 组织成 pipeline，目标是降低小 fragment 场景的下发开销。

需要注意：

```text
H2DFftsPipelineDepth
H2DFftsMaxReadyLanes
```

只是配置项打印，不代表已经启用 FFTS pipeline。

是否启用要看：

```text
H2DTransport == "ffts_pipeline"
```

若日志显示：

```text
H2DTransport to ce
```

则实际仍是普通 CE H2D。

相关代码：

`@ucm/store/cache/cc/cache_store.cc`

`@ucm/store/cache/cc/trans_manager.h`

`@ucm/store/cache/cc/load_queue.cc`

## 当前启动日志的几个判断点

### 1. 是否进入 FAWA

看日志：

```text
FAWA KV group config: ...
Init UCM FAWA connector.
```

只要 `UCMFAWAConnector.can_handle_kv_cache_config(kv_cache_config)` 返回 true，`UCMConnector` 会优先选择 FAWA connector。

因此即使配置里有：

```text
use_layerwise: True
```

只要 DSV4 Ascend KV layout 匹配，仍然会先走 FAWA，而不是普通 layerwise connector。

相关代码：

`@ucm/integration/vllm/ucm_connector.py`

### 2. Cache|Empty 的含义

如果配置里是：

```text
store_pipeline: Cache|Empty
share_buffer_enable: True
persist_token_threshold: 0
```

说明后端是 CacheStore 加 EmptyStore。

没有 PosixStore 这类持久化 backend 时，external hit 只能来自同一个 `unique_id` 下的 CacheStore 共享 host buffer。EmptyStore 本身不会提供持久化命中。

### 3. device_id=-1 的 scheduler store

日志里可能看到 scheduler/engine 侧创建 store：

```text
device_id = -1
TensorSizes = []
ShardSize = 0
BlockSize = 0
```

这不是 worker store 配置错误。

`device_id=-1` 的 CacheStore 主要用于 scheduler 侧 lookup，不负责 H2D load，因此不需要 tensor size、shard size、block size。

### 4. 8 个 worker rank 共享 namespace

当前日志显示 device id 从 0 到 7 都创建同样的 FA/WA store：

```text
unique_id = ..._fawa_fa
device_id = 0..7
local_rank_size = 8

unique_id = ..._fawa_wa
device_id = 0..7
local_rank_size = 8
```

这符合 DSV4 MLA 的设计。

`local_rank_size=8` 表示这些 rank 属于同一个本地 TP/MLA rank group。`share_buffer_enable=True` 且 `unique_id` 相同，表示这些进程打开同一组逻辑共享 CacheStore namespace。每个 worker 仍然用自己的 `device_id` 负责 H2D 到本 rank device。

## 如何判断能命中多少 block

给定请求长度和 HBM 已计算 tokens：

```text
full_canonical_blocks = floor(request_tokens / 512)
hbm_hit_blocks = num_computed_tokens / 512
candidate_external_blocks = full_canonical_blocks - hbm_hit_blocks
```

最终能命中的 external blocks 不等于 candidate 数量，而是：

```text
external_hit_blocks =
    min(FA 连续 prefix 命中范围内，WA 有最终 boundary 的最大 block 数)
```

例子：

```text
请求 10000 tokens
full_canonical_blocks = floor(10000 / 512) = 19
num_computed_tokens = 0
candidate_external_blocks = 19

如果 FA 连续命中 19，WA 第 19 个 boundary 存在:
    external_hit_blocks = 19

如果 FA 连续命中 19，但 WA 只在第 12 个 boundary 存在:
    external_hit_blocks = 12

如果 FA 连续命中 19，但 WA 全 miss:
    external_hit_blocks = 0
```

如果命中刚好覆盖整个请求，代码会把 `external_hit_tokens` 减 1，避免调度侧认为整个请求已完全 computed；但 block 级的命中判断仍按 512-token canonical blocks 进行。

## 性能分析时应该看什么

对 Ascend DSV4，命中率不能只看“命中了多少 token”。

建议同时看：

- 是否进入 FAWA connector。
- `hash_block_size` 是否是 512。
- FA groups 和 WA groups 是否符合预期。
- FA store 是否连续 prefix 命中。
- WA store 是否存在最终 boundary key。
- FA row size 和 WA row size。
- 当前 H2D transport 是 `ce` 还是 `ffts_pipeline`。
- external hit 是否来自 CacheStore DRAM，还是还要从 PosixStore 等 backend 回填。

长 prompt、高命中场景下：

```text
N * FA_row_bytes
```

会成为主要 H2D IO。

低命中场景下，WA row 的固定成本更明显：

```text
WA_row_bytes = 6,496,256 bytes
```

## 排查建议

按下面顺序看日志：

```text
1. 是否进入 FAWA:
   Init UCM FAWA connector.
   FAWA KV group config: ... is_ascend_layout=True ...

2. FA/WA group 是否合理:
   fa_groups=[0, 3, 8]
   window_groups=[1, 2, 4, 5, 6, 7, 9, 10]
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

如果 FA hit 很多但 external hit 偏低，优先检查 WA store 是否缺少对应 boundary key。

如果 external hit 正常但 TTFT 仍高，优先估算：

```text
N * FA_row_h2d_bytes + WA_row_h2d_bytes
```

并确认当前是否仍在 CE 小 fragment 路径。

如果日志里只看到 FFTS pipeline 参数，但仍显示：

```text
H2DTransport to ce
```

说明 FFTS pipeline 没有实际启用。

## 关键代码索引

FAWA connector 识别 DeepSeek-V4 / Ascend DSV4 layout：

`@ucm/integration/vllm/hma_connector.py`

`@ucm/integration/vllm/hma_connector.py`

Group meta 初始化与 `token_block_size` 计算：

`@ucm/integration/vllm/hma_connector.py`

`@ucm/integration/vllm/hma_connector.py`

`@ucm/integration/vllm/hma_connector.py`

FA/WA store 创建：

`@ucm/integration/vllm/hma_connector.py`

`@ucm/integration/vllm/hma_connector.py`

`@ucm/integration/vllm/hma_connector.py`

tensor fragment size 计算：

`@ucm/integration/vllm/hma_connector.py`

`@ucm/integration/vllm/hma_connector.py`

`@ucm/integration/vllm/hma_connector.py`

FA/WA 命中逻辑：

`@ucm/integration/vllm/hma_connector.py`

`@ucm/integration/vllm/hma_connector.py`

FA/WA load：

`@ucm/integration/vllm/hma_connector.py`

FA/WA dump：

`@ucm/integration/vllm/hma_connector.py`

CacheStore H2D transport：

`@ucm/store/cache/cc/cache_store.cc`

`@ucm/store/cache/cc/trans_manager.h`

`@ucm/store/cache/cc/load_queue.cc`
