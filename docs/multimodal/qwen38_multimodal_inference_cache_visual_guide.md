# Qwen3.8 多模态推理与缓存复用图解

> 版本口径：vLLM 0.23.0、vLLM Ascend 0.23.0、Qwen3.8-27B。本文只解释图片/视频怎样进入模型，以及 MM Processor Cache、本地 Encoder Cache、KV/Prefix Cache 和可选外部 EC 的关系。

## 1. 先看一个完整请求

一个请求可以把文本、图片和视频交错排列。Processor 会保留原始内容顺序，为每个媒体项生成对应的视觉占位区。

![文本、多图和视频交错请求](assets/qwen38-multimodal-cache/01-request-anatomy.svg)

全文使用下面这组尺寸帮助理解。它们是讲解示例，不是模型要求的固定输入尺寸；实际服务还会受到动态 Resize、最大像素数和抽帧策略影响。

| 内容项 | 讲解用输入 | Patch Grid | Vision处理的Patch数 | Language接收的视觉Token数 |
|---|---:|---:|---:|---:|
| 图片1 | `4096×4096` | `[1,256,256]` | 65536 | 16384 |
| 图片2 | `2048×2048` | `[1,128,128]` | 16384 | 4096 |
| 视频 | 8帧、`1024×1024` | `[4,64,64]` | 16384 | 4096 |

这里的两个核心公式是：

```text
Npatch = T × Hgrid × Wgrid
Nvis   = Npatch / spatial_merge_size² = Npatch / 4
```

## 2. 端到端推理流程

从执行位置看，整个请求可以分为四条泳道：客户端/API、CPU 上的 MM Processor、NPU 上的 Vision Encoder，以及 NPU 上的 Language Model。

![Qwen3.8多模态推理端到端泳道图](assets/qwen38-multimodal-cache/02-end-to-end-pipeline.svg)

需要抓住三个转换点：

1. 文本被 Tokenizer 转换为 `token_ids [Ntext]`；
2. 图片和视频被转换为 `pixel_values [Npatch,1536]`，再由 Vision Encoder 转换为 `visual embeddings [Nvis,5120]`；
3. 文本与视觉结果最终都具有相同的 hidden size `5120`，因此可以组成一条序列进入 Language Transformer。

## 3. 每一步的Tensor维度

下面的瀑布图把文本、两张图片和视频分开画出。虽然它们的原始形态和序列长度不同，但进入 Language Transformer 前的最后一维必须统一为 `5120`。

![文本、图片和视频Tensor维度瀑布图](assets/qwen38-multimodal-cache/03-tensor-waterfall.svg)

以 4096×4096 图片为例：

```text
原始RGB                  [4096,4096,3]
image_grid_thw           [1,256,256]
pixel_values             [65536,1536]
Patch Embed / Vision     [65536,1152]
2×2 Patch Merger         [16384,5120]
```

所以需要区分：Vision Encoder 实际处理了 65536 个 Patch，Language Model 最终接收了 16384 个视觉 Token。`2×2 Patch Merger` 将空间上相邻的四个 Patch 合并成一个 Language 视觉 Token。

视频多一个时间维度。示例中的 8 帧按 `temporal_patch_size=2` 形成 `T=4`，然后与图片共用 Vision Encoder 和空间 Patch Merger。

## 4. 文本与视觉结果怎样合并

Processor 先在 Prompt 中展开图片和视频占位区；Vision Encoder 完成后，再用每个媒体项的视觉 Embedding 替换它自己的占位位置。

![视觉Embedding按原位置回填](assets/qwen38-multimodal-cache/04-embedding-interleave.svg)

因此最终序列仍保持用户的原始表达顺序：

```text
文本A → 图片1 → 文本B → 图片2 → 文本C → 视频 → 最终问题
```

Language Transformer 接收到的是一个统一 Tensor：

```text
merged_inputs_embeds [S,5120]
```

从这一步开始，模型不再接收彼此分离的“图片 Tensor”和“文本 Tensor”；它处理的是一条按位置排列的统一 Embedding 序列。

## 5. 三级本地缓存和可选外部EC

三个本地缓存位于不同计算阶段，缓存的也不是同一份数据：

![三级缓存及可选外部EC的位置和容量](assets/qwen38-multimodal-cache/05-cache-layers.svg)

| 实现 | 默认容量 | 常驻位置 | 缓存内容 | 命中依据 | 命中后节省什么 |
|---|---|---|---|---|---|
| MM Processor Cache | 每个API/Engine进程4 GiB | CPU进程内存 | `pixel_values`、`grid_thw`、Prompt更新结果 | 进程内LRU存在`mm_hash` | 媒体解码、Resize、Normalize、Patchify |
| 本地 Encoder Cache | `max(max_num_batched_tokens, max_tokens_per_mm_item)`个Embedding Token | NPU HBM | `visual embeddings [Nvis,5120]` | 本地管理器记录`identifier/mm_hash` | Vision Encoder |
| KV/Prefix Cache | 由可用HBM和Cache Block配置决定 | NPU HBM Cache Block | Language各层KV或GDN状态 | Token内容及多模态身份组成的前缀Block Hash匹配 | 已命中前缀的Language Prefill |
| 外部 ECExampleConnector | 默认关闭；示例实现不限制总容量 | `/tmp`所在文件系统 | Encoder输出的Safetensors文件 | `<mm_hash>/encoder_cache.safetensors`存在 | Vision Encoder，但需要先加载回本地HBM |

原生 vLLM 的多模态 Prefix Block Hash 会纳入媒体身份及媒体在 Block 内的相对位置。因此“文本相同但图片不同”不应该被当成同一个多模态前缀命中。

### 5.1 MM Processor Cache

vLLM 0.23.0 默认使用进程内 LRU，容量是每个 API/Engine 进程 4 GiB。命中键 `mm_hash` 由模型身份、媒体内容或用户提供的 UUID，以及 Processor 参数共同决定。

本例中一张 4096×4096 图片的 `pixel_values [65536,1536]` 按 FP32 估算约为 384 MiB。4 GiB 理论上大约容纳十张这种图片，但实际还要扣除 Prompt 更新结果、其他 Tensor 和缓存管理开销。

### 5.2 本地 Encoder Cache

本地 Encoder Cache 使用视觉 Embedding Token 数定容，而不是直接配置 GiB：

```text
effective_encoder_cache_size
= max(max_num_batched_tokens, max_tokens_per_mm_item)
```

一张 4096×4096 图片产生 `image_embeddings [16384,5120]`，BF16 约为 160 MiB/rank。当 `max_num_batched_tokens <= 16384` 时，有效容量是 16384 个 Token，约等于只能保留一张这种规格的图片。这与“一张不同图片进入后，旧图 Encoder Cache 很容易被淘汰”的实测现象一致。

### 5.3 外部 ECExampleConnector

EC Connector 默认不启用。vLLM 0.23.0 中的示例实现把每个媒体项保存为：

```text
/tmp/<mm_hash>/encoder_cache.safetensors
```

它通过目标文件是否存在来判断命中。命中后仍要把视觉 Embedding 加载回当前实例的本地 HBM，因此外部 EC 节省的是 Vision Encoder 计算，并不是零开销命中。

示例 Connector 没有固定容量上限，也没有自动淘汰。`ec_buffer_size=1e9` 不是它的 1 GB 文件容量限制。`/tmp` 最终位于磁盘、容器文件系统还是 DRAM `tmpfs`，取决于机器的实际挂载方式；写文件前还会经过一次 Device→CPU 转换。

## 6. 第二次请求怎样命中

重复请求到来时，不是查到一个缓存就代表所有阶段都命中。Processor、Encoder 和 Language Prefix 分别维护自己的缓存状态。

![第二次相同请求的缓存命中决策](assets/qwen38-multimodal-cache/06-cache-hit-decision.svg)

最重要的三个结论是：

1. Prefix Cache 命中并不自动代表 MM Processor Cache 或 Encoder Cache 命中；
2. 本地 Encoder Cache miss 后，只有配置了 EC Connector 才会继续查外部 Encoder Cache；
3. 外部 EC 命中后需要把结果恢复到本地 HBM，随后才能跳过 Vision Encoder。

## 7. 一页总结

```text
文本 ─→ Token IDs ─→ Text Embedding ─────────────────────────┐
                                                             │
图片/视频 ─→ MM Processor ─→ Vision Encoder ─→ Visual Embedding
                │缓存1             │缓存2          │          │
                ▼                  ↕              │          │
        预处理结果·CPU       可选外部EC·文件系统    │          │
                                                   ▼          ▼
                                        统一序列 [S,5120]
                                                   │
                                                   ▼
                                        Language Transformer
                                                   │缓存3
                                                   ▼
                                            KV/GDN Blocks
                                                   │
                                                   ▼
                                                Decode
```

- 图片和视频先变成视觉 Token，再按原始位置与文本组成统一序列；
- MM Processor Cache、本地 Encoder Cache 和 KV/Prefix Cache 分别复用三个不同阶段；
- 可选外部 EC 是 Encoder 输出的外部副本，用于扩大跨请求或跨实例的视觉结果复用范围。
