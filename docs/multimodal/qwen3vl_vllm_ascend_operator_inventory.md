# Qwen3-VL 在 vLLM Ascend 中的算子与缓存路径调研

调研版本：vLLM 0.21.0、vLLM Ascend 0.21.0rc1  
调研日期：2026-07-16

## 一、结论先行

1. 导师所说的“Qwen3-VL 还是使用通用 GQA 算子”，主要指语言模型 Decoder 的自注意力路径。更准确地说，GQA 是 Q 头数大于 KV 头数的注意力结构，不是只属于 decode 的一个独立阶段。Qwen3-VL 在 prefill 和 decode 都使用 GQA。
2. 当前 Ascend 后端的主路径是 `npu_fused_infer_attention_score`。它同时接收 Q 头数和 KV 头数，因此可以执行 MHA、GQA、MQA；并且覆盖无缓存 prefill、前缀缓存命中、chunked prefill 和大部分 decode。只有满足特定图模式且请求形状在 `pa_shape_list` 中时，纯 decode 才切换到 `_npu_paged_attention`。默认 `pa_shape_list` 为空，因此不能把“Qwen3-VL 的 GQA 算子”简单等同于 `_npu_paged_attention`。
3. Qwen3-VL-8B 的文本 Decoder 是 32 个 Q 头、8 个 KV 头，即每 4 个 Q 头共享一组 KV；Qwen3-VL-30B-A3B 是 32 个 Q 头、4 个 KV 头，即每 8 个 Q 头共享一组 KV。TP 切分后是本卡局部头数，但 GQA 关系不变或通过复制少量 KV 头保持。
4. 视觉塔内部没有 Decoder 式的逐层 KV cache。vLLM V1 缓存的是视觉塔最终产生的多模态 embedding，也就是 encoder output。图像 embedding 被送入 Decoder 后，才会在语言模型每一层生成 Decoder KV cache。
5. 因此“先不重点研究 vision encoder”在复用率高、输出较长的在线场景中是合理的：同一请求的 decode 不会重复执行视觉塔，相同多模态输入还可能命中 encoder cache。但是在大量不同图片、高分辨率图片或冷请求场景，视觉塔仍可能是明显开销，不能从研究结论中完全删除。
6. 如果课题仍然围绕 CANN 9.0.0 `BlockSparseAttention`，直接替换 Decoder GQA 并不合适：prefill 需要严格 causal 语义，decode 又依赖分页 KV cache，而该 BSA 接口不具备直接等价替换条件。它更容易接入无 causal、无分页 KV cache 的视觉注意力；如果坚持研究 Decoder，应把课题改成“支持 causal 与 paged KV 的稀疏 Decoder Attention”，而不是简单调用现有 BSA。

## 二、Qwen3-VL 的总体执行结构

Qwen3-VL 不是“视觉 Encoder + 文本 Decoder cross-attention”架构。它的融合过程是：

```text
图片/视频预处理
  -> Vision Transformer
  -> 视觉 embedding 与 DeepStack 特征
  -> 替换 prompt 中的视觉占位 token embedding
  -> Qwen3 Decoder-only LLM
  -> LM Head / Logits / Sampling
```

视觉特征进入 `inputs_embeds` 后，与文本 token 一起经过同一个语言 Decoder。模型中不存在一组可以单独替换的视觉—语言 cross-attention 算子。

一次典型请求可分成三个计算阶段：

| 阶段 | 主要计算 | 是否每个生成 token 执行 |
|---|---|---:|
| 多模态编码 | Vision Transformer、Patch Merger | 否，通常只在需要且未命中 encoder cache 时执行 |
| Decoder prefill | 整段文本和视觉 token 的 causal GQA | 否，主要在提示词阶段执行 |
| Decoder decode | 单步 Q 查询历史分页 KV cache | 是，每个生成 token 都执行 |

## 三、导师说的“通用 GQA 算子”具体是什么

### 3.1 模型层结构

Qwen3 Decoder 的每一层注意力接收独立的 `num_heads` 和 `num_kv_heads`：

```text
hidden_states
  -> QKV Parallel Linear
  -> Q/K RMSNorm
  -> MRoPE
  -> Attention(Q heads, KV heads)
  -> Output Projection
```

vLLM Ascend 针对 Qwen3-VL 做了一层模型补丁，将下面几个步骤融合为自定义算子：

```text
QKV split + Q RMSNorm + K RMSNorm + MRoPE
  -> triton_split_qkv_rmsnorm_mrope
```

然后仍然调用通用的 `self.attn(q, k, v)`。也就是说，Qwen3-VL 没有一个专属的 Decoder Attention Kernel；模型特有部分主要是 MRoPE 和前处理融合，真正的注意力交给 Ascend 通用 Attention Backend。

### 3.2 Ascend 运行时分派

| Attention 状态 | K/V 来源 | 当前主要算子 |
|---|---|---|
| `PrefillNoCache` | 本轮新算出的连续 K/V | `npu_fused_infer_attention_score` |
| `PrefillCacheHit` | 分页 KV cache，并带 block table | `npu_fused_infer_attention_score` |
| `ChunkedPrefill` | 新 K/V + 已有分页 KV cache | `npu_fused_infer_attention_score` |
| `DecodeOnly` 默认路径 | 分页 KV cache | `npu_fused_infer_attention_score` |
| `DecodeOnly` 可选路径 | 分页 KV cache | `_npu_paged_attention` |

`_npu_paged_attention` 不是默认无条件启用。当前代码要求：

- 必须是纯 decode；
- 不能启用 speculative decoding；
- 不能是 A5 路径；
- 图模式必须是 `FULL_DECODE_ONLY`；
- 当前 token/batch 形状必须出现在 `pa_shape_list`；
- 不能使用 sliding window。

所以对当前默认配置而言，更准确的表述是：

> Qwen3-VL 的 Decoder 在 Ascend 上使用支持 GQA 和分页 KV 的通用融合推理注意力算子；decode 在少数显式配置的形状下可切换到专用 PagedAttention。

## 四、缓存到底复用了什么

### 4.1 三类缓存不能混为一谈

| 缓存 | 缓存内容 | 避免的工作 | 是否是 KV cache |
|---|---|---|---:|
| MM processor cache | 图片处理、映射后的多模态输入数据 | 重复图片预处理和部分 processor 工作 | 否 |
| MM encoder cache | Vision Transformer 最终输出的视觉 embedding | 重复执行整个视觉塔 | 否 |
| Decoder KV / prefix cache | 语言 Decoder 每层的 K/V block | 重复执行已处理 prompt token 的 Decoder 注意力与前向 | 是 |

### 4.2 同一请求中的实际情况

第一次处理图像时，Vision Transformer 产生视觉 embedding。随后这些 embedding 被放到 prompt 的对应位置，并经过语言 Decoder prefill；此时每层的视觉 token K/V 与文本 token K/V 一起写入 Decoder KV cache。

进入逐 token decode 后：

- 不再执行 Vision Transformer；
- 不重新计算历史视觉 token 的 K/V；
- 当前 token 的 Q 直接查询包含视觉与文本上下文的历史分页 KV cache。

因此，如果研究目标是长输出阶段的持续吞吐和单 token 时延，Decoder GQA 确实比视觉塔更接近核心热路径。

### 4.3 跨请求复用并非无条件成立

相同图片可以按照多模态 hash 复用 encoder output，完整 prompt 还可能命中 prefix cache。但是否真正命中取决于缓存容量、淘汰、请求内容、服务配置和 worker 生命周期。冷图片或不同图片仍然需要重新执行视觉塔。

## 五、Qwen3-VL 涉及的主要算子清单

以下清单按模型语义和当前 Ascend 落地算子两层描述。量化模型会把部分普通 Linear 替换成对应量化矩阵乘算子，通信策略也会随 TP、DP、EP 配置变化。

### 5.1 图片与视频预处理

这部分通常运行在前端或 CPU processor，不属于 NPU 模型主图，主要包括：

- 图片/视频解码；
- resize、归一化；
- patch 排列与 `grid_thw` 计算；
- 视觉占位 token 和多模态位置构造；
- MRoPE position 生成。

### 5.2 Vision Transformer 算子

Qwen3-VL-8B 与 30B-A3B 使用相同类型的视觉骨干，主要算子为：

| 模块 | 模型算子 | Ascend 侧关键实现 |
|---|---|---|
| Patch Embedding | `Conv3dLayer`，kernel/stride 为时间 patch 和空间 patch 大小 | Conv3D 路径 |
| 位置编码 | 位置插值、位置 embedding 加法 | 张量索引、插值、Add |
| Vision Norm | `LayerNorm` | NPU LayerNorm 后端 |
| Vision QKV | `QKVParallelLinear` | 普通或量化 MatMul/Linear |
| Vision RoPE | 对 Q/K 应用旋转位置编码 | `npu_rotary_mul` |
| Vision Attention | 非 causal MHA，Q 头数等于 KV 头数 | BF16/FP16 常走 `_npu_flash_attention_unpad`；部分设备或 FP32 走 `npu_fusion_attention` |
| Vision O Projection | `RowParallelLinear` | MatMul/Linear + 可选 TP 通信 |
| Vision MLP | FC1、激活、FC2 | Linear + GELU 类激活 + Linear |
| 残差 | Attention/MLP 残差加法 | Add |
| Patch Merger | LayerNorm、patch reshape、FC1、GELU、FC2 | Norm + reshape + Linear + GELU + Linear |
| DeepStack | 中间视觉层 merger、特征拼接、送入前几个 Decoder 层 | Merger + Cat + Add |

视觉注意力不是 GQA。视觉 QKV 投影中 Q 头数与 KV 头数相同，属于 MHA；它使用 `MMEncoderAttention` 抽象，由 Ascend 替换为 `AscendMMEncoderAttention`。

### 5.3 Dense Qwen3-VL-8B Decoder 算子

每一层的主要顺序如下：

```text
Embedding / multimodal embedding merge
  -> RMSNorm 或 Add+RMSNorm
  -> QKV Linear
  -> fused split + Q/K RMSNorm + MRoPE
  -> reshape_and_cache
  -> GQA Attention
  -> O Projection
  -> Add+RMSNorm
  -> Gate/Up Linear
  -> SwiGLU
  -> Down Linear
```

对应的关键 Ascend 算子族包括：

| 功能 | 当前关键算子或路径 |
|---|---|
| RMSNorm | `npu_rms_norm` |
| 残差融合 Norm | `npu_add_rms_norm` |
| 普通 Linear | `unquantized_gemm` 或量化 Linear 实现 |
| QKV 拆分、Q/K Norm、MRoPE | `triton_split_qkv_rmsnorm_mrope` |
| MRoPE 非融合回退 | `npu_mrope` |
| 新 K/V 写入分页缓存 | `_npu_reshape_and_cache` |
| 通用 GQA Attention | `npu_fused_infer_attention_score` |
| 可选纯 decode PA | `_npu_paged_attention` |
| SwiGLU | `npu_swiglu` |
| TP 聚合 | AllReduce、ReduceScatter 或插件配置的通信融合路径 |
| 最终输出 | Final RMSNorm、Parallel LM Head、Logits Processor、Sampler |

### 5.4 Qwen3-VL-30B-A3B MoE 额外算子

30B-A3B 的视觉路径和 Decoder Attention 基本相同，主要差异是 dense FFN 被稀疏 MoE 替换：

```text
Router/Gate Linear
  -> Top-K Expert Selection
  -> Token Dispatch / Permute
  -> Expert Grouped MatMul 1
  -> SwiGLU
  -> Expert Grouped MatMul 2
  -> Token Combine / Unpermute
```

常见 Ascend 算子族包括：

- `moe_gating_top_k`；
- `npu_moe_init_routing`；
- `npu_moe_token_permute`、`npu_moe_token_unpermute`；
- `npu_moe_distribute_dispatch`、`npu_moe_distribute_combine`；
- `npu_grouped_matmul`；
- `npu_swiglu`；
- EP/TP 下的 AllToAll、AllGather、ReduceScatter 或 MC2 融合通信。

具体选择依赖是否启用 Expert Parallel、Sequence Parallel、量化和通信融合，不能把其中某一种 dispatch 路径当成所有部署的固定调用序列。

## 六、哪些算子才是课题可能的替换目标

### 6.1 从模型代码看，Attention 的替换点在哪里

模型层面的逻辑替换点是 Decoder 中的：

```text
self.attn(q, k, v)
```

但工程上不应只修改 Qwen3-VL 模型文件。Qwen3-VL、Qwen3 和 Qwen3-MoE 都进入通用 Ascend Attention Backend，因此真正需要改的是后端 Attention 的状态分派：针对 prefill、cache-hit、chunked prefill、decode 分别选择算子，并维护 KV cache 的写入和读取语义。

### 6.2 CANN 9.0.0 BlockSparseAttention 能否替换 Decoder GQA

| 目标 | 直接替换可行性 | 主要原因 |
|---|---:|---|
| Vision Attention | 较高 | 非 causal、没有分页 KV cache，语义更接近 BSA |
| Decoder prefill GQA | 低 | 必须保证 token 级严格 causal；仅块级下三角会在对角块内泄露未来 token |
| Decoder decode GQA | 很低 | Q 长度通常为 1，历史 K/V 位于分页缓存；BSA 不能直接读取 vLLM Paged KV |
| 视觉—语言 cross-attention | 不存在 | Qwen3-VL 使用 embedding 融合，没有该模块 |

因此，如果暂时不研究 Vision Encoder，需要接受一个结论：

> 现有 CANN 9.0.0 BSA 不是 Qwen3-VL Decoder 通用 GQA 的直接替代品。

如果想把研究重心放到语言 Decoder，更合理的技术方向是：

1. 研究支持 GQA、strict causal 和 block table 的 sparse fused-infer-attention；
2. 在 decode 中增加稀疏 KV block 选择，再接支持分页缓存的 Attention；
3. 研究视觉 token 在 Decoder KV 中的保留、压缩或选择策略；
4. 把 prefill 和 decode 分成两套算子与稀疏策略，而不是要求一个 BSA 接口覆盖全部阶段。

其中第 3 点最具有“多模态模型”特色：视觉 token 在进入 Decoder 后已经和文本 token 位于同一序列中，但可以利用多模态位置范围，设计“保留关键视觉块、压缩冗余视觉 KV、文本 KV 保持正常”的策略。它比直接把 Vision Attention 换成 BSA 更贴近 decode 热路径，但工作量和正确性风险也更高。

## 七、建议的研究与验证顺序

### 7.1 第一阶段：先确认性能占比

至少分开测试以下三类请求：

1. 冷图片、短输出：突出 Vision Encoder 和 prefill；
2. 相同图片重复请求：观察 encoder cache 与 prefix cache 命中；
3. 冷图片、长输出：观察 decode GQA 是否成为绝对主导。

建议通过 NPU profiler 分别统计：

- Vision `_npu_flash_attention_unpad` / `npu_fusion_attention`；
- Decoder `npu_fused_infer_attention_score`；
- `_npu_paged_attention` 是否真的出现；
- `_npu_reshape_and_cache`；
- Decoder Linear、RMSNorm、SwiGLU；
- MoE 场景的 routing、GMM 和通信。

### 7.2 第二阶段：决定课题边界

| 业务特征 | 更值得研究的方向 |
|---|---|
| 图片重复率高、输出长 | Decoder GQA / Paged KV / 稀疏 KV 选择 |
| 图片大多不同、分辨率高 | Vision BlockSparseAttention |
| 长多模态 prompt、输出较短 | Decoder prefill causal sparse attention |
| MoE 模型、并行规模大 | MoE routing、GMM、AllToAll 与 Attention 联合分析 |

### 7.3 当前建议

如果导师的原意是“Qwen3-VL 的语言侧仍然只是通用 GQA，没有针对多模态 token 做特别优化”，那么可形成下面这个课题：

> 面向 Qwen3-VL Decoder 的多模态感知 KV 稀疏选择与 GQA/PagedAttention 协同优化。

该方向不是直接套用 CANN 9.0.0 `BlockSparseAttention`，而是以多模态 token 区间为先验，在 Decoder 的 KV block 或 Attention block 上进行选择，并保持 causal、GQA 和分页缓存语义。

如果导师明确要求必须使用现有 `BlockSparseAttention` 算子，那么课题仍应优先落在 Vision Attention，并在报告中把 encoder cache 命中率作为收益边界，而不是把不兼容的 BSA 强行替换 Decoder Attention。

## 八、代码与资料依据

本地代码：

`@vllm-0.21.0/vllm/model_executor/models/qwen3_vl.py`

`@vllm-0.21.0/vllm/model_executor/models/qwen3.py`

`@vllm-0.21.0/vllm/model_executor/models/qwen3_moe.py`

`@vllm-0.21.0/vllm/model_executor/models/qwen2_5_vl.py`

`@vllm-0.21.0/vllm/model_executor/models/qwen2.py`

`@vllm-0.21.0/vllm/v1/core/sched/scheduler.py`

`@vllm-0.21.0/vllm/v1/worker/gpu_model_runner.py`

`@vllm-ascend-0.21.0rc1/vllm_ascend/patch/worker/patch_qwen3vl.py`

`@vllm-ascend-0.21.0rc1/vllm_ascend/attention/attention_v1.py`

`@vllm-ascend-0.21.0rc1/vllm_ascend/attention/utils.py`

`@vllm-ascend-0.21.0rc1/vllm_ascend/ops/mm_encoder_attention.py`

`@vllm-ascend-0.21.0rc1/vllm_ascend/device/device_op.py`

`@vllm-ascend-0.21.0rc1/vllm_ascend/ops/layernorm.py`

`@vllm-ascend-0.21.0rc1/vllm_ascend/ops/activation.py`

`@vllm-ascend-0.21.0rc1/vllm_ascend/ops/rotary_embedding.py`

`@vllm-ascend-0.21.0rc1/vllm_ascend/ops/fused_moe`

外部资料：

- [Qwen3-VL-8B-Instruct 配置](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json)
- [Qwen3-VL-30B-A3B-Instruct 配置](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct/blob/main/config.json)
- [vLLM EncoderCacheManager 文档](https://docs.vllm.ai/en/latest/api/vllm/v1/core/encoder_cache_manager/)
- [CANN 9.0.0 BlockSparseAttention README](https://gitcode.com/cann/ops-transformer/blob/9.0.0/attention/block_sparse_attention/README.md)

