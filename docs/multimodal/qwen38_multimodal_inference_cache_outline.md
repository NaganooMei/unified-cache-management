# Qwen3.8 多模态推理与三级缓存可视化文档大纲

> 可视化成稿：[Qwen3.8 多模态推理与缓存复用图解](qwen38_multimodal_inference_cache_visual_guide.md)

## 一、文档定位

### 1.1 目标

通过一个包含文本、多张图片和视频的完整请求，让不了解多模态推理的读者直观看懂：

- 文本、图片和视频最初是什么形态；
- MM Processor 如何把图片和视频转换成模型可以处理的 tensor；
- Vision Encoder 如何生成视觉 token；
- 视觉 token 如何和文本 token 拼接成统一序列；
- Language Transformer 最终接收到什么；
- MM Processor Cache、Encoder Cache、KV/Prefix Cache 分别缓存哪一步的结果、存在哪里、容量大致有多大。

### 1.2 文档边界

正文只讲以下内容：

1. 多模态请求的端到端推理流程；
2. 每一步输入输出 tensor 的维度变化；
3. 文本与多图、视频的融合方式；
4. 三级缓存的对象、位置、默认容量和命中判断；
5. 第二次相同请求的缓存复用流程。

不展开性能实测、UCM、EPD部署拓扑、网络传输后端、算子优化和源码细节。缓存章节只补充 vLLM 0.23.0 中 `ECExampleConnector` 的单机外部缓存实现，用来说明它和本地 Encoder Cache 的区别。

---

## 二、示例请求：文本 + 多图 + 视频

设计一个贯穿全文的请求，内容顺序固定为：

```text
系统提示词
→ 文本 A
→ 图片 1
→ 文本 B
→ 图片 2
→ 文本 C
→ 视频 1
→ 最终问题
```

示例内容：

```text
请比较下面两张图片，并结合视频说明场景发生了什么。
[图片1：道路全景]
第一张图片是事件发生前。
[图片2：车辆特写]
第二张图片是事件发生后。
[视频1：车辆移动过程]
请给出结论。
```

### 本章配图：原始请求内容图

用接近聊天界面的形式画出一条消息，按真实顺序排列文本框、图片缩略图和视频帧缩略图。

图中只标记：

- `Text A / Image 1 / Text B / Image 2 / Text C / Video 1 / Question`
- 图片原始尺寸，例如 `[H, W, 3]`
- 视频原始尺寸，例如 `[F, H, W, 3]`

---

## 三、多模态推理全流程

这是全文最重要的一章，用一张横向主流程图讲完整链路。

### 3.1 主流程

```text
原始请求
  ↓
文本分支                    图片/视频分支
Tokenizer                   MM Processor
  ↓                           ↓
文本 Token IDs              pixel_values / grid_thw
  │                           ↓
  │                         Vision Encoder
  │                           ↓
  │                         Visual Embeddings
  └──────────────┬────────────┘
                 ↓
       按原始内容顺序合并 Embeddings
                 ↓
          Language Transformer
                 ↓
              输出 Token
```

### 3.2 图中需要表达的并行关系

- 文本走 Tokenizer；
- 每张图片分别进行预处理；
- 视频先抽帧，再进行类似图片的预处理；
- 多张图片和视频可以在 Vision Encoder 前合批；
- Encoder 输出后重新按图片、视频 item 拆开；
- 最后按照原始请求中的位置插回文本序列。

### 本章配图：端到端推理泳道图

建议使用六条泳道：

1. Request；
2. Tokenizer；
3. MM Processor；
4. Vision Encoder；
5. Embedding Merge；
6. Language Transformer。

每一条箭头同时标注 tensor 名称和 shape，避免只有模块名、看不到数据变化。

---

## 四、每一步 tensor 维度怎么变化

## 4.1 文本分支

```text
原始文本
  ↓ Tokenizer
input_ids: [Ntext]
  ↓ Text Embedding
text_embeddings: [Ntext, 5120]
```

说明：

- `Ntext` 是文本经过 Chat Template 和 Tokenizer 后的 token 数；
- `5120` 是 Qwen3.8-27B Language Transformer 的 hidden size。

## 4.2 单张图片分支

先给通用公式：

```text
原始图片
[H, W, 3]

Resize / Normalize
[H', W', 3]

Patchify
grid_thw = [1, H'/16, W'/16]

pixel_values
[Npatch, 1536]

其中：
Npatch = 1 × H'/16 × W'/16
1536 = 3 × 2 × 16 × 16
```

再给一个 4096×4096 的具体例子：

```text
原始 RGB                  [4096, 4096, 3]
image_grid_thw            [1, 256, 256]
pixel_values              [65536, 1536]
Patch Embed 输出          [65536, 1152]
Vision Transformer 输出  [65536, 1152]
2×2 Patch Merger 输出     [16384, 5120]
```

图中突出两个数字：

```text
Vision Encoder处理：65536个patch
Language Model接收：16384个视觉token
```

## 4.3 多张图片

以两张不同大小图片为例：

```text
Image 1 pixel_values: [Npatch_1, 1536]
Image 2 pixel_values: [Npatch_2, 1536]

Vision Encoder合批输入：
[Npatch_1 + Npatch_2, 1536]

Vision Encoder输出后拆分：
Image 1: [Nvis_1, 5120]
Image 2: [Nvis_2, 5120]
```

其中：

```text
Nvis_i = Npatch_i / 4
```

## 4.4 视频分支

视频只讲必要过程：

```text
原始视频
[F, H, W, 3]
  ↓ 抽帧
[F', H, W, 3]
  ↓ Resize / Normalize / Patchify
video_grid_thw = [T, H'/16, W'/16]
pixel_values_videos = [Npatch_video, 1536]
  ↓ Vision Encoder + Patch Merger
video_embeddings = [Nvis_video, 5120]
```

说明：视频和图片共用 Vision Encoder，主要区别是视频具有时间维度 `T`。

### 本章配图：Tensor 维度瀑布图

一张图分成三列：

- 文本；
- 图片 1 / 图片 2；
- 视频。

三列最后都汇聚到宽度为 `5120` 的 embedding，直观说明为什么它们可以进入同一个 Language Transformer。

---

## 五、文本和视觉信息如何合并

## 5.1 Prompt 中的占位位置

MM Processor 会把每个图片或视频占位符展开成相应数量的视觉 token 位置：

```text
[Text A]
[Image 1 Placeholder × Nvis_1]
[Text B]
[Image 2 Placeholder × Nvis_2]
[Text C]
[Video Placeholder × Nvis_video]
[Question]
```

## 5.2 Embedding 替换

先生成统一的文本 embedding：

```text
inputs_embeds: [S, 5120]
```

再把图片、视频占位位置替换为 Vision Encoder 输出：

```text
Image 1位置 ← image_embeddings_1 [Nvis_1, 5120]
Image 2位置 ← image_embeddings_2 [Nvis_2, 5120]
Video位置   ← video_embeddings   [Nvis_video, 5120]
```

最终仍然得到：

```text
merged_inputs_embeds: [S, 5120]
```

Language Transformer 从这一刻开始不再区分“文本 tensor”和“图片 tensor”，看到的是一条按原始内容顺序排列的统一 embedding 序列。

### 本章配图：Embedding 拼接长条图

使用不同颜色表示：

- 蓝色：文本 embedding；
- 绿色：Image 1 embedding；
- 橙色：Image 2 embedding；
- 紫色：Video embedding。

图的最终形态：

```text
| Text A | Image 1 | Text B | Image 2 | Text C | Video | Question |
                         [S, 5120]
```

---

## 六、三级缓存分析

三级缓存按推理阶段划分。需要明确：它们缓存的是三种不同的数据，不是同一份数据的三层副本。

## 6.1 MM Processor Cache

```text
缓存位置：CPU内存
缓存内容：pixel_values、grid_thw和Prompt更新结果
典型shape：[Npatch, 1536]
命中后跳过：图片解码、Resize、Normalize、Patchify
```

4096×4096图片示例：

```text
pixel_values [65536,1536]
FP32约384 MiB
```

## 6.2 Encoder Cache

```text
缓存位置：NPU HBM
缓存内容：Vision Encoder输出
典型shape：[Nvis, 5120]
命中后跳过：完整Vision Encoder
```

4096×4096图片示例：

```text
image_embeddings [16384,5120]
BF16约160 MiB/rank
```

## 6.3 KV/Prefix Cache

```text
缓存位置：NPU HBM Cache Block
缓存内容：Language Transformer各层的KV或GDN状态
组织方式：Physical Block + Request Block Table
命中后跳过：已命中前缀对应的Language Transformer Prefill
```

说明：Prefix Cache是对已有KV/GDN Block进行Hash查找和跨请求复用，不是另一份独立的KV副本。

## 6.4 当前 MM Cache、Encoder Cache 与外部 EC Cache 的实现

这一节先统一术语，避免把两个都带有“Encoder Cache”含义的组件混在一起：

- **MM Processor Cache**：缓存媒体预处理结果；
- **本地 Encoder Cache**：缓存当前实例正在复用的视觉 Embedding，位于 NPU HBM；
- **外部 EC Cache**：由可选的 EC Connector 保存 Encoder 输出，用于在本地 Encoder Cache 未命中时恢复结果。它不是默认启用的第四级计算缓存，而是 Encoder 输出的外部副本。

### 6.4.1 MM Processor Cache

```text
默认状态：启用
默认类型：LRU
默认容量：每个API/Engine进程4 GiB
存储位置：对应进程的CPU内存
缓存键：由模型ID、媒体内容或用户提供的UUID、MM Processor参数共同计算的mm_hash
命中判断：mm_hash是否已经存在于进程内LRU中
命中结果：直接复用pixel_values、grid_thw和Prompt更新结果
```

需要在图下注明：4 GiB 是**每个进程**的默认预算，不是整个服务共享的总容量。若有多个 API Server 或 Data Parallel Engine，整体最大占用会随进程数增加。

对于本例的一张 4096×4096 图片：

```text
pixel_values [65536,1536]，FP32约384 MiB
4 GiB理论上约容纳10张，实际数量还会受其他缓存字段和管理开销影响
```

### 6.4.2 本地 Encoder Cache

```text
默认状态：随多模态调度启用
容量单位：视觉Embedding token数，不是GiB
有效容量：max(max_num_batched_tokens, max_tokens_per_mm_item)
存储位置：当前推理实例的NPU HBM
缓存键：多模态条目的identifier，通常就是mm_hash
命中判断：identifier是否存在于本地Encoder Cache管理器中
命中结果：从HBM复用visual embeddings，跳过Vision Encoder
```

对于 Qwen3.8 的一张 4096×4096 图片：

```text
max_tokens_per_mm_item = 16384
image_embeddings [16384,5120]，BF16约160 MiB/rank

当max_num_batched_tokens不超过16384时：
本地Encoder Cache有效容量为16384个token，约等于一张该规格图片
```

### 6.4.3 外部 EC Cache：ECExampleConnector

```text
默认状态：关闭；未配置ec_connector时占用为0
默认示例实现：ECExampleConnector
默认目录：/tmp
存储格式：/tmp/<mm_hash>/encoder_cache.safetensors
容量限制：示例实现没有固定容量上限，也没有自动淘汰；由目录所在文件系统容量决定
命中判断：对应mm_hash的safetensors文件是否存在
命中结果：把文件中的visual embeddings加载回本地Encoder Cache，再跳过Vision Encoder
未命中结果：运行Vision Encoder，并按Connector角色决定是否写入外部缓存
```

配置中的 `ec_buffer_size=1e9` 不能理解为 `ECExampleConnector` 默认只占 1 GB；该参数是给需要传输缓冲区的 Connector 使用的，示例文件 Connector 并不用它限制 `/tmp` 中的文件总量。

图中需要特别标注：`/tmp` 不必然代表物理磁盘。如果系统把 `/tmp` 挂载为 `tmpfs`，数据实际由 DRAM/Swap 承载；如果是 ext4、XFS 或容器文件系统，则落在相应文件系统。保存时还会先执行一次 Device→CPU 转换，因此即使最终落盘，也会产生临时 CPU 内存和操作系统页缓存占用。

对于本例的一张 4096×4096 图片，单个 Encoder Cache 文件约为 160 MiB；16 张不同图片约为 2.5 GiB，未计文件头和文件系统开销。

### 6.4.4 当前实现对比表

| 实现 | 默认容量 | 常驻位置 | 如何判断命中 | 命中后复用什么 |
|---|---:|---|---|---|
| MM Processor Cache | 每个进程4 GiB | CPU内存 | 进程内LRU是否存在`mm_hash` | 预处理tensor和Prompt更新结果 |
| 本地Encoder Cache | `max(max_num_batched_tokens, max_tokens_per_mm_item)`个token | NPU HBM | 本地管理器是否记录该`identifier/mm_hash` | visual embeddings |
| 外部EC Cache | 默认关闭；示例实现不设容量上限 | `/tmp`所在文件系统 | `<mm_hash>/encoder_cache.safetensors`是否存在 | 加载visual embeddings回本地HBM |

### 6.4.5 三种实现的命中关系图

```text
同一媒体内容
    ↓ 计算mm_hash
MM Processor Cache中存在？
    ├─ 是：复用预处理结果
    └─ 否：重新预处理并写入CPU LRU
    ↓
本地Encoder Cache中存在？
    ├─ 是：直接复用HBM中的visual embeddings
    └─ 否：若配置EC Connector，检查外部文件
              ├─ 文件存在：加载到本地HBM后复用
              └─ 文件不存在：运行Vision Encoder，并可写入外部缓存
```

相同的 `mm_hash` 可以贯穿多模态预处理、本地 Encoder Cache 和外部 EC Cache，但三个位置保存的数据对象不同；“Hash 相同”只表示媒体身份一致，不表示三个缓存一定同时命中。

## 6.5 三级缓存对比表

| 缓存 | 缓存的阶段输出 | 典型数据 | 位置 | 命中节省什么 |
|---|---|---|---|---|
| MM Processor Cache | 预处理输出 | `[Npatch,1536]` | CPU | 图片/视频预处理 |
| Encoder Cache | Vision输出 | `[Nvis,5120]` | NPU HBM | Vision Encoder |
| KV/Prefix Cache | Language层状态 | KV、Conv/SSM State | NPU HBM Block | Language Prefill |

### 本章配图：三级缓存位置与大小关系图

沿推理链路放置三个缓存框：

```text
原始媒体
  ↓
MM Processor
  ├─ MM Processor Cache：CPU，4K图片约384 MiB
  ↓
Vision Encoder
  ├─ Encoder Cache：HBM，4K图片约160 MiB/rank
  ├─ 可选外部EC：/tmp下的safetensors文件，4K图片约160 MiB
  ↓
Language Transformer
  └─ KV/Prefix Cache：HBM Block，随模型层数和序列长度增长
```

---

## 七、第二次相同请求如何复用

只画一个简洁的命中流程：

```text
第二次相同请求
  ↓
MM Processor Cache命中？
  ├─ 是：复用pixel_values/grid_thw
  └─ 否：重新预处理
  ↓
构造统一Prompt和多模态身份
  ↓
Prefix Cache命中？
  ├─ 是：复用已命中的Language状态Block
  └─ 否：需要Language Prefill
  ↓
未命中计算区间是否包含图片/视频？
  ├─ 否：不需要Vision Encoder
  └─ 是：查询本地Encoder Cache
          ├─ 命中：复用HBM中的visual embeddings
          └─ 未命中：是否配置EC Connector？
                       ├─ 否：运行Vision Encoder
                       └─ 是：外部EC Cache命中？
                                ├─ 是：加载到本地HBM后复用
                                └─ 否：运行Vision Encoder，并可写入外部缓存
```

本章只保留三个结论：

1. 三个本地缓存命中的是不同阶段，必须分别判断；
2. Prefix Cache命中并不自动代表MM Processor Cache或Encoder Cache命中；
3. 外部EC命中后仍需把visual embeddings加载回本地HBM，它节省的是Vision Encoder计算，不等于零开销。

### 本章配图：第二次请求缓存决策图

使用绿色表示命中，灰色表示跳过，红色表示重新计算，虚线表示可选的外部EC分支。图中不展开UCM、EPD拓扑和网络传输。

---

## 八、全文总结图

最后用一页汇总：

```text
文本 → Token IDs ───────────────────────────────┐
                                               │
图片/视频 → MM Processor → Vision Encoder ─────┤
              │缓存1          │缓存2            │
              ▼               ▼                 ▼
          Processor输出    Visual Embeddings  统一Embedding序列
                                  ↕
                              可选外部EC
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

总结只保留三句话：

1. 图片和视频先被转换成视觉token，再与文本token组成统一序列；
2. 三种模态最终都必须转换成相同的Language hidden size；
3. 三级本地缓存分别复用预处理结果、视觉Encoder结果和Language状态；可选外部EC用于在本地Encoder Cache未命中时恢复视觉Encoder结果。

---

## 九、建议的最终图表清单

正文共保留六张图和两张表：

1. 文本 + 两图 + 视频的原始请求内容图；
2. 多模态推理端到端泳道图；
3. 文本、图片、视频Tensor维度瀑布图；
4. 文本与多模态Embedding拼接长条图；
5. 三级缓存及可选外部EC的位置与大小关系图；
6. 第二次相同请求的缓存决策图；
7. 三级缓存对比表；
8. MM Cache、本地Encoder Cache、外部EC Cache实现对比表。

建议全文控制在 8～10 页，以图为主、文字为辅，每张图下只解释“输入是什么、输出是什么、维度怎么变化、这一步能被哪个缓存复用”。
