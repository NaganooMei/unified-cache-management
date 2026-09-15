# Cache 内存按 Rank 划分、NUMA 绑定与 Lookup 首层预取

## 1. 背景

H2D KV Cache 传输时延是影响 UCM 性能的重要因素。我们先后探索了 A2 循环提交小 I/O PCIe 拷贝、A2 I/O 聚合和 A3 SDMA Direct。通过聚合小 I/O 并使用 SDMA，以 GLM-5.1 为例，单卡加载一层 KV Cache 的带宽已从约 4 GB/s 提升到约 30 GB/s。

但在实际业务中，多卡需要同时从一块共享内存读取同一份 KV Cache。早期版本的单机聚合带宽只有约 100 GB/s，平均单卡约 7 GB/s。继续增加拷贝 stream 不仅没有提升带宽，有时还会出现异常劣化，说明瓶颈已经从拷贝接口转移到 Host 内存侧。

加入 `MAP_POPULATE` 后，多进程并发预取共享页面，使部分物理页无意中分散到不同 NUMA 节点，性能有所提升。但这种分布受进程调度和页面预取时序影响，随机且不均匀，导致带宽波动，也无法稳定达到预期的 350 GB/s 以上。结合内存控制器监控，我们确认多卡流量仍集中在少数 NUMA 节点，并触及单个 NUMA 节点约 100 GB/s 的带宽上限。

此外，按层加载虽然可以让后续层的数据传输与前序层计算重叠，第一层却没有这样的重叠窗口。如果 Lookup 只确认下层存储命中，正式 Load 第一层时仍要等待“下层存储到 Host Cache”的读取，这段等待会进入 TTFT 关键路径。

因此，本方案包含两项互补优化：

- 将共享数据区按 rank 分段，并将不同数据段精确放置到不同 NUMA 节点，突破单个内存控制器的带宽瓶颈；
- 在 `LookupOnPrefix` 确认下层存储命中后，后台预取每个 block 的首个 shard，利用 Lookup 到正式 Load 之间的时间隐藏下层读取时延。

## 2. 控制区与按 Rank 分段的数据区

### 2.1 内存布局

对外仍然提供一个逻辑 Cache，内部由一个共享控制区和多个 rank 数据段组成。控制区不是独立的控制进程，而是所有参与进程共同映射的一块共享元数据区。它不存放 KV 数据，主要负责：

- 将 `(blockId, shardIndex)` 映射到 `globalSlot`；
- 维护每个 slot 的 `Loading / Ready / Failed` 状态、引用计数和 CLOCK 访问标记；
- 记录各数据段的初始化状态，避免访问尚未就绪的数据段；
- 保存 slot 大小、数据段数量和 NUMA 节点列表，校验各参与进程使用相同布局；
- 保存每个在线 rank 的有界预取命令环。

KV 数据按本机参与共享的 rank 数切分到多个独立数据段。每个 rank 负责创建、初始化并注册自己的数据段，然后发布就绪状态。其他 worker 在数据段就绪后映射它，从而在每个进程中都能解析任意全局 slot。

~~~text
逻辑 Cache
  ├─ 共享控制区：全局索引、slot 生命周期、rank 状态、预取命令环
  ├─ Rank 0 数据段：KV payload -> 指定 NUMA 节点或节点组
  ├─ Rank 1 数据段：KV payload -> 指定 NUMA 节点或节点组
  ├─ Rank 2 数据段：KV payload -> 指定 NUMA 节点或节点组
  └─ Rank N 数据段：KV payload -> 指定 NUMA 节点或节点组
~~~

全局 slot 编号覆盖所有数据段，并可稳定解析为实际数据段和段内位置：

~~~text
segment = globalSlot / slotsPerSegment
localSlot = globalSlot % slotsPerSegment
~~~

共享控制区不保存进程相关的虚拟地址。各进程把 `globalSlot` 解析为自己的 Host 地址或 Device 可访问地址，因此不同进程可以使用不同虚拟地址访问同一个物理数据段，同时共享一致的 Cache 索引和状态。

### 2.2 KV 放置与并行加载

- MLA：由 worker 0 执行 Dump 时，按任务中分片的原始位置 `originalIndex % segmentCount` 选择优先数据段。同一层不同 block 的 KV 会分散存放，避免全部写入 rank 0 数据段。
- 已缓存的 `(blockId, shardIndex)` 始终使用当前实际位置，不因后续请求来自不同 rank 而迁移。
- 目标数据段没有可回收 slot 时，允许按段顺序回退到其他已就绪数据段；实际位置仍以 `globalSlot` 为准。

`LoadQueue` 在处理任务前根据当前 rank 重排分片顺序。对于 `segmentCount` 个数据段，rank 0 优先处理序号为 `0, segmentCount, 2 * segmentCount...` 的分片，rank 1 优先处理序号为 `1, segmentCount + 1...` 的分片，以此类推。重排只改变读取顺序，不改变数据位置，也不表示 rank 之间存在同步屏障。

以 4 个 rank、4 个数据段为例：

| 加载阶段 | Rank 0 优先读取 | Rank 1 优先读取 | Rank 2 优先读取 | Rank 3 优先读取 |
|---|---|---|---|---|
| 0 | 数据段 0 | 数据段 1 | 数据段 2 | 数据段 3 |
| 1 | 数据段 1 | 数据段 2 | 数据段 3 | 数据段 0 |
| 2 | 数据段 2 | 数据段 3 | 数据段 0 | 数据段 1 |
| 3 | 数据段 3 | 数据段 0 | 数据段 1 | 数据段 2 |

这样可以让多个 NUMA 节点在同一阶段并行供数。结合异步传输和多 stream，可同时利用多 NUMA 聚合带宽与 SDMA 并行传输能力。

### 2.3 分段 CLOCK 淘汰

每个数据段维护独立的 CLOCK 指针。分配新 slot 时，优先在目标数据段内扫描：近期访问过的 slot 获得一次保留机会，仍被 Handle 引用的 slot 不允许淘汰。

如果目标数据段经过两轮扫描仍找不到可回收 slot，则继续扫描其他已就绪数据段。该策略优先在目标 NUMA 数据段内完成淘汰和复用，同时在局部容量不足时允许跨段回退，避免单个数据段耗尽导致 Cache 分配失败。

## 3. NUMA 绑定

### 3.1 节点选择策略

最新实现不再只依赖进程运行在哪个 CPU 上触发 first-touch，而是先确定 Host 数据区应该使用的 NUMA 节点。选择顺序如下：

1. vLLM worker 通过 `npu-smi info -t topo` 获取 NPU 的 CPU Affinity，再用 `lscpu -e=cpu,node` 将这些 CPU 映射到 NUMA 节点。只有所有亲和 CPU 唯一落在同一 NUMA 节点时，才生成内部的设备亲和节点提示。该提示优先级最高，同时适用于 MLA 的共享 rank 数据段和 GQA 的私有 Buffer。
2. 无法得到唯一设备亲和节点时，共享 Buffer 使用 `share_buffer_numa_nodes`。配置为空时，程序取“有内存的在线节点”与当前进程 `Mems_allowed_list` 的交集。节点按数据段数分组，同一组中的页面在节点间均匀划分；rank、数据段与 NUMA 节点不要求一一对应。
3. 对没有可靠拓扑信息的 NPU GQA 私有 Buffer，按 TP rank 在当前允许使用的内存节点之间轮转，避免所有私有 Buffer 都依赖同一个 first-touch 节点。
4. 以上信息都不可用时，保留普通 first-touch 作为兜底。

拓扑探测只在 worker 进行。由连接器生成的设备亲和节点和 TP rank 回退提示属于内部策略，不作为用户配置项。显式节点若不在当前进程的 `Mems_allowed_list` 中，会在初始化阶段报错，而不是静默退回到随机放置。

### 3.2 绑定、触页与验证

每个 rank 创建本地数据段后，严格按以下顺序初始化：

1. 按系统页大小规划地址范围；一个数据段对应多个 NUMA 节点时，完整页面在节点间均分，页数差不超过一页。
2. 在任何页面被触碰前，对每个地址范围调用 `mbind(MPOL_BIND | MPOL_F_STATIC_NODES)`。数据映射不能提前使用 `MAP_POPULATE`，否则物理页可能在策略生效前已经分配。
3. 调用 `memset` 触发物理页分配。
4. 分批调用 `move_pages` 查询每个页面的实际节点；任一页面不在预期节点都视为初始化失败。
5. NUMA 校验成功后注册 Host Buffer，最后把本 rank 数据段发布为 Ready。其他 rank 只映射 Ready 的数据段。

日志中的 `SHM NUMA bind` 给出文件、偏移、长度和目标节点，`SHM NUMA verify` 给出期望节点、实际节点、页数和 mismatch 数。验证失败的数据段会发布失败状态，其他参与者不会把它当作可用数据段继续运行。

## 4. Lookup 首层预取

### 4.1 哪个 Connector 调用 Lookup

Lookup 由 vLLM scheduler 侧发起。新请求进入调度时，scheduler 先查询本地 HBM 命中，再调用外层 `UCMConnector.get_num_new_matched_tokens`；外层 Connector 只负责选择并转发给实际的内部 Connector。

普通 KV Cache 场景由 `use_layerwise` 选择内部 Connector：

| 配置 | 内部 Connector | Lookup 实现 |
|---|---|---|
| `use_layerwise: false` | `UCMDirectConnector` | `UCMDirectConnector.get_num_new_matched_tokens` |
| `use_layerwise: true` | `UCMLayerWiseConnector` | 继承 `UCMDirectConnector.get_num_new_matched_tokens`，没有单独覆盖 |

因此，整块存取和 LayerWise 都会走 `RankConsistencyManager.lookup_on_prefix -> store.lookup_on_prefix`。只要该调用最终进入 `CacheStore::LookupOnPrefix`，就会由 `BufferManager::LookupOnPrefixFast` 检查 Host Cache、查询下层存储，并对“Host Cache 未命中但下层前缀命中”的 block 触发预取。已经在 Host Cache 中、下层也未命中或没有 Ready worker 时，不会产生有效预取任务。

这里的 lookup 预取与显式 `store.prefetch()` 不是同一机制：前者把数据从下层存储提前读入 Host Cache；后者是下层存储的独立提示接口。

### 4.2 整块存取与 LayerWise 的预取含义

两种模式都会触发 lookup 预取，但 `shardIndex = 0` 表示的数据不同：

| 模式 | Store 中的布局 | lookup 预取的实际内容 |
|---|---|---|
| 整块存取（Direct） | 一个 UCM block 只有一个 shard，所有本地层的 KV 被展平到 `shard 0` | 预取整个 KV block，而不是只预取模型第 0 层 |
| LayerWise | 每个模型层分别作为一个 shard，正式按层访问使用实际 `layer_id` | 当前实现固定预取 `shard 0`，通常对应模型第 0 层 |

LayerWise 之所以优先预取第一层，是因为第一层开始计算前没有上一层计算可用于掩盖后端读取；后续层可以在前一层计算期间提前准备。vLLM 的 LayerWise Connector 在模型 forward 前提交本地第一层，进入每个 attention layer 时等待当前层并启动下一层，因此 lookup 阶段只需要抢先填充最难隐藏的第一层。

需要注意一个当前实现边界：`PrefetchQueue` 固定调用 `Buffer::Get(block, 0, false)`，而 `UCMLayerWiseConnector` 的正式按层访问使用 `first_layer_id` 和后续实际 `layer_id`。在常见的非流水线并行场景中 `first_layer_id = 0`，预取能够命中本地第一层；在 pipeline parallel 的非首 stage 中，本地 `first_layer_id` 可能大于 0，此时 `shard 0` 不是该 stage 的本地第一层，当前预取不能覆盖它。若目标语义是“每个 PP stage 的本地第一层”，预取命令还需要携带对应的 `first_layer_id`，不能把 shard 固定为 0。

### 4.3 命令分发、去重与失败恢复

`PrefetchOnLookup` 把下层确认命中的连续前缀按 round-robin 分发给当前 Ready 的 worker rank。每个 rank 在共享控制区中拥有一个深度为 4096 的有界命令环；`PrefetchQueue` 后台线程每批最多取 64 个 block，队列为空时休眠 1 ms。

后台线程执行 `Buffer::Get(block, 0, false)`。`allowReserved = false` 会保留正式访问专用 slot；只有取得 owner 的 Handle 才读取下层存储，已经 Ready 或正在由其他任务填充的 block 不会重复读取。批量 `Load` 和 `Wait` 成功后发布 Ready，失败则发布 Failed；失败 Handle 释放后，后续请求仍可重新取得 owner 并重试。

预取是 best-effort 性能提示：没有在线 worker、命令环已满或下层读取失败都不能改变 Lookup 结果，也不能影响后续按需访问的正确性。环满造成的丢弃会累计到 dropped 计数。

### 4.4 预取时序图

```mermaid
sequenceDiagram
    autonumber
    participant S as vLLM Scheduler
    participant U as UCMConnector
    participant I as Direct / LayerWise Connector
    participant R as RankConsistencyManager
    participant M as CacheStore / BufferManager
    participant B as 下层存储
    participant Q as 共享预取命令环
    participant P as Worker PrefetchQueue
    participant H as Buffer / Host Cache

    S->>U: get_num_new_matched_tokens(request)
    U->>I: 转发给内部 Connector
    Note over I: LayerWise 继承 Direct 的 lookup 实现
    I->>R: lookup_on_prefix(block_ids)
    R->>M: store.lookup_on_prefix(block_ids)
    M->>M: 检查 Host Cache
    M->>B: LookupOnPrefix(Host 未命中 blocks)
    B-->>M: 返回下层命中的连续前缀
    loop 按 Ready worker rank 轮转
        M->>Q: EnqueuePrefetch(rank, block)
    end
    M-->>R: 返回命中 block 数
    R-->>I: 返回连续前缀长度
    I-->>U: 返回匹配 token 数
    U-->>S: 返回匹配 token 数

    P->>Q: DrainPrefetch(最多 64 个)
    P->>H: Get(block, shard 0, allowReserved=false)
    alt 取得 owner
        P->>B: Load(batch, Host 地址)
        P->>B: Wait(task)
        B-->>P: 成功或失败
        P->>H: MarkReady() / MarkFailed()
    else 已 Ready 或已有填充者
        P->>P: 跳过重复读取
    end
    Note over P,H: Direct: shard 0 是整块 KV<br/>LayerWise: shard 0 通常是模型第 0 层
```

## 5. 预取类图

下图只展示从 vLLM scheduler lookup 到后台预取的相关类。LayerWise Connector 继承 Direct Connector 的 lookup 实现，所以两种模式的触发入口相同；布局差异只改变 `shard 0` 所代表的数据。

```mermaid
classDiagram
    class Scheduler {
        +schedule()
    }
    class UCMConnector {
        -connector
        +get_num_new_matched_tokens(request)
    }
    class UCMDirectConnector {
        -store
        -rank_consistency
        +get_num_new_matched_tokens(request)
    }
    class UCMLayerWiseConnector {
        <<LayerWise>>
    }
    class RankConsistencyManager {
        +lookup_on_prefix(store, block_ids)
    }
    class UcmKVStoreBaseV1 {
        +lookup_on_prefix(block_ids)
    }
    class CacheStore {
        +LookupOnPrefix(blocks)
    }
    class BufferManager {
        +LookupOnPrefixFast(blocks)
        -PrefetchOnLookup(blocks)
    }
    class Buffer {
        +Exist(block, shard)
        +EnqueuePrefetch(rank, blocks)
        +DrainPrefetch(rank)
        +Get(block, shard, allowReserved)
    }
    class PrefetchQueue {
        -PrefetchLoop()
        -PrefetchBatch(blocks)
    }
    class CtrlLayout {
        +RingPush(rank, blocks)
        +RingDrain(rank, blocks)
    }
    class PrefetchRing {
        +head
        +tail
        +dropped
        +entries[4096]
    }
    class Handle {
        +Owner()
        +Data()
        +MarkReady()
        +MarkFailed()
    }
    class StoreV1 {
        <<下层存储>>
        +LookupOnPrefix(blocks)
        +Load(task)
        +Wait(handle)
    }

    Scheduler --> UCMConnector : 查询外部命中
    UCMConnector --> UCMDirectConnector : use_layerwise=false
    UCMConnector --> UCMLayerWiseConnector : use_layerwise=true
    UCMLayerWiseConnector --|> UCMDirectConnector
    UCMDirectConnector --> RankConsistencyManager
    RankConsistencyManager --> UcmKVStoreBaseV1 : lookup_on_prefix
    UcmKVStoreBaseV1 --> CacheStore : Python/C++ binding
    CacheStore *-- BufferManager
    CacheStore *-- PrefetchQueue
    BufferManager *-- Buffer
    BufferManager --> StoreV1 : 查询下层前缀
    PrefetchQueue --> Buffer
    PrefetchQueue --> StoreV1 : 预取 shard 0
    PrefetchQueue ..> Handle : 持有 owner 并发布状态
    Buffer ..> Handle : Get 返回
    Buffer *-- CtrlLayout
    CtrlLayout *-- "1..N" PrefetchRing
```

## 6. 测试方法

### 6.1 历史自测数据

历史自测使用 GLM-5.1、64K 输入、并发 32 和 100% 命中，结果如下。

| 版本 | HBM PC TTFT | Cache TTFT | Cache 单层 Load | Posix TTFT | Posix 单层 Load |
|---|---:|---:|---:|---:|---:|
| 优化前 | 600 ms | 975 ms | Avg 7.46 ms，P99 18.6 ms | 1380 ms | Avg 13.6 ms，P99 40 ms |
| 优化后 | 570 ms | 740 ms | Avg 3.1 ms，P99 4.35 ms | 1350 ms | Avg 13.3 ms，P99 40 ms |

### 6.2 NUMA 验收

使用至少包含两个 NUMA 内存节点的服务器，固定模型、输入、总 Cache 容量、rank 数、并发和 stream 数。初始化后检查：

- 每个数据段的 `SHM NUMA verify` 记录均为 `mismatches=0`；
- 各数据段已用 slot 大致均衡，多个内存控制器同时产生有效读带宽；
- 与流量集中在单节点的基线相比，`ucm:cache_load_duration_ms` 均值和 P99 下降；
- Posix 命中的 TTFT 与单层 Load 不出现稳定劣化。

建议使用 A3、MLA 模型、长序列高命中（100%）负载，同时比较 HBM PC、Cache 和 Posix 三条命中路径。预期 Cache 命中的 `ucm:cache_load_duration_ms` 相对单块共享内存版本降低 20% 以上，并体现为 TTFT 下降。

### 6.3 首层预取验收

测试数据应满足“下层存储已有 block、Host Cache 尚未缓存”。在模型、block 数、数据大小以及 Lookup 到正式访问的调度间隔一致时，对比不触发预取和先执行 `LookupOnPrefix` 两组流程：

- Lookup 结果与未启用预取时一致；
- Direct 和 LayerWise Connector 都调用 `lookup_on_prefix` 并产生预取命令；
- Direct 模式的 `shard 0` 包含整块 KV，LayerWise 模式的 `shard 0` 只包含模型第 0 层，正式请求前均不发生 H2D；
- 非流水线并行的 LayerWise 模式中，预取完成后第 0 层不再创建重复的下层读取，首层 backend wait 接近 0；
- pipeline parallel 的非首 stage 应单独验证并记录当前 `shard 0` 与 `first_layer_id` 不一致的行为；在预取命令携带 shard id 前，不应把它计为本地首层预取命中；
- 预取尚未完成时，同一 slot 不发生重复的下层读取；命令环溢出或预取失败后，后续访问仍能按需成功；
- 同时覆盖多个生产者写入预取环、FIFO 顺序、溢出 dropped 计数和队列复用。

绝对耗时受存储介质和数据大小影响，不建议设置统一的毫秒阈值；应在相同环境下比较第一层 backend wait、`ucm:cache_load_duration_ms` 和 TTFT 的分布。
