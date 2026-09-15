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

### 4.1 触发范围与命令分发

这里的“第一层”是一个 block 的首个 shard，即 `shardIndex = 0`。预取只完成“下层存储到 Host Cache”，不提前执行 H2D，也不改变 Lookup 的返回结果。

`BufferManager::LookupOnPrefixFast` 先查询 Host Cache，再把未命中的 block 交给下层存储执行前缀查询。对下层确认命中的连续前缀，`PrefetchOnLookup` 按稳定的 round-robin 方式分发给当前 Ready 的 worker rank；每次调用都从列表第一个 block 开始分配，因此相同列表位置会稳定落到相同 rank。

每个 rank 在共享控制区中拥有一个深度为 4096 的有界预取命令环。`PrefetchQueue` 的后台线程每次最多取 64 个 block，队列为空时休眠 1 ms。没有在线 worker 或命令环已满时允许跳过命令；环满造成的丢弃会累计到 dropped 计数。预取是性能提示，这些情况不能影响正式 Lookup 和 Load 的正确性。

### 4.2 去重、状态复用与失败恢复

后台线程对每个命令执行 `Buffer::Get(block, 0, false)`：

- `allowReserved = false` 保留 Load 专用 slot，不让推测性预取耗尽正式加载资源；
- 只有取得 owner 的 Handle 才负责下层读取；如果 slot 已 Ready，或另一个预取/正式 Load 正在填充，同一 block 不会重复读取；
- 一个批次只提交一次下层 `Load` 和 `Wait`。成功时所有 owner Handle 发布 Ready，失败时整批发布 Failed；
- Handle 在存活期间增加引用计数，防止 slot 被 CLOCK 淘汰。失败 Handle 释放后，下一次取得 owner 的预取或正式 Load 会把状态恢复为 Loading 并重新读取。

正式 `LoadQueue` 获取同一 `(block, shard 0)` 时有三种情况：已经 Ready 则直接从 Host Cache 提交 H2D；仍在 Loading 则等待同一 slot 的结果；处于可恢复的 Failed 状态并取得 owner 时，由正式 Load 重新从下层存储读取。因而预取完成得越早，第一层等待下层存储的时间越短；预取未完成或失败只会退化为原有按需加载路径。

### 4.3 预取时序图

```mermaid
sequenceDiagram
    autonumber
    participant E as 推理引擎
    participant M as BufferManager
    participant C as Buffer / CtrlLayout
    participant B as 下层存储
    participant P as Rank PrefetchQueue
    participant H as Rank Host 数据段
    participant L as LoadQueue
    participant D as NPU/GPU

    E->>M: LookupOnPrefix(blocks)
    M->>C: Exist(block, shard 0)
    C-->>M: Host Cache 命中与未命中集合
    M->>B: LookupOnPrefix(未命中 blocks)
    B-->>M: 下层存储命中的连续前缀
    loop 对前缀 block 按 Ready rank 轮转
        M->>C: EnqueuePrefetch(rank, block)
    end
    M-->>E: 返回 Lookup 结果

    par 后台预取
        P->>C: DrainPrefetch(本 rank，最多 64 个)
        P->>C: Get(block, shard 0, allowReserved=false)
        C-->>P: Handle 与 owner 状态
        alt 取得 owner
            P->>B: Load(batch, Handle.Data)
            B->>H: 写入 Host Cache
            P->>B: Wait(task)
            B-->>P: 成功或失败
            alt 成功
                P->>C: MarkReady()
            else 失败
                P->>C: MarkFailed()
            end
        else 已 Ready 或已有填充者
            P->>P: 跳过重复读取
        end
        P->>C: 释放 Handle 引用
    and 正式第一层 Load
        E->>L: Load(block, shard 0)
        L->>C: Get(block, shard 0)
        C-->>L: Handle 与 slot 状态
        alt 已 Ready
            L->>H: 取得 Host 地址
        else 预取仍在 Loading
            L->>C: 等待同一 slot Ready / Failed
        else 正式 Load 取得 owner
            L->>B: 按需 Load 并 Wait
            B->>H: 写入 Host Cache
            L->>C: 发布 Ready / Failed
        end
        opt slot Ready
            L->>D: H2D
            D-->>E: 第一层 KV 就绪
        end
        L->>C: 释放 Handle 引用
    end
```

## 5. 类图

下图按职责简化接口。`RankDataSegment` 是概念上的共享数据段；每个进程通过 `DataStrategy` 保存自己的本地映射。NUMA 拓扑探测、控制区命令分发、预取和正式 Load 最终都汇聚到同一个 Buffer/Handle 状态机。

```mermaid
classDiagram
    class UCMDirectConnector {
        -_configure_partitioned_store(config)
        -_configure_numa_placement(config)
    }
    class Device {
        +get_numa_node(deviceOrdinal)
    }
    class CacheStore {
        +LookupOnPrefix(blocks)
        +Load(task)
    }
    class BufferManager {
        +LookupOnPrefix(blocks)
        -PrefetchOnLookup(blocks)
    }
    class PrefetchQueue {
        -PrefetchLoop()
        -PrefetchBatch(blocks)
    }
    class TransManager
    class LoadQueue {
        +Submit(task)
    }
    class DumpQueue {
        +Submit(task)
    }
    class Buffer {
        +Exist(block, shard)
        +Get(block, shard, allowReserved, preferredSegment)
        +EnqueuePrefetch(rank, blocks)
        +DrainPrefetch(rank)
    }
    class Handle {
        +Owner()
        +GlobalSlot()
        +Segment()
        +Data()
        +DeviceData()
        +MarkReady()
        +MarkFailed()
    }
    class CtrlStrategy {
        +Setup(config)
        +Layout()
    }
    class CtrlLayout {
        +SlotMetaArr()
        +RingPush(rank, blocks)
        +RingDrain(rank, blocks)
    }
    class PrefetchRing {
        +head
        +tail
        +dropped
        +entries[4096]
    }
    class DataStrategy {
        +Setup(...)
        +MapAllSegments()
        +DataAt(globalSlot)
        +DeviceDataAt(globalSlot)
    }
    class ShmNuma {
        <<namespace>>
        +DataNodes(...)
        +SegmentNodes(...)
        +Initialize(data, bytes, nodes)
    }
    class RankDataSegment {
        <<共享 KV 数据段>>
        +creatorRank
        +numaNodes
        +slotRange
    }
    class StoreV1 {
        <<下层存储>>
        +LookupOnPrefix(blocks)
        +Load(task)
        +Wait(handle)
    }

    UCMDirectConnector --> Device : 探测 NPU NUMA
    UCMDirectConnector --> CacheStore : 传入分段与节点提示
    CacheStore *-- BufferManager
    CacheStore *-- PrefetchQueue
    CacheStore *-- TransManager
    TransManager *-- LoadQueue
    TransManager *-- DumpQueue
    BufferManager *-- Buffer
    BufferManager --> StoreV1
    PrefetchQueue --> Buffer
    PrefetchQueue --> StoreV1
    LoadQueue --> Buffer
    LoadQueue --> StoreV1
    DumpQueue --> Buffer
    Buffer *-- CtrlStrategy
    CtrlStrategy *-- CtrlLayout
    CtrlLayout *-- "1..N" PrefetchRing
    Buffer *-- DataStrategy
    DataStrategy ..> ShmNuma : 绑定并验证页面
    DataStrategy --> "1..N" RankDataSegment : 创建本段并映射各段
    Buffer ..> Handle : 返回实际位置与引用
    Handle --> Buffer : 析构时释放引用
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

测试数据应满足“下层存储已有 block、Host Cache 尚未缓存”。在模型、block 数、数据大小以及 Lookup 到 Load 的调度间隔一致时，对比不触发预取和先执行 `LookupOnPrefix` 两组流程：

- Lookup 结果与未启用预取时一致；
- 预取只加载 shard 0，正式请求前不发生 H2D；
- 预取完成时，正式 Load 第一层不再创建重复的下层读取，首层 backend wait 接近 0；
- 预取尚未完成时，正式 Load 等待同一 slot，不发生重复读取；
- 命令环溢出或预取失败后，正式 Load 仍能按需加载成功；
- 同时覆盖多个生产者写入预取环、FIFO 顺序、溢出 dropped 计数和队列复用。

绝对耗时受存储介质和数据大小影响，不建议设置统一的毫秒阈值；应在相同环境下比较第一层 backend wait、`ucm:cache_load_duration_ms` 和 TTFT 的分布。
