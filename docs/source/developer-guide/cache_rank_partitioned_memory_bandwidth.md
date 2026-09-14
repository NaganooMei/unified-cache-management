# Cache 内存按 Rank 划分，突破单 DDR 控制器带宽瓶颈

## 1. 背景

多 rank 并发从 Host Cache 加载 KV 时，若物理页集中在同一 NUMA 节点，读流量将集中到该节点的内存控制器。内存带宽饱和后，继续增加 rank 或拷贝 stream 难以提升总吞吐，单层 Cache Load 时延随竞争加剧而上升。

MLA 模型中，多个 TP rank 复用同一份 KV，容易形成集中读取。当加载耗时无法被层间计算掩盖时，推理需等待 KV 到位，增加首 token 时延（TTFT）。将 Cache 数据按 rank 分段，并结合 NUMA 绑定和均衡放置，可利用多个内存控制器并行供数，降低单层加载时延。

## 2. 实现

### 2.1 内存布局

各 rank 共享控制区中的 block 索引、slot 状态和数据段状态。KV 数据按本机实际 rank 数划分，由各 rank 分别创建、初始化和注册。各进程通过统一索引访问数据，对外保持一个逻辑 Cache。

以下为数据段分布在不同 NUMA 节点的示意，实际绑定关系由硬件拓扑确定：

~~~text
逻辑 Cache
  ├─ 共享控制区
  ├─ Rank 0 数据段 -> NUMA/DDR 0
  ├─ Rank 1 数据段 -> NUMA/DDR 1
  ├─ Rank 2 数据段 -> NUMA/DDR 2
  └─ Rank N 数据段 -> NUMA/DDR N
~~~

Cache 容量按共享域总量配置，按实际 rank 数划分完整 slot，对齐余量不计入可用容量。

全局 slot 编号可稳定解析为数据段和段内位置：

~~~text
segment = globalSlot / slotsPerRank
localSlot = globalSlot % slotsPerRank
~~~

控制区以 `(blockId, shardIndex)` 为键保存全局 slot 编号。各 worker 在启动阶段完成数据段映射和设备注册，访问时将 slot 编号解析为进程本地地址。

### 2.2 NUMA 绑定

各 rank 在目标 NUMA 节点的 CPU 亲和性下初始化数据段，利用 first-touch 机制分配物理页。部署时需核对 rank、CPU、内存与设备的亲和关系，确保数据页分布覆盖多个内存控制器。

### 2.3 KV 放置策略

- MLA：由 worker 0 执行 Dump 时，按任务中分片的原始位置 `originalIndex % segmentCount` 选择优先数据段。同一层不同 block 的 KV 分散存放，避免全部写入 rank 0。
- GQA：KV 与 rank 相关，默认优先放在产生该 KV 的本地 rank 数据段。
- 已缓存的 `(blockId, shardIndex)` 使用当前实际位置，不因后续请求来自不同 rank 而迁移。
- 目标数据段暂时没有空闲 slot 时，可以按规则回退到其他数据段，但必须记录实际位置。

数据段的生命周期由所属 rank 管理，数据放置由分配策略决定，支持跨 rank 写入与读取。

### 2.4 并行加载

各 rank 根据自身编号错开任务处理顺序，并从 Handle 返回的实际数据段读取。重排只改变加载顺序，不改变数据位置。MLA 的各 rank 最终均加载所需的完整 KV。

例如，同一层 block A、B 分别位于数据段 0、1：rank 0 先读取 A，rank 1 先读取 B，再交换读取其余 block。结合异步传输与多 stream，可使多个内存控制器同时供数。这里的分片序号是任务内位置，不是层号。

### 2.5 类图

类图表示需求设计中的职责关系，接口按职责简化。PlacementPolicy 表示放置策略，RankDataSegment 表示共享数据段；各进程分别维护数据段的本地映射。

~~~mermaid
classDiagram
    class Buffer {
        <<索引与 slot 分配>>
        +Get(key, preferredSegment)
    }
    class CtrlStrategy {
        <<共享控制区生命周期>>
        +CreateOrJoin()
    }
    class CtrlLayout {
        <<全局索引与状态>>
        +BlockShardIndex
        +SlotMeta
    }
    class PlacementPolicy {
        <<新 slot 放置策略>>
        +PreferredSegment(originalIndex, rank)
    }
    class DataStrategy {
        <<NUMA 初始化与地址解析>>
        +SetupSegments()
        +DataAt(globalSlot)
        +DeviceDataAt(globalSlot)
    }
    class RankDataSegment {
        <<共享 KV 数据段>>
        +creatorRank
        +numaNode
        +slotRange
    }
    class Handle {
        <<实际位置与引用>>
        +GlobalSlot()
        +Segment()
        +Data()
        +DeviceData()
    }
    class LoadQueue {
        <<错序读取与 H2D>>
        +Submit(task)
    }
    class DumpQueue {
        <<分段写入与 D2H>>
        +Submit(task)
    }

    LoadQueue ..> PlacementPolicy
    DumpQueue ..> PlacementPolicy
    LoadQueue --> Buffer
    DumpQueue --> Buffer
    Buffer *-- CtrlStrategy : 持有
    CtrlStrategy *-- CtrlLayout : 持有布局视图
    Buffer *-- DataStrategy : 持有本地映射
    DataStrategy --> "1..N" RankDataSegment : 创建本段并映射各段
    Buffer ..> Handle : 返回实际位置
    Handle --> Buffer : 析构时释放引用
~~~

### 2.6 时序图

以同一层的两个 block、两个 rank 为例。各数据段已完成 NUMA 初始化、映射和注册。

~~~mermaid
sequenceDiagram
    autonumber
    participant W0 as Worker 0 DumpQueue
    participant C as Buffer（共享索引）
    participant S0 as Rank 0 数据段
    participant S1 as Rank 1 数据段
    participant R0 as Rank 0 LoadQueue
    participant R1 as Rank 1 LoadQueue

    Note over W0,S1: 同一层 A、B 首次写入，分别优先选择段 0、1
    W0->>C: Get(A, layer L, preferred 0)
    C-->>W0: owner Handle，实际段 0
    W0->>S0: 从设备 D2H 写入 A 的第 L 层 KV
    W0->>C: Get(B, layer L, preferred 1)
    C-->>W0: owner Handle，实际段 1
    W0->>S1: 从设备 D2H 写入 B 的第 L 层 KV
    W0->>W0: 等待 D2H 完成
    W0->>C: 发布 A、B 的 slot 为 Ready

    Note over C,R1: Load 同一层，两个 rank 均需要 A 和 B
    R0->>C: Get(A, L) 与 Get(B, L)
    C-->>R0: 返回实际段 0、1 的 Handle
    R1->>C: Get(B, L) 与 Get(A, L)
    C-->>R1: 返回实际段 1、0 的 Handle

    par Rank 0 先读取 A
        R0->>S0: 提交 A 到 device 0 的 H2D
    and Rank 1 先读取 B
        R1->>S1: 提交 B 到 device 1 的 H2D
    end
    par Rank 0 继续读取 B
        R0->>S1: 提交 B 到 device 0 的 H2D
    and Rank 1 继续读取 A
        R1->>S0: 提交 A 到 device 1 的 H2D
    end
    Note over R0,R1: 各自等待传输完成后释放 Handle；两个设备均持有本层完整 KV
~~~

## 3. 测试方法

使用多 NUMA 节点服务器部署 MLA 模型，构造较长缓存前缀、较少未命中 token 的请求，使 KV 加载量较大、剩余计算较少，Cache 加载耗时无法被计算掩盖。测试数据预置于 Host Cache，确保请求实际走 Host Cache 到设备的加载路径。

固定模型、输入、总 Cache 容量、rank 数、并发及 stream 数，完成预热后，对比内存分段前后的两个指标：

- TTFT：观察首 token 时延变化。
- `ucm:cache_load_duration_ms`：在按层提交 Load 的条件下，观察单层 Cache Load 耗时变化。

重复执行相同负载，比较两项指标的均值及 P99。预期分段后单层 Cache Load 耗时下降，并在上述场景下体现为 TTFT 降低。
