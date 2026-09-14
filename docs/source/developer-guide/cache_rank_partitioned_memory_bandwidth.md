# Cache 内存按 Rank 划分，突破单 DDR 控制器带宽瓶颈

本文面向需求串讲和测试，说明为什么 Cache 数据内存需要按 rank 划分、如何利用多路 DDR/NUMA 带宽，以及如何验证收益。

## 1. 需求结论

- 所有 rank 共享一份 Cache 控制区，保证 block 状态一致。
- Cache 数据内存按本机实际 rank 数切成多个数据段，每个 rank 负责一个数据段的创建、初始化和注册。
- KV shard 按放置规则分散到不同 rank 数据段，多个 rank Load 时可以并行访问不同的 DDR/NUMA 节点。
- 对外仍是一个逻辑 Cache；调用方不需要感知数据具体位于哪个 rank。

按 rank 分段只是基础。要获得带宽收益，还必须同时保证数据段落在不同 NUMA 节点、KV 分布均衡，并且传输任务能够并行执行。

## 2. Why：为什么要做

多 rank 并发从 Host Cache 向设备加载 KV 时，如果数据集中在同一块共享内存及其对应的 DDR/NUMA 节点，所有 DMA 都会争用同一个内存控制器。

典型现象是：

- 增加 rank 或拷贝 stream 后，总带宽增长不明显；
- Cache 单层 Load 时延在高并发下明显升高；
- 第一层或无法被计算掩盖的层进入 TTFT 关键路径；
- 设备侧仍有余量，但主机侧内存带宽已经饱和。

因此，本需求不是单纯把一个大内存文件拆成多个名字，而是要让不同数据段实际使用不同的内存控制器，并让访问流量均匀落到这些数据段上。

## 3. How：我们怎么做

### 3.1 一份控制区，多份数据区

控制区保存全局 block 索引、slot 状态和每个 rank 的数据段状态。数据区按 rank 划分：

~~~text
逻辑 Cache
  ├─ 共享控制区
  ├─ Rank 0 数据段 -> NUMA/DDR 0
  ├─ Rank 1 数据段 -> NUMA/DDR 1
  ├─ Rank 2 数据段 -> NUMA/DDR 2
  └─ Rank N 数据段 -> NUMA/DDR N
~~~

配置的 Cache 总容量按实际 rank 数切分，各数据段容量之和等于逻辑 Cache 总容量。

全局 slot 编号可稳定解析为数据段和段内位置：

~~~text
segment = globalSlot / slotsPerRank
localSlot = globalSlot % slotsPerRank
~~~

控制区只保存稳定的 slot 编号，不保存某个进程的虚拟地址。每个进程根据 slot 编号解析本地地址；首次访问其他 rank 的数据段时，完成映射和设备注册。

### 3.2 数据真正落到多路 DDR

每个 rank 创建自己的数据段，并在与该 rank 对应的 NUMA 亲和性下完成页面初始化。利用 first-touch 机制，让物理页落到目标 NUMA 节点。

运行环境需要保证 rank、CPU 和 NUMA 的绑定关系正确。若多个 rank 最终仍绑定到同一 NUMA 节点，即使逻辑上有多个数据段，也无法突破单控制器瓶颈。

### 3.3 KV 放置策略

- MLA：多个 rank 复用同一份 KV。即使只有 worker 0 执行 Dump，也按 shard 序号把数据均匀放到不同 rank 数据段，例如 preferred rank = shard index % rank count。
- GQA：KV 与 rank 相关，默认优先放在产生该 KV 的本地 rank 数据段。
- 已存在的 block 始终使用首次分配的实际位置，不因后续请求来自不同 rank 而迁移。
- 目标数据段暂时没有空闲 slot 时，可以按规则回退到其他数据段，但必须记录实际位置。

数据段的创建者和 KV 的放置位置是两个概念。数据段由对应 rank 管理，不代表所有数据只能由该 rank 的请求产生。

### 3.4 类图

~~~mermaid
classDiagram
    class Buffer {
        +Get(block, shard, preferredRank)
        +Resolve(globalSlot)
    }

    class CtrlStrategy {
        +CreateOrJoin()
    }

    class CtrlLayout {
        +BlockIndex
        +SlotMeta
        +RankDataDesc
    }

    class PlacementPolicy {
        +PreferredRank(modelType, shard, callerRank)
        +FallbackRank()
    }

    class DataStrategy {
        +CreateLocalSegment(rank)
        +MapRemoteSegment(rank)
        +DataAt(globalSlot)
        +DeviceDataAt(globalSlot)
    }

    class RankDataSegment {
        +rank
        +numaNode
        +slots
        +hostAddress
        +deviceAddress
    }

    class Handle {
        +GlobalSlot()
        +ActualRank()
        +Data()
        +DeviceData()
    }

    class LoadQueue {
        +Submit()
        +H2D()
    }

    class DumpQueue {
        +Submit()
        +D2H()
    }

    Buffer *-- CtrlStrategy
    CtrlStrategy *-- CtrlLayout
    Buffer --> PlacementPolicy
    Buffer *-- DataStrategy
    DataStrategy *-- "1..N" RankDataSegment
    Buffer --> Handle
    LoadQueue --> Buffer
    DumpQueue --> Buffer
    LoadQueue --> Handle
    DumpQueue --> Handle
~~~

### 3.5 时序图：MLA 分散写入与并行读取

~~~mermaid
sequenceDiagram
    autonumber
    participant W0 as MLA Worker 0
    participant P as PlacementPolicy
    participant C as 共享控制区
    participant S0 as Rank 0 数据段
    participant S1 as Rank 1 数据段
    participant R0 as Rank 0 Load
    participant R1 as Rank 1 Load
    participant D0 as Device 0
    participant D1 as Device 1

    Note over W0,S1: Dump 阶段：worker 0 提交，数据按 shard 分散
    W0->>P: 请求 shard 0 的目标段
    P-->>W0: preferred rank 0
    W0->>C: 为 shard 0 分配 slot
    C-->>W0: Rank 0 实际 slot
    W0->>S0: D2H 写入 shard 0

    W0->>P: 请求 shard 1 的目标段
    P-->>W0: preferred rank 1
    W0->>C: 为 shard 1 分配 slot
    C-->>W0: Rank 1 实际 slot
    W0->>S1: D2H 写入 shard 1
    W0->>C: 标记两个 slot Ready

    Note over R0,D1: Load 阶段：不同 DDR/NUMA 节点并行供数
    par Rank 0 读取
        R0->>C: 查询 shard 0
        C-->>R0: Rank 0 slot
        R0->>S0: 获取数据地址
        S0->>D0: H2D
    and Rank 1 读取
        R1->>C: 查询 shard 1
        C-->>R1: Rank 1 slot
        R1->>S1: 获取数据地址
        S1->>D1: H2D
    end
~~~

## 4. 怎么测试

### 4.1 测试目标

测试需要同时证明两件事：

1. 数据确实分散到了不同 rank 对应的 NUMA/DDR 节点。
2. 在 Cache Load 无法被模型计算掩盖时，单层 Load 时延和 TTFT 得到改善。

只看到多个数据段文件或地址不同，不能证明已经使用多路 DDR 带宽。

### 4.2 环境和数据准备

- 使用至少包含两个 NUMA/DDR 节点的服务器，并确认各 rank 的 CPU、内存和设备亲和关系。
- 使用 MLA 模型，使多个 TP rank 共享同一份 KV。
- 提前把测试前缀完整写入 Host Cache，测试阶段避免落到下层存储。
- 使用较长前缀和较大的 KV 数据量，输出 token 数设置为 1，减少 Decode 对 TTFT 的干扰。
- 提高并发或批量，使多个 rank 同时执行 Cache 到设备的加载，形成可观测的内存带宽压力。
- 选择“单层 Cache Load 时间大于或接近该层计算时间”的参数，使加载无法被计算完全掩盖。

### 4.3 对比方法

在总 Cache 容量、模型、输入、并发、stream 数和绑核方式一致的前提下，对比：

- 单数据段或流量集中在一个 NUMA 节点的基线；
- 数据按 rank 分段、均匀落到多个 NUMA 节点的方案。

每组先预热，再执行多轮稳定测试，至少比较 P50、P90 和 P99，避免用单次结果下结论。

### 4.4 观察项

主要业务指标：

- TTFT：确认关键路径是否缩短；
- ucm:layerwise_layer_load_duration_ms：观察 Cache Load 一层的墙钟时延；
- ucm:connector_wait_for_layer_load_duration_ms：观察模型等待层数据的时间。

Cache 侧辅助指标：

- ucm:cache_load_duration_ms；
- ucm:cache_load_bandwidth_gbps；
- ucm:cache_h2d_sync_ms；
- ucm:cache_load_wait_shards_total；
- ucm:cache_load_backend_shards_total。

同时使用平台内存带宽工具观察各 NUMA/DDR 控制器流量，并检查：

- 各 rank 数据段的已用 slot 是否大致均衡；
- 各数据段的物理页是否位于预期 NUMA 节点；
- 多个内存控制器是否同时产生有效读带宽；
- 测试期间下层存储读取计数是否接近 0。

### 4.5 通过标准

- Cache 总容量没有因 rank 数增加而意外放大或缩小。
- MLA 数据没有集中在 worker 0 的数据段，各 rank 数据段占用基本均衡。
- 多个 NUMA/DDR 控制器同时工作，总内存读带宽高于单节点基线。
- 在无法掩盖 Cache Load 的场景下，单层 Load 时延下降，TTFT 的 P50、P90 和 P99 同方向改善。
- ucm:cache_load_wait_shards_total 和 ucm:cache_load_backend_shards_total 未异常增加，确认收益不是由测试数据来源变化造成。

如果单层 Load 指标下降但 TTFT 没有变化，说明加载仍被计算或其他阶段掩盖，需要进一步缩短计算、增加 KV 数据量或提高并发；如果数据段均衡但只有一个内存控制器有流量，应优先检查 NUMA 绑定和 first-touch 是否生效。
