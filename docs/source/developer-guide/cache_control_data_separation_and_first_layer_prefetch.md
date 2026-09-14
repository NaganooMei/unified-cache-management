# Cache 控制区与数据区分离及 Lookup 首层预取

本文面向需求串讲和测试，重点说明两个问题：为什么要拆分 Cache 控制区与数据区，以及为什么在 Lookup 后预取第一层。

## 1. 需求结论

- Cache 控制信息放在独立的共享控制区，KV 数据放在按 rank 管理的数据区。
- Lookup 确认下层存储存在可复用前缀后，后台提前把这些 block 的第一层读入 Host Cache。
- 后续正式 Load 第一层时，如果预取已经完成，可直接从 Host Cache 搬到设备，减少第一层等待下层存储的时间。
- 预取是性能优化，不改变 Lookup 结果；预取未完成或失败时，正式 Load 仍按原流程读取数据。

这里的“第一层”指一个 block 的第一个 shard，即 shard index 为 0。预取只完成“下层存储到 Host Cache”，不会提前执行 Host Cache 到设备的拷贝。

## 2. Why：为什么要做

### 2.1 控制信息和 KV 数据的使用方式不同

控制信息包括 block 是否存在、slot 状态、引用计数和淘汰信息。它体积小，但 Lookup 和并发协调会频繁访问。

KV 数据体积大，只在 Load、Dump 和预取时参与数据搬运。两者分离后：

- Lookup 主要访问轻量的控制区，不需要关心大块数据内存的创建和地址管理。
- 控制区可以被所有进程共享，保证大家看到一致的 block 状态。
- 每个 rank 的数据区可以独立创建、注册和释放，便于后续做 NUMA 和带宽优化。

### 2.2 第一层的下层存储等待会直接影响 TTFT

Layerwise Load 可以让后续层的数据搬运与前面层的计算重叠，但第一层开始执行前没有上一层计算可用于掩盖等待。

当数据只存在于下层存储时，原流程是：

~~~text
Lookup 确认命中 -> 正式 Load 第一层 -> 等待下层存储 -> H2D -> 第一层计算
~~~

第一层的存储等待直接落在 TTFT 关键路径上。Lookup 已经知道哪些 block 在下层存储命中，因此可以利用 Lookup 到正式 Load 之间的时间，提前把第一层放进 Host Cache。

## 3. How：我们怎么做

### 3.1 控制区与数据区

控制区保存：

- 全局 Header 和布局信息；
- block 到 slot 的索引；
- slot 状态、引用计数和淘汰信息；
- 每个 rank 的在线状态；
- 每个 rank 的预取命令队列。

数据区只保存 KV payload。每个 slot 通过全局 slot 编号关联控制信息和实际数据地址，业务侧通过 Handle 使用 slot，不直接管理共享内存地址。

### 3.2 类图

~~~mermaid
classDiagram
    class CacheStore {
        +LookupOnPrefix(blocks)
        +Load(task)
    }

    class BufferManager {
        +LookupOnPrefix(blocks)
        -PrefetchOnLookup(blocks)
    }

    class Buffer {
        +Exist(block, shard)
        +Get(block, shard)
        +EnqueuePrefetch(rank, blocks)
        +DrainPrefetch(rank)
    }

    class CtrlStrategy {
        +CreateOrJoin()
        +Layout()
    }

    class CtrlLayout {
        +Header
        +HashBuckets
        +SlotMeta
        +RankStatus
        +PrefetchRing
    }

    class DataStrategy {
        +DataAt(slot)
        +DeviceDataAt(slot)
        +MapRankData(rank)
    }

    class Handle {
        +Owner()
        +Ready()
        +Data()
        +MarkReady()
        +MarkFailed()
    }

    class PrefetchQueue {
        +DrainCommands()
        +LoadFirstShard()
    }

    class BackendStore {
        +LookupOnPrefix(blocks)
        +Load(shards)
        +Wait(task)
    }

    CacheStore --> BufferManager
    CacheStore --> PrefetchQueue
    BufferManager --> Buffer
    Buffer *-- CtrlStrategy
    CtrlStrategy *-- CtrlLayout
    Buffer *-- DataStrategy
    Buffer --> Handle
    PrefetchQueue --> Buffer
    PrefetchQueue --> BackendStore
    BufferManager --> BackendStore
~~~

### 3.3 Lookup 后首层预取

处理流程如下：

1. Lookup 先查询 Host Cache。
2. Host Cache 未命中的 block 再查询下层存储。
3. 对下层存储确认命中的前缀 block，生成预取命令。
4. 命令分发到在线 rank 的预取队列。
5. rank 后台线程为 block 获取 Cache slot，只加载 shard 0。
6. 下层存储读取完成后，将 slot 标记为 Ready。
7. 正式 Load 到来时，Ready 的第一层直接执行 H2D；若仍在预取，则只等待剩余时间。

多个请求命中同一 block 时，只有取得 owner 的任务执行实际预取，其他任务复用同一 slot，避免重复读取。

### 3.4 时序图

~~~mermaid
sequenceDiagram
    autonumber
    participant E as 推理引擎
    participant M as BufferManager
    participant C as 共享控制区
    participant B as 下层存储
    participant P as Rank PrefetchQueue
    participant D as Rank 数据区
    participant L as LoadQueue
    participant N as NPU/GPU

    E->>M: LookupOnPrefix(blocks)
    M->>C: 查询 Host Cache
    C-->>M: 返回命中与未命中 block
    M->>B: 查询未命中 block
    B-->>M: 返回下层存储命中的前缀
    M->>C: 写入对应 rank 的预取命令
    M-->>E: 返回 Lookup 结果

    par 后台预取第一层
        P->>C: 取出预取命令
        P->>C: Get(block, shard 0)
        C-->>P: 返回 owner Handle
        P->>B: Load shard 0 到 Host Cache
        B->>D: 写入 KV 数据
        B-->>P: Wait 完成
        P->>C: 将 slot 标记为 Ready
    and 推理继续准备正式 Load
        E->>E: 调度第一层
    end

    E->>L: Load shard 0
    L->>C: Get(block, shard 0)
    C-->>L: 返回 Handle
    alt 预取已完成
        L->>D: 获取 Ready 数据地址
    else 预取仍在进行
        L->>C: 等待 slot Ready
        C-->>L: Ready
        L->>D: 获取数据地址
    end
    L->>N: H2D
    N-->>E: 第一层可计算
~~~

## 4. 怎么测试

### 4.1 自验证目标

验证重点不是单独看预取线程是否执行，而是确认正式请求第一层的 wait 时间确实下降。

建议在正式 Load 的第一层等待路径增加临时打点，至少记录：

- request/task 标识；
- block 标识和 shard index；
- 获取 Handle 时的状态：Ready、Loading 或新 owner；
- 是否创建了下层存储 Load task；
- Wait 开始、结束和耗时；
- 数据来源是已有 Host Cache、预取中的 Host Cache，还是正式 Load 触发的下层读取。

聚合指标 ucm:cache_shard_backend_wait_ms 包含所有 shard，不能单独说明第一层效果，因此首层专项打点是本需求的主要验证手段。

### 4.2 测试步骤

1. 准备一组“下层存储有数据、Host Cache 无数据”的 block。
2. 使用相同 block 做两组测试：
   - 基线组：不触发 Lookup 首层预取，直接执行正式 Load；
   - 预取组：先执行 LookupOnPrefix，再执行正式 Load。
3. 两组使用相同模型、block 数、数据大小和并发度，每轮前清理 Host Cache，避免历史命中干扰。
4. 对比第一层打点中的 Wait 耗时，并辅助观察：
   - ucm:cache_shard_backend_wait_ms；
   - ucm:cache_load_wait_shards_total；
   - ucm:cache_load_backend_shards_total；
   - ucm:cache_load_duration_ms。
5. 补充一个“Lookup 后立即 Load”的用例，确认预取尚未完成时正式 Load 可以等待同一 slot，而不是重复读取或返回错误。
6. 注入一次预取失败，确认 slot 会进入可恢复状态，下一次正式 Load 仍能重新加载成功。

### 4.3 通过标准

- Lookup 返回值与未开启预取时一致。
- 预取只加载 shard 0，正式请求到来前不发生 H2D。
- 预取完成后，正式 Load 第一层不再创建下层读取任务，第一层 Wait 接近 0。
- 预取只完成一部分时，正式 Load 只等待剩余时间，Wait 小于冷加载基线。
- 同一 block 不发生重复加载，预取失败也不影响正式 Load 的正确性。

绝对耗时受存储介质和数据大小影响，不建议设置固定毫秒阈值；应在相同环境下比较基线组和预取组的第一层 Wait 分布。
