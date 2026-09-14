# Cache 控制区与数据区分离及 Lookup 首层预取

## 1. 背景

Cache 的控制信息用于缓存查找、状态同步和淘汰，KV 数据用于 Load、Dump 等数据传输。二者的访问频率、内存规模和生命周期不同，需要分别管理。控制区独立后，调度进程可直接查询缓存状态并下发预取命令，数据区由各 rank 负责管理。

按层加载 KV 时，后续层的数据传输可与前序层计算重叠，第一层则缺少这一重叠窗口。若第一层仅在下层存储命中，推理需要等待存储读取及 H2D 完成，增加首 token 时延（TTFT）。利用 Lookup 到 Load 之间的调度时间预取第一层，可以减少正式加载时的存储等待。

## 2. 实现

### 2.1 控制区与数据区分离

控制区由各进程共享，保存布局信息、block 索引、slot 状态、引用计数、淘汰信息、rank 状态及预取命令队列。数据区保存 KV 数据，由各 rank 独立创建、注册和释放。

控制区通过全局 slot 编号关联数据位置，各进程将编号解析为本地地址。Load、Dump 和预取通过 Handle 访问 slot，并在使用期间持有引用，防止数据被提前淘汰。

### 2.2 类图

类图按职责简化接口。调度进程通过控制区下发命令，worker 侧的预取与 Load 共用 Buffer 和 slot 状态。

~~~mermaid
classDiagram
    class BufferManager {
        <<查询与预取触发>>
        +LookupOnPrefix(blocks)
        -PrefetchOnLookup(blocks)
    }
    class PrefetchQueue {
        <<后台预取>>
        -PrefetchLoop()
        -PrefetchBatch(blocks)
    }
    class LoadQueue {
        <<正式加载>>
        +Submit(task)
        -WaitBackendTaskReady(task)
    }
    class Buffer {
        <<缓存索引与状态管理>>
        +Exist(block, shard)
        +Get(block, shard)
        +EnqueuePrefetch(rank, blocks)
        +DrainPrefetch(rank)
    }
    class CtrlStrategy {
        <<控制区生命周期>>
        +Setup(config)
        +Layout()
    }
    class CtrlLayout {
        <<共享控制区布局>>
        +SlotMeta
        +PrefetchRing
    }
    class DataStrategy {
        <<数据区与地址管理>>
        +DataAt(slot)
        +DeviceDataAt(slot)
    }
    class Handle {
        <<使用期间持有引用>>
        +Owner()
        +GetState()
        +Data()
        +MarkReady()
        +MarkFailed()
    }
    BufferManager --> Buffer
    PrefetchQueue --> Buffer
    LoadQueue --> Buffer
    Buffer *-- CtrlStrategy : 持有
    CtrlStrategy *-- CtrlLayout : 持有布局视图
    Buffer *-- DataStrategy : worker 持有
    Buffer ..> Handle : 返回
    Handle --> Buffer : 析构时释放引用
~~~

### 2.3 Lookup 首层预取

1. `LookupOnPrefix` 查询 Host Cache，并向下层存储查询未命中的 block。
2. 对下层存储确认命中的前缀 block，向在线 rank 的命令队列投递预取任务，随后返回 Lookup 结果。
3. rank 后台线程获取 Cache slot，将 block 的首个 shard（index 为 0）读入 Host Cache，完成后标记为 Ready。
4. 正式 Load 获取同一 slot。数据已 Ready 时直接执行 H2D；预取尚未完成时等待剩余读取。

预取范围为下层存储到 Host Cache，不包含 H2D，也不改变 Lookup 返回值。同一 `(block, shard)` 由取得 owner 的任务负责填充，其他任务复用其结果。命令队列满或没有在线 worker 时允许跳过预取，由正式 Load 按需读取。

读取成功后发布 Ready，失败时发布 Failed。等待中的 Load 遇到 Failed 返回重试状态；引用释放后，后续取得 owner 的加载任务可重新读取。

### 2.4 时序图

~~~mermaid
sequenceDiagram
    autonumber
    participant E as 推理引擎
    participant M as BufferManager
    participant P as PrefetchQueue
    participant C as Buffer（控制区与数据区）
    participant B as 下层存储
    participant L as LoadQueue
    participant N as NPU/GPU

    E->>M: LookupOnPrefix(blocks)
    M->>C: 查询 Host Cache
    C-->>M: 返回命中与未命中 block
    M->>B: 查询未命中 block
    B-->>M: 返回下层存储命中的前缀
    M->>C: 投递首层预取命令
    M-->>E: 返回 Lookup 结果

    par 后台预取
        P->>C: 消费本 rank 的命令
        P->>C: Get(block, shard 0)
        C-->>P: Handle
        opt 获得 owner 且数据未 Ready
            P->>B: Load 到 Handle.Data()
            B-->>P: 返回异步任务
            P->>B: Wait(task)
            B-->>P: 读取结果
            P->>C: 成功发布 Ready，失败发布 Failed
        end
        P->>C: 释放预取引用
    and 正式加载
        E->>L: 提交第一层 Load
        L->>C: Get(block, shard 0)
        C-->>L: Handle 与当前状态
        opt 本次 Load 获得 owner 且数据未 Ready
            L->>B: Load 到 Handle.Data()
            B-->>L: 返回异步任务
        end
        Note over L: 打点：首层 backend wait 开始
        alt 数据已 Ready
            L->>L: 跳过下层读取等待
        else 已有 owner 正在读取
            loop 等待 Ready 或 Failed
                L->>C: 读取 slot 状态
                C-->>L: 当前状态
            end
        else 本次 Load 获得 owner
            L->>B: Wait(task)
            B-->>L: 读取结果
            L->>C: 成功发布 Ready，失败发布 Failed
        end
        Note over L: 打点：首层 backend wait 结束
        alt 数据已 Ready
            L->>N: 提交 H2D 并等待传输完成
            L-->>E: 第一层数据就绪
        else 读取失败
            L-->>E: 返回错误或重试状态
        end
        L->>C: 释放加载引用
    end
~~~

## 3. 测试方法

本需求由开发自验证，计划在第一层等待路径打点，对比预取前后的 Wait 耗时。

测试数据满足下层存储已缓存、Host Cache 未缓存。保持输入及 Lookup 到 Load 的调度间隔一致，分别执行不触发预取和触发预取的加载流程。预期预取能够减少第一层等待下层读取的时间；预取完成时，该部分等待接近 0，H2D 耗时仍由正式 Load 承担。
