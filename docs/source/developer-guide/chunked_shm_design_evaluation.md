# 分块 SHM 方案评估与融合建议

评估基线：`debug-perf-2609` 的 `e19a131d` 与 `debug-perf-2609-mag1c` 文档编写前的 `0e932226`。本文只评估设计与实现取舍，不修改产品代码。

## 1. 结论

最终方案不应在 `debug-perf-2609` 与 `debug-perf-2609-mag1c` 之间整版二选一，建议采用以下组合：

- **架构抽象采用 mag1c 方案**：拆分 `Buffer`、control layout、control lifecycle、data lifecycle 与 Handle lifetime，避免继续扩大单个 `BufferStrategy` 接口。
- **共享内存实现采用 `debug-perf-2609` 方案**：保留按 cache domain 计算总容量、一个共享 metadata 区加多个 data segment、基于 `unique_id` 的命名隔离、NUMA 初始化、segment ready 协议，以及启动阶段完成 peer segment 映射和 host register。
- **Load/Dump 数据路径采用 `debug-perf-2609` 方案**：保留确定性的 segment placement、`RearrangeIndex`、实际 segment 回传、下一 shard 同段预分配、即时进入 transfer queue，以及已经验证过的可变 stream 路径。
- **缓存查找和淘汰短期保留已验证逻辑**：不在第一次融合时同时引入 mag1c 的 optimistic lookup 和 lock-free pin 协议；该部分应作为独立优化评估。
- **预取采用混合方案**：可复用 mag1c 的共享控制面命令队列，但实际 S2H 必须接入 demand-aware 的 Load 调度；metadata-only `Prealloc` 改为有界、可失败、低优先级的 `TryPrealloc`。

一句话概括：**采用 mag1c 的职责边界和资源管理方式，采用 `debug-perf-2609` 的 SHM 语义、placement 与传输实现。**

## 2. 需求和不变量

分块 SHM 的目标不只是把一个大文件拆成多个文件，还需要同时满足：

1. 所有参与者看到一致的缓存目录，能够对同一 `(BlockId, shard)` 协调 owner、引用计数和 Ready 状态。
2. payload 可以按 rank/NUMA 拆分，避免一块超大 SHM 的创建、first-touch、注册和带宽热点。
3. SHM 中只保存稳定的逻辑索引，不保存进程本地虚拟地址。
4. GQA 的 rank-specific KV 不能被其他 rank 错误复用；MLA 的公共 KV 可以跨 rank 复用。
5. `cache_buffer_capacity_gb` 的语义必须明确且保持兼容。
6. speculative work（预取、预分配）不能阻塞真实 Load/Dump。
7. 新架构不能以牺牲已验证的传输并行度为代价。

## 3. 两版实现的真实结构

### 3.1 `debug-perf-2609`

主要代码位于：

- `ucm/store/cache/cc/trans_buffer.{h,cc}`
- `ucm/store/cache/cc/load_queue.cc`
- `ucm/store/cache/cc/dump_queue.cc`
- `ucm/store/cache/cc/shm_numa.h`
- `ucm/store/cache/cc/shm_numa_layout.h`

结构为：

```text
TransBuffer
  -> BufferStrategy
       -> SharedBufferStrategy
            -> RankStripedSharedBufferStrategy
```

`BufferStrategy` 同时暴露锁、metadata、淘汰候选、host/device 地址和 segment 路由。分块实现使用一个 `_rs_meta` SHM 和 `localRankSize` 个 `_rs_data_<segment>` SHM。

`cache_buffer_capacity_gb` 表示一个 cache domain 的**总容量**。总 slot 数平均切分到所有 segment：

```text
totalSlots = bufferCapacity / slotSize
slotsPerSegment = totalSlots / segmentCount
```

每个 rank 创建自己的 data segment，但所有 worker 在启动阶段映射和 register 所有 segment。全局 `iNode` 通过以下方式定位数据：

```text
segment = iNode / slotsPerSegment
localSlot = iNode % slotsPerSegment
address = dataBases[segment] + localSlot * slotSize
```

Load/Dump 使用 `originalIndex % localRankSize` 作为 `preferredSegment`。已有条目保留实际位置；新条目优先在指定 segment 分配，目标 segment 无可用 slot 时允许向其他 segment 溢出。

优势：

- 总容量、NUMA 与 segment placement 语义清晰；
- MLA worker0-only Dump 可以把 shard 打散到全部 data segment；
- Load/Dump 已完成 segment-aware 调度和可变 stream 优化；
- 当前有明确的性能测试结果。

局限：

- 一个 Strategy 接口同时承担缓存算法、锁、布局、SHM、NUMA 和 device register；
- 所有 peer segment 启动时全部映射/register，启动成本和资源占用随 rank 数增长；
- `Handle` 可复制，引用生命周期需要额外推理；
- `Prealloc` 复用无界 `Alloc`，speculative work 可能阻塞 demand dispatcher；
- data SHM 的 unlink 责任需要进一步收敛到明确 owner。

### 3.2 `debug-perf-2609-mag1c`

活跃实现已经从旧 `TransBuffer` 切换到：

- `ucm/store/cache/cc/cache_buffer.h`
- `ucm/store/cache/cc/cache_types.h`
- `ucm/store/cache/cc/ctrl_layout.h`
- `ucm/store/cache/cc/ctrl_strategy.h`
- `ucm/store/cache/cc/data_strategy.h`
- `ucm/store/cache/cc/prefetch_queue.h`

结构为：

```text
Buffer
  -> CtrlStrategy
       -> CtrlLayout
  -> DataStrategy
  -> move-only Handle
```

主要设计：

- `CtrlLayout` 只描述 Header、bucket、striped lock、SlotMeta 和 PrefetchRing 的内存布局。
- `CtrlStrategy` 通过 memfd 和 UNIX socket 传递 FD，负责 control 区的创建、加入和生命周期。
- `DataStrategy` 让每个 rank 创建/register 自己的数据段；远端数据段首次访问时才 mmap/register。
- `Buffer` 只处理查找、pin、owner、状态和 CLOCK 淘汰。
- `Handle` 只能 move，析构自动释放 pin。

优势：

- control plane、data plane、缓存算法和 lifetime 的职责边界清晰；
- 每个数据段有明确的创建者；
- joiner 从共享 Header 读取布局参数，版本和 ready 发布顺序明确；
- remote mapping 是进程本地资源，并有集中清理位置；
- move-only Handle 更容易证明引用计数配对。

当前问题：

- `cache_buffer_capacity_gb` 被解释为**每 rank 容量**，总数据量约为活跃 rank 数乘以配置值；control metadata 又按 `kMaxRanks` 预留。
- control/data socket 名称固定，未使用 `unique_id` 隔离多个 cache domain。
- worker 的新分配严格限制在 `myRank` slot 范围；该 placement 适合 GQA，但不适合 MLA worker0-only Dump。
- `share_buffer_enable=false` 不再让 device worker 选择本地 Buffer；它只在 scheduler 和默认容量计算中残余生效。
- remote register 首次失败后不重试，rank ready 又是 sticky 状态，异常退出和重启语义不完整。
- optimistic lookup、原子 pin、cache-line padding 和 lazy mapping 的实际性能尚未拆解。
- 新 `PrefetchQueue` 绕过正常 LoadQueue，可能与真实请求竞争 backend queue、带宽和 cache slot。

## 4. GQA 与 MLA 的 placement 差异

### 4.1 GQA

GQA 中每个 worker Dump 自己的 rank-specific KV，Connector 使用 rank-specific Store BlockId。

- `debug-perf-2609` 的非共享模式为每个 worker 提供独立本地 buffer。
- mag1c 的共享 control 加 local-only allocation 也会形成 `rank N -> data segment N`，数据面通常不发生跨 rank 命中。

因此 mag1c 的 placement 对 GQA 基本合理，但它依赖 Connector 对 BlockId 做 rank 隔离；`share_buffer_enable=false` 本身没有在 CacheStore 内禁止跨 rank 复用。

### 4.2 MLA

MLA 的各 TP rank KV payload 相同，当前只有 worker0 执行 Dump。

`debug-perf-2609`：

```text
worker0 Dump
  -> shard 0 -> segment 0
  -> shard 1 -> segment 1
  -> ...
  -> shard N -> segment N % rankCount
```

mag1c：

```text
worker0 Dump
  -> all new slots -> worker0 data segment
other workers Load
  -> global metadata hit
  -> lazy map/register worker0 data segment
  -> remote H2D
```

mag1c 将“每个 rank 拥有自己的 data segment”与“只能在自己的 segment 分配”绑定在一起。前者是合理的 resource ownership；后者只是 placement policy，不适合 MLA worker0-only Dump。最终设计应允许 owner 创建 segment，但不能因此禁止共享模式下的显式跨 segment placement。

## 5. 推荐的目标结构

### 5.1 类和接口

采用 mag1c 的组合结构，但替换其具体 SHM 和传输实现：

```text
Buffer
  -> CtrlStrategy interface
       -> PosixSharedCtrlStrategy
  -> DataStrategy interface
       -> RankStripedPosixDataStrategy
       -> LocalDataStrategy
  -> move-only Handle
```

职责建议：

- `Buffer`：key lookup、owner、state、reference、淘汰流程。
- `CtrlLayout`：共享 metadata 的字段、offset、版本和兼容校验。
- `CtrlStrategy`：control SHM 的创建、attach、owner 和 cleanup。
- `DataStrategy`：segment 创建、NUMA、mmap、host register、地址解析和 cleanup。
- `PlacementPolicy`：决定新 slot 的 preferred/required segment，不与 `DataStrategy` 的物理 ownership 绑定。
- `Handle`：move-only pin；同时保留 `Segment()` 和具体失败状态。

### 5.2 SHM 创建和容量

采用 `debug-perf-2609` 的行为：

- 使用 `unique_id` 构造 domain 唯一的 metadata/data 名称；
- 一个 control/metadata SHM，加实际 rank 数量的数据段；
- `cache_buffer_capacity_gb` 表示整个 cache domain 的总 data capacity；
- 按实际 rank 数切分 slot，不按 `kMaxRanks` 扩大 payload 或 metadata；
- 每个 rank 创建并 NUMA 初始化自己的 segment；
- 所有 segment ready 后再进入可服务状态；
- 第一阶段继续启动时 mmap/register 所有 segment，避免把冷 mmap/register 放入首个 Load 热路径；lazy remote mapping 可在独立性能验证后作为可选策略。

cleanup 则采用 mag1c 的 ownership 思路：mapping、FD、host registration 和 SHM name 分别由明确对象管理；只有 domain owner 在确认退出协议后 unlink，普通 joiner 只释放自己的进程本地资源。

### 5.3 placement

- `share_buffer_enable=false`：保留真正的 local-buffer 语义，metadata 与 data 均不跨 worker 共享。
- `share_buffer_enable=true`：使用全局 slot 编号和显式 placement。
- MLA：使用 `originalIndex % segmentCount` 或等价的确定性策略打散 worker0 Dump 和首次 Load。
- GQA：默认 local placement；rank-specific key 继续用于后端和缓存命名空间隔离。
- 已存在的 key 永远返回实际 segment，不能因本次 request position 改变而迁移。
- preferred segment 不可用时允许有界 fallback；是否允许跨 segment fallback 应作为 policy 参数，而不是写死在 data strategy 中。

### 5.4 Load

采用 `debug-perf-2609` 的实现语义：

1. `RearrangeIndex` 只改变处理顺序，不搬运数据。
2. `Get` 接受 preferred segment，并返回实际 segment。
3. owner 立即提交 backend S2H，并尽快把 shard task 推入 transfer queue，保留 S2H/H2D overlap。
4. 下一 shard 的 metadata 预分配使用当前 Handle 的实际 segment。
5. H2D 使用可配置的多 stream `CopyStream`，保持当前已验证路径。

缓存命中、backend wait、H2D submit 和 stream sync 的指标定义保持稳定，避免架构重构同时改变观测口径。

### 5.5 Dump

采用 `debug-perf-2609` 的实现语义：

1. GQA 各 worker 独立 Dump 自己的 rank-specific KV。
2. MLA 仍由 worker0 Dump，但使用 segment-aware `Get` 将 shard 打散到全部 data segment。
3. D2H 的 host/device 地址来自 Handle 的实际 segment。
4. stream sync 后统一 `MarkReady`，再向 backend 提交 host 地址。
5. 保留 backend Dump 的异步完成与 Handle lifetime。

这避免了 mag1c 中 MLA 数据全部集中到 worker0 segment、其他 segment 空闲、所有 rank 后续集中读取 worker0 SHM 的问题。

### 5.6 预取

需要严格区分两种能力：

- `Prealloc/TryPrealloc`：只创建下一 shard 的 metadata placeholder，不做 S2H/H2D。
- backend prefetch：执行 Posix -> SHM 的真实 S2H，但不提前执行 H2D。

建议：

- 采用 mag1c 的 control-plane command ring 思路，使 scheduler 可以向目标 rank/segment 投递 best-effort hint；
- command 至少包含 BlockId、shard/offset 和 placement hint，不能固定只支持 shard 0；
- 不直接在独立 `PrefetchQueue` 中同步 `backend_->Load + Wait`；将其接入与真实 Load 共用、可观测且 demand-aware 的调度器；
- demand queue 非空时延后或跳过 prefetch；限制在途数量、占用 slot 数和 backend 带宽；
- `TryPrealloc` 必须有最大尝试次数并返回结果，失败不能影响真实 Load；
- 记录 enqueue、drop、deduplicate、backend submit、ready、later-hit 和 wasted-eviction 指标。

## 6. 逐项取舍

| 关注点 | 采用方案 | 结论 |
|---|---|---|
| 类职责拆分 | mag1c | `Buffer/CtrlLayout/CtrlStrategy/DataStrategy` 组合优于宽泛的 Strategy 继承接口 |
| Handle lifetime | mag1c 为主 | move-only；补回 `Segment()` 和具体 error status |
| control/data 分离 | mag1c 抽象 + 本方案实现 | 保留清晰边界，但不采用当前固定 socket 和 `kMaxRanks` sizing |
| SHM 创建机制 | `debug-perf-2609` | POSIX SHM、`unique_id`、一个 metadata 加 N 个 data segment |
| 容量语义 | `debug-perf-2609` | 配置表示 domain 总容量，避免按 rank 静默放大 |
| NUMA | `debug-perf-2609` | 保留 segment NUMA 初始化和校验 |
| peer mmap/register | `debug-perf-2609` 第一阶段 | 启动时完成，避免首请求冷路径；lazy 方案后续独立评估 |
| placement | `debug-perf-2609` | 显式 preferred/actual segment；不能硬编码为 `myRank` |
| GQA local 模式 | `debug-perf-2609` | `share_buffer_enable=false` 恢复物理 local-buffer 语义 |
| Load | `debug-perf-2609` | 保留 segment-aware 排序、即时 enqueue、overlap 和多 stream |
| Dump | `debug-perf-2609` | MLA worker0 Dump 仍能打散到所有 segment |
| metadata lookup/pin | 短期 `debug-perf-2609` | mag1c 原子协议单独验证，不与架构迁移绑定上线 |
| cleanup ownership | mag1c | 明确 creator/joiner 与 FD/mapping/register/unlink 责任 |
| metadata Prealloc | `debug-perf-2609` 意图，需修正 | 改为有界、best-effort `TryPrealloc` |
| backend prefetch | mag1c 控制面 + 现有 Load 引擎 | hint 可丢弃，执行必须让 demand 优先并纳入统一指标 |

## 7. 架构质量与性能结论分开

可以从代码确认：mag1c 在职责拆分、move-only Handle、control/data lifecycle 和发布顺序上更清晰；`debug-perf-2609` 在容量、NUMA、placement 和 Load/Dump overlap 上更符合当前需求。

当前性能事实：

- `debug-perf-2609` 优化后，Cache 命中 TTFT 从约 975 ms 降至约 740 ms；单层 Cache Load 平均从 7.46 ms 降至 3.1 ms，P99 从 18.6 ms 降至 4.35 ms。
- Posix 命中的 Cache 总时延仍约为平均 14 ms、P99 40 ms；H2D sync 从约 2 ms 降至 1.25 ms，但总时延基本不变，说明该收益被更长的未重叠阶段或既有 S2H 路径掩盖，不能只凭一个 sync 指标判断端到端收益。
- mag1c 当前观察到更大的性能问题，但没有时延拆解，因此不能确认是架构问题还是具体实现问题。

需要 profiling 才能确认：

- mag1c 的实际瓶颈来自 optimistic pin、control metadata cache locality、lazy remote map/register、local-only placement，还是独立 PrefetchQueue 对 backend 的竞争；
- MLA 集中访问 worker0 segment 是否造成 NUMA/内存带宽热点；
- 启动时全量 register 与首次访问 lazy register 的真实收益边界；
- metadata-only Prealloc 是否减少分配开销，还是通过淘汰和 dispatcher 阻塞造成负收益。

## 8. 推荐实施顺序

1. **冻结行为契约**：明确 capacity、`share_buffer_enable`、GQA/MLA key、placement、cleanup 和失败恢复语义，并补充测试。
2. **只迁移抽象**：引入 `Buffer/CtrlLayout/CtrlStrategy/DataStrategy` 边界和 move-only Handle，内部继续调用现有 SHM、锁和传输逻辑，要求性能不回退。
3. **迁移 SHM 实现**：把 rank-striped POSIX SHM、NUMA、segment ready 和地址解析移动到新的 strategy 对象，不改变数据布局和运行路径。
4. **迁移 Load/Dump**：保持现有排序、placement、enqueue 时机和 stream 行为，使用阶段指标做逐步 A/B。
5. **接入有界预取**：先完成 `TryPrealloc`，再引入受 demand 优先级约束的真实 S2H prefetch。
6. **独立评估优化项**：分别验证 optimistic lookup、lock-free pin 和 lazy remote mapping，只有数据证明收益后再合入。

## 9. 评审 open questions

1. `cache_buffer_capacity_gb` 是 domain 总容量还是每 rank 容量？
2. `share_buffer_enable=false` 是否仍是正式支持的物理 local-buffer 模式？
3. MLA worker0-only Dump 是否必须使用全部 segment 容量和 NUMA 带宽？
4. placement 由 request position、BlockId hash、scheduler 指派还是首次 owner race 决定？
5. data segment 的 creator、运行期 owner 和最终 unlink owner 分别是谁？
6. rank 进程异常退出或重启时，Ready、SlotMeta、socket 和 data segment 如何恢复？
7. 是否接受首个远端 hit 承担 mmap/register 时延？失败是否重试？
8. PrefetchRing 的单 producer 假设是否覆盖全部 connector/thread 模式？
9. prefetch 如何保证不抢占 demand backend queue、slot 和内存带宽？
10. mag1c 的大幅性能下降需要哪些最小消融：关闭 prefetch、预热 remote mapping、只测本地 hit、只替换 lookup/pin？

## 10. 验收标准

- `share_buffer_enable=false` 下，不同 worker 使用物理独立的 host cache。
- `share_buffer_enable=true` 下，共享 metadata 对同一 key 只产生一个 owner。
- 配置总容量与所有 data segment 实际大小之和一致。
- MLA worker0 Dump 后，slot 在所有 segment 上近似均匀分布；GQA 数据保持 rank-local。
- 所有进程看到相同的 global slot -> segment/localSlot 映射。
- Handle move、析构、失败和异步 backend 完成不会提前释放或重复释放 slot。
- rank 初始化失败、remote register 失败和进程重启不会留下可见但不可用的 Ready 状态。
- speculative work 在所有 slot 被 pin 时能够快速失败，真实 Load 不被阻塞。
- 关闭预取时，融合版本不得回退当前 `debug-perf-2609` 的 Cache 命中与 Posix 命中性能；开启预取必须分别报告命中收益和资源竞争成本。
