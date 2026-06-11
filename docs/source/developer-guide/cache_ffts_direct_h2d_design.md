---
orphan: true
---

# CacheStore FFTS Direct H2D 方案

## 背景

CacheStore load 阶段原本使用普通 per-tensor H2D copy：一个 shard 从 backend 读入 host buffer 后，再按 `tensor_size_list` 切分，逐个 tensor 调用异步 H2D copy 写回各个 KV cache device address。

这条路径的问题是：DeepSeekV4 这类 KV cache layout 下，一个 shard 内部会对应大量碎片化 device 目标地址。碎片数量较多时，host 侧提交大量小 H2D copy 的开销会明显放大。

已有的 IO aggregation 路径解决的是另一个问题：它先把 host shard 拷到 device aggregation buffer，再用 FFTS scatter/gather 在 device 侧分发或聚合。本次方案不复用这条 staging buffer 设计，而是直接把 CacheStore load 阶段的 H2D copy 换成 FFTS SDMA descriptor。

## 目标

本次改动目标是：

- 在 CacheStore load 阶段提供 direct FFTS H2D 路径。
- 用 FFTS SDMA descriptor 替换普通 per-tensor `aclrtMemcpyAsync` H2D 提交。
- 保持 dump 阶段 D2H 仍走原 copy stream 路径。
- 保持和 `cache_io_aggregation` 语义互斥，避免两个优化路径混在一起。
- 支持两种 launch 组织方式：
  - `shard`：一个 CacheStore shard 对应一次 FFTS launch。
  - `task`：一个 CacheStore load task 内所有 shard 合并成一次 FFTS launch。

## 开关

新增构建开关：

```yaml
UCM_ENABLE_ASCEND_FFTS_DIRECT_H2D
```

新增运行时配置：

```yaml
cache_ffts_direct_h2d: false
cache_ffts_direct_h2d_launch_mode: "shard"
cache_ffts_direct_h2d_max_ready_lanes: 8
```

`cache_ffts_direct_h2d` 和 `cache_io_aggregation` 不能同时开启。前者是 direct host-to-device SDMA，后者是 host-to-device staging buffer 加 device-side scatter/gather。

## 数据路径

开启 `cache_ffts_direct_h2d` 后，load 路径保持 CacheStore 原有任务语义：

```text
CacheStore.Load
  -> TransManager
  -> LoadQueue
  -> backend.Load 到 TransBuffer host shard
  -> FFTS direct H2D 从 mapped host shard 写入 shard.addrs
```

对于一个 shard，host buffer 中的数据仍然按 `tensor_size_list` 顺序排列。direct H2D executor 会按相同顺序生成 FFTS copy spec：

```text
mapped_host + offset_0 -> shard.addrs[0], size=tensor_size_list[0]
mapped_host + offset_1 -> shard.addrs[1], size=tensor_size_list[1]
...
```

这些 copy spec 会交给 `FftsD2DDispatcher` 生成 SDMA context，并通过 FFTS Plus task 提交到 Ascend runtime。

## Buffer 映射

FFTS SDMA source 需要 device 侧可访问的 host mapped address。因此方案把 mapped host pointer 放在 `TransBuffer` 层处理，而不是在每次 H2D 提交时临时查询。

本地 buffer：

- 普通 pinned host buffer 在开启 direct H2D 后会额外 register 成 mapped host buffer。
- `io_direct` host buffer 已经通过 direct-IO buffer 路径完成 register，direct H2D 只取 mapped pointer。

shared buffer：

- shared memory data region 已经在 `SharedBufferStrategy` 中按 device register。
- direct H2D 直接使用该 shared region 对应的 mapped pointer。

这样 direct H2D 不要求必须使用 shm，也不要求必须使用 IO aggregation 的 staging buffer。

## Launch 模式

`cache_ffts_direct_h2d_launch_mode: "shard"`：

- 每个 shard 的 tensor fragments 组织成一次 FFTS launch。
- 粒度更小，提交时机和原 per-shard transfer 更接近。
- 适合先做功能验证，也便于观察单 shard 的行为。

`cache_ffts_direct_h2d_launch_mode: "task"`：

- LoadQueue 在一个 task 的多个 shard 上累积 copy specs。
- 到该 task 的最后一个 shard 时，合并所有 shard 的 fragments 做一次 FFTS launch。
- 可以减少 FFTS launch 次数，适合 shard 数较多、小 fragments 较多的场景。

两种模式下，completion 语义都保持在 LoadQueue task 末尾 `Synchronize` 处完成，和原 CacheStore load task 的 waiter 释放点一致。

## 同步语义

direct H2D executor 使用独立 FFTS stream 提交 load copy。

load 阶段：

- backend load 完成后，host shard buffer 标记 ready。
- direct H2D executor 生成 SDMA descriptors。
- `shard` 模式立即 launch。
- `task` 模式先缓存 descriptors，在 task 最后一个 shard 时统一 launch。
- task 末尾同步 FFTS stream，完成后释放 waiter。

dump 阶段：

- 不走 direct H2D executor。
- D2H 保持原 copy stream 逻辑。

如果外部 task 带 prerequisite event，executor 会让 copy stream 和 FFTS stream 都等待该 event，保证依赖关系不被绕过。

## 和 IO Aggregation 的边界

direct H2D 和 IO aggregation 的边界如下：

```text
direct H2D:
  host shard mapped address -> shard.addrs

IO aggregation:
  host shard -> device aggregation buffer -> shard.addrs
```

direct H2D 不创建 aggregation buffer，不做 device-side staging，也不改变 dump 路径。

因此两个开关互斥：

```text
cache_ffts_direct_h2d == true
cache_io_aggregation == false
```

## 配置建议

功能验证阶段建议：

```yaml
cache_ffts_direct_h2d: true
cache_ffts_direct_h2d_launch_mode: "shard"
cache_ffts_direct_h2d_max_ready_lanes: 8
cache_io_aggregation: false
```

性能对比阶段建议增加 `task` 模式：

```yaml
cache_ffts_direct_h2d: true
cache_ffts_direct_h2d_launch_mode: "task"
cache_ffts_direct_h2d_max_ready_lanes: 8
cache_io_aggregation: false
```

`cache_ffts_direct_h2d_max_ready_lanes` 控制 FFTS dependency graph 的 ready lanes 上限。默认 8，后续可以结合 shard 内 fragment 数和 profiling 结果调参。

## 验证重点

建议验证以下场景：

- `cache_ffts_direct_h2d=false` 时，原 CacheStore load/dump 行为不变。
- `cache_ffts_direct_h2d=true` 且 `launch_mode=shard` 时，load 能完成且命中数据正确。
- `cache_ffts_direct_h2d=true` 且 `launch_mode=task` 时，多 shard load 能完成且数据正确。
- `cache_ffts_direct_h2d=true` 和 `cache_io_aggregation=true` 同时开启时，配置校验应失败。
- shared buffer 和 local buffer 都能获得 mapped host pointer。
- dump 阶段指标和行为不被 direct H2D 开关改变。

## 风险点

- Ascend runtime 需要支持 FFTS Plus task 和 SDMA context。
- 非 Ascend runtime 下 direct H2D 编译宏会关闭，运行时开启该配置会报未编译。
- `task` 模式会把一个 CacheStore task 内所有 shard 的 copy specs 合并提交；如果单 task fragments 极多，需要关注 FFTS context 数量和 descriptor buffer 大小。
- mapped host buffer register 失败会导致 CacheStore setup 或 load 失败，应优先检查 CANN 版本、host memory register 权限和 direct-IO host buffer 配置。
