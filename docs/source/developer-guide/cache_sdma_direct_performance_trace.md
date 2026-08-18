# Cache SDMA Direct 多 Stream 性能打点指南

## 目标

本打点用于定位 Cache SDMA Direct 在单 Stream 与多 Stream 之间出现的吞吐或长尾差异，重点回答以下问题：

- 请求进入 UCM 前是否已经发生 DP 或 TP 节奏偏移。
- Cache task 是否在排队、backend ready、H2D/D2H 提交或 stream 同步阶段变慢。
- shard 是否均匀分配到各条 stream。
- 最终长尾来自 CPU 侧异步提交，还是设备侧 stream drain。
- 95% 命中场景中，D2H dump 是否与其他请求的 H2D load 形成竞争。

打点默认关闭，不改变原有指标更新和拷贝流程。仅在性能诊断时开启。

## 开启方式

在 UCM connector 配置中增加：

```yaml
ucm_connectors:
  - ucm_connector_name: "UcmPipelineStore"
    ucm_connector_config:
      store_pipeline: "Cache|Fake"
      cache_sdma_direct: true
      cache_stream_number: 4
      cache_sdma_trace: true
```

启动日志应包含以下配置确认：

```text
Set CacheStore::StreamNumber to 4.
Set CacheStore::CacheSdmaDirect to true.
Set CacheStore::CacheSdmaTrace to true.
```

该开关同时启用 Direct、LayerWise、FAWA 请求层日志和 Cache C++ 队列日志。FAWA 会创建 FA 和 WA 两个 Store，必须分别检查它们的 `unique_id`、`shard_size` 和 stream 配置，避免把两个 Store 的同名 task id 混在一起。

## 日志格式和关联键

所有新增日志统一使用：

```text
[UCM_SDMA_TRACE]
```

FAWA 推荐使用以下组合键关联一条请求在 Python 和 C++ 中的执行：

```text
request_id
  -> FA/WA label
  -> native task id
  -> unique_id + device + task id
```

其中：

- `request_id` 关联服务 TTFT 和 E2E。
- `label` 区分 FA 与 WA。
- `task` 是 PipelineStore 返回的 native task id。
- `unique_id` 区分 FA Store、WA Store以及不同 rank 的 Store 实例。
- `device`、`dp_rank` 和 `tp_rank` 用于寻找拖慢整个请求的最慢 rank。

非 FAWA Direct 和 LayerWise 使用相同的 native task id 关联 C++ 日志。LayerWise 还需要保留 `batch + layer + request_id`，因为同一请求会按层产生多个 task。

## 打点位置

### FAWA 请求层

实现位置：

```text
ucm/integration/vllm/hma_connector.py
```

事件：

| 事件 | 位置 | 作用 |
| --- | --- | --- |
| `fawa_load_batch_begin` | `start_load_kv` 入口 | 记录当前 batch、请求数和 rank |
| `fawa_load_submit` | `_submit_load_task` | 建立 request、FA/WA 与 native task id 的映射 |
| `fawa_load_complete` | `_wait_load_task` | 同时记录 wait 阻塞时间和 task 生命周期 |
| `fawa_load_batch_complete` | `start_load_kv` 结束 | 记录一个 FAWA load batch 的总墙钟时间 |
| `fawa_dump_submit` | `_submit_dump_task` | 建立 FA/WA dump 与 native task id 的映射 |

`wait_block_ms` 只表示调用 `wait()` 后的剩余阻塞时间。FAWA 顺序等待多个 task，后等待的 task 可能已经完成，因此判断真实 task 长尾应优先使用 `lifetime_ms`。

### 非 FAWA Direct 请求层

实现位置：

```text
ucm/integration/vllm/ucm_connector.py
```

事件：

| 事件 | 位置 | 作用 |
| --- | --- | --- |
| `direct_load_batch_begin` | `start_load_kv` 入口 | 记录 batch、请求数和 rank |
| `direct_load_submit` | `submit_load` 返回后 | 建立 request、block 数与 native task id 的映射 |
| `direct_load_complete` | `wait_load` 返回后 | 记录每个请求的 wait 阻塞时间、task 生命周期和状态 |
| `direct_load_batch_complete` | `start_load_kv` 结束 | 记录整个 Direct load batch 的墙钟时间 |
| `direct_dump_submit` | `wait_for_save` 中提交 dump 后 | 建立 dump 请求集合、block 数与 native task id 的映射 |
| `connector_dump_complete` | dump wait 或 poll 完成后 | 记录 dump 的剩余阻塞时间、完整生命周期和状态，`mode=direct` |

Direct 会先提交一个 batch 中的所有 load，再依次等待。与 FAWA 相同，`wait_block_ms` 可能因等待顺序而偏小；对比 C++ `load_complete` 时应优先按 `task` 关联并查看 `lifetime_ms`。

### 非 FAWA LayerWise 请求层

实现位置：

```text
ucm/integration/vllm/ucm_connector.py
```

事件：

| 事件 | 位置 | 作用 |
| --- | --- | --- |
| `layerwise_load_batch_begin` | `start_load_kv` 入口 | 记录 batch、命中请求数和 rank |
| `layerwise_load_submit` | 每层 `submit_load` 返回后 | 建立 batch、layer、request 与 native task id 的映射 |
| `layerwise_load_complete` | 当前层 `wait_load` 返回后 | 记录单请求单层 task 的阻塞时间、生命周期和状态 |
| `layerwise_layer_complete` | 当前层等待并提交下一层后 | 记录该层 task 数、阻塞时间和是否存在下一层 |
| `layerwise_load_batch_complete` | `wait_for_save` 末尾 | 记录整个 layerwise forward 中 load wait 累计值和总墙钟时间 |
| `layerwise_dump_submit` | 每层 dump 提交后 | 建立 layer、请求集合、block 数与 native task id 的映射 |
| `connector_dump_complete` | dump wait 或 poll 完成后 | 记录 dump 生命周期，`mode=layerwise` 并携带 `layer` |

LayerWise 的关键对比值是 `layerwise_layer_complete blocking_ms`。如果只有个别层明显变慢，再用该层的 `task` 下钻到 C++ `load_complete`；如果每层 C++ task 正常而层间间隔变大，应优先检查模型计算、调度或其他通信，而不是 SDMA。

### Cache LoadQueue

实现位置：

```text
ucm/store/cache/cc/load_queue.cc
```

事件 `load_dispatch` 复用现有 metrics 点位，记录：

- `queue_wait_ms`
- `dispatch_ms`
- `backend_shards`
- `shards`
- `bytes`

Cache 完全命中时，`backend_shards` 应为 0。非 0 表示仍有 shard 下沉到 Fake、Posix 或其他 backend，不能把总时间全部归因于 SDMA。

事件 `load_complete` 记录：

- backend wait 的总和、最大值和最慢 shard。
- H2D async submit 的总和、最大值、最慢 shard 和对应 stream。
- 第一次 submit 到最后一次 submit 的 `submit_span_ms`。
- 最后一次 submit 后同步全部 stream 的 `sync_total_ms`。
- 第一次 submit 到全部 stream 完成的 `h2d_window_ms`。
- 从 Cache task 提交到完成的 `total_ms`。
- 每条 stream 的 shard 数、字节数、submit 累计时间和同步等待时间。

时间关系如下：

```text
Cache task submit
  -> queue wait
  -> backend dispatch / ready wait
  -> first H2D submit
  -> last H2D submit
  -> synchronize all streams
  -> Cache task complete
```

`cache_h2d_submit_ms` 是 CPU 侧异步提交成本，不是实际传输时间。`cache_h2d_sync_ms` 是最后一个 shard 提交后的剩余 drain，不包含此前已经与提交过程重叠完成的传输。完整拷贝窗口应看 `h2d_window_ms`。

### CopyStream

实现位置：

```text
ucm/store/cache/cc/copy_stream.h
```

打点返回每次异步提交实际选择的 stream index，并分别测量顺序同步各条 stream 时的等待时间。

`stream_sync_wait_ms` 是按 stream 顺序调用同步接口时观察到的剩余等待，不是各条 stream 独立的完整执行时间。例如：

```text
stream_sync_wait_ms=[300, 0, 0, 0]
```

表示同步第一条 stream 时等待了 300 ms；其他 stream 可能在这 300 ms 内同时完成，不能据此断言只有第一条 stream 慢。

### Cache DumpQueue

实现位置：

```text
ucm/store/cache/cc/dump_queue.cc
```

事件 `dump_complete` 记录：

- dump queue wait
- 实际发生 D2H 的 shard 数和字节数
- 每条 stream 的 shard/bytes 分配
- D2H async submit 的总和和最大值
- stream sync 总时间和每条 stream 的剩余等待
- 完整 `d2h_window_ms`
- prerequisite event wait、backend submit 和 task 总时间

事件 `dump_backend_complete` 记录异步 backend dump 的最终等待时间。

95% 命中不是纯 H2D。32K 请求通常仍会 dump 少量未命中 block，64K 请求的 dump 数量更多。并发请求下，前一请求的 D2H 可能与后一请求的 H2D 重叠，因此必须同时保留 load 和 dump 日志。

SDMA Direct 当前不支持 stream callback，`prereq_wait_ms` 可能无法从 `d2h_window_ms` 中完全拆出。此时 `sync_total_ms` 可能同时包含等待计算 event 和实际 D2H，不能直接当作纯 D2H 时间。

## 复用的现有 Metrics

新增日志沿用现有 metrics 的计时边界，建议同时保存测试窗口前后的 Prometheus 快照：

| Metrics | 含义 |
| --- | --- |
| `fawa_worker_start_load_kv_ms` | FAWA worker 生成并等待 load task 的总时间 |
| `fawa_worker_wait_wait_all_load_task_ms` | FAWA 等待全部 load task 的时间 |
| `load_duration` | 非 FAWA Direct load batch 的总时间 |
| `layerwise_wait_blocking_ms` | 非 FAWA LayerWise 单层等待 load task 的阻塞时间 |
| `layerwise_batch_load_wait_total_load_only_ms` | 非 FAWA LayerWise 纯 load batch 的 load 阻塞累计值 |
| `layerwise_batch_load_wait_total_load_save_ms` | 非 FAWA LayerWise load/save batch 的 load 阻塞累计值 |
| `layerwise_batch_total_ms` | 非 FAWA LayerWise 一个 batch 的总墙钟时间 |
| `cache_load_duration_ms` | Cache load task 端到端时间 |
| `cache_load_queue_wait_duration_ms` | Cache load 排队时间 |
| `cache_load_backend_submit_duration_ms` | buffer 准备和 backend submit 时间 |
| `cache_shard_backend_wait_ms` | 单 shard backend ready 等待 |
| `cache_h2d_submit_ms` | 单 shard H2D 异步提交成本 |
| `cache_h2d_sync_ms` | 最后 submit 后的 stream drain |
| `cache_dump_queue_wait_duration_ms` | Cache dump 排队时间 |
| `cache_dump_prereq_wait_ms` | dump 等待计算 event 的时间，Direct 路径可能不可用 |
| `cache_d2h_duration_ms` | dump 同步窗口，可能包含 prerequisite wait |
| `cache_dump_backend_wait_duration_ms` | backend dump 完成等待 |

Metrics 是累计 histogram。必须对测试前后做差，不能直接使用服务启动以来的累计值。Metrics 适合判断整体趋势，精确定位最慢 request、rank、task 和 stream 时应使用 trace 日志。

## 推荐实验

保持模型启动参数、请求 seed、并发数、32K/64K 输入和约 95% 命中率不变，仅调整：

| 组别 | SDMA Direct | Stream 数 |
| --- | --- | --- |
| A | 关闭 | 4 |
| B | 开启 | 1 |
| C | 开启 | 4 |

推荐交替运行：

```text
A -> B -> C -> B -> C
```

每组至少三轮。保存完整服务启动日志、测试端结果和 `[UCM_SDMA_TRACE]` 日志。

提取 trace：

```bash
rg '\[UCM_SDMA_TRACE\]' vllm_serve.log > sdma_trace.log
```

## 判读顺序

### 第一步：确认工作量一致

比较：

- request 数
- external hit/query
- FA/WA key 数
- `shards`、`processed_shards` 和 `bytes`
- `backend_shards`

工作量不一致时不要继续比较带宽。

### 第二步：寻找最慢 rank

同一 request 应比较所有 TP rank 的 `lifetime_ms` 和 `load_complete total_ms`，使用最大值作为请求 makespan。不要平均 rank，也不要把各 rank 的瞬时带宽相加。

### 第三步：拆解 Cache load

- `queue_wait_ms` 大：Cache dispatcher 或 CPU 调度拥塞。
- `backend_wait_max_ms` 大：数据未在 Cache ready，或 backend/shared ready 长尾。
- `submit_max_ms` 大：FFTS context 构造或 Runtime launch 路径值得继续细分。
- `submit_span_ms` 大但单次 submit 正常：大量 shard 提交本身形成串行 CPU 开销。
- `sync_total_ms` 大：设备侧 SDMA/FFTS 执行或 stream 排队长尾。
- `h2d_window_ms` 大：完整 H2D 路径变慢，是 stream 1/4 的主要对比值。

### 第四步：检查 stream 分配

对相同大小 shard，round-robin 应近似均匀。例如 60 个 shard、4 条 stream 应为 `[15, 15, 15, 15]`，121 个 shard应为 `[31, 30, 30, 30]`。

如果 shard/bytes 均匀但 `sync_total_ms` 明显波动，问题更可能在设备或 Runtime 调度，而不是 round-robin 数量分配。

### 第五步：检查 load/dump 重叠

按日志墙钟时间检查慢 load 周围是否存在 `dump_complete` 或较大的 `d2h_window_ms`。如果 stream 4 仅在 load/dump 重叠时变慢，应继续测试 FA/WA Store 分流或 load/dump stream 数分别配置。

## 注意事项

- Trace 使用 INFO 日志，仅用于诊断，性能结论完成后应关闭。
- 不要在每个 shard 后新增同步；这会破坏被测异步并发。
- 不要打印 tensor 地址、完整 hash 或每个 shard 的独立日志。
- Windows 静态检查不能证明 Ascend Runtime 行为，最终结论必须来自 Linux/A3 实测。
- 如果 `submit_max_ms` 已经定位到 CPU 提交侧，再在 SDMA Direct copier 内继续拆分 `BuildHostToDeviceSpecs`、`BuildCopies` 和 Runtime launch；首轮不应无条件打印每个 FFTS object。
