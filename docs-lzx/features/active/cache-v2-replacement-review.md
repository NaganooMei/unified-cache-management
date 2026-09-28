# Cache v2 替换检视与暂缓事项

日期：2026-09-28。代码基线：`lzx/feature-a5@c947b04f`。

**当前安排：先验证推理服务。以下四项仅记录，暂不修改代码；后续兼容性收尾时再处理。** 暂缓不代表问题已解决，也不代表当前版本已满足 develop 商用兼容验收。

## 已完成的切换

| PLATFORM | Cache 核心与入口 | 数据后端 |
|---|---|---|
| ascend-a5 | v2 | HAL |
| ascend / ascend-a3 / cuda | v2 | Memfd |
| simu | cc、v2 均编译，入口使用 v2 | Memfd |
| musa / maca | cc | 原实现 |

IO aggregation 已调用批量接口，普通 stream 使用基类逐片段实现，SDMA 使用自身批量实现。`c947b04f` 恢复提交失败后对当前 stream 的同步，再返回原始错误。上述分派和部分提交失败路径已通过实际 CopyStream/Trans::Stream 配合 mock transport 验证，未做硬件验证。

## 暂缓处理的四项问题

| 编号 | 优先级 | 问题与触发条件 | 后续处理方向 |
|---|---|---|---|
| R1 | P1 | 独立 Store 调用缺少 NUMA 输入。当前仅 vLLM connector 自动传入亲和节点或本机 worker rank；`cache_fake_bw.py` 没有这两个字段，初始化到 `Numa::Resolve()` 时会报 `no device affinity and no local worker rank`。MindIE 的 Cache pipeline 参数也尚未接入。 | 补齐独立脚本、MindIE 等调用入口的亲和性探测和本机 worker rank 传递，保持统一 NUMA 选择规则。 |
| R2 | P1 | MLA connector 仍以 `/dev/shm` 总容量检查整个缓存，但 v2 数据后端已经使用 Memfd/HAL。小 `/dev/shm` 容器可能在创建 Store 前被错误拒绝；HMA 也有同样检查。 | 按实际后端调整容量检查；保留既定 FA/WA 容量拆分。 |
| R3 | P2 | v2 的通用 `Lookup()` 直接返回 `Unsupported`，无法兼容 cc 的逐 block 命中结果。`cache_on_empty_test.py`、`cache_on_posix_test.py` 等仍有调用；此前只迁移了 `cache_fake_bw.py` 的查询方式。 | 若保留旧 API 则补实现；若只保留 prefix/reverse，则明确接口范围并迁移剩余调用方。 |
| R4 | P2 | CUDA GDR 路径未迁移。v2 读取 `use_gdr`，普通路径却始终创建 `MakeSharedStream()`；cc 原有的 `MakeGdrStream()` 选择及 GPU KV buffer 注册尚未接入。 | 迁移 GDR 支持，或明确拒绝该模式，避免配置被静默忽略。 |

定位（相对仓库根目录，行号对应上述基线）：

- R1：`ucm/store/cache/v2/data_strategy.h:193`、`numa/numa_policy.cc:187`（相对 v2）、`ucm/store/test/e2e/cache_fake_bw.py:407`；MindIE 入口为 `ucm/integration/mindie/unifiedcache_mempool.py`。
- R2：`ucm/integration/vllm/ucm_connector.py:1536`、`ucm/integration/vllm/hma_connector.py:719`。
- R3：`ucm/store/cache/v2/cache_store.cc:59`。
- R4：`ucm/store/cache/v2/copy_stream.h:56`，对照 `ucm/store/cache/cc/copy_stream.h` 和 `cc/cache_store.cc`。

R1 的脚本参数缺失、R2 的提前拒绝已通过提取实际 Python 函数并使用 mock 检查；R3/R4 为调用链静态检视结果。本轮没有修改业务代码，没有执行 Linux 多进程或 GPU/NPU 推理服务验证。

此前暂缓的 A5 HAL NUMA 落实、旧模式开关兼容、socket 超时问题继续保留，不在本轮展开整改。推理服务验证结果后续补充平台、模型、DP/TP 配置、代码版本、启动/推理结果及相关日志；当前不预填通过结论。
