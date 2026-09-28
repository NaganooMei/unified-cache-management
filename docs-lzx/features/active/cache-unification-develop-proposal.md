# A5 Cache 扩展及后续合入 Develop 的方案

更新日期：2026-09-28。

状态：A5 已比较稳定，直接在现有 `lzx/feature-a5` 分支补齐平台能力；尚未开始实现。详细约定见 [Cache 统一实现任务书（供编码 Agent 使用）](cache-platform-datastrategy-design.md)。

## 目标

直接保留 A5 分支已有的 connector、控制区/数据区分离和乐观无锁查找，在同一缓存核心上补齐普通内存、NUMA 和 SDMA 能力。平台内存差异集中在 DataStrategy；完成验证后再单独处理合入 develop。

## 要做的工作

1. **隔离 DataStrategy 实现。** 同步最新 feature_a5 后直接继续开发，抽出公共接口和 HAL 实现，通过构建时 PLATFORM 选择。保留现有 Buffer、控制区、缓存协议，以及 connector 的 unique ID 和 MLA dump 按模均匀分工。
2. **补普通内存 DataStrategy。** 非 A5 使用 memfd_create、ftruncate、mmap 创建共享数据内存，补齐 FD 传递、跨进程映射，以及传输路径需要的设备注册、Host/Device 地址和释放逻辑；A5 继续使用 HAL 数据实现。
3. **独立设计 NUMA 模块。** 模块负责节点选择、绑定及放置验证，由普通内存 DataStrategy 在创建数据段时调用。顺序为映射、NUMA 绑定、首次触页、设备注册、发布可用状态。明确拓扑或权限不足时的降级行为。
4. **支持配置 SDMA stream 数。** 参考 develop 已有 SDMA Direct 适配接入 v2，配置贯通 connector、store 和 stream 创建/使用逻辑。普通拷贝默认 4、SDMA Direct 默认 1，显式配置按指定数量执行；核对 Load/Dump 的分配与同步。

## 已确认的 rank 处理

feature_a5@3bf8dec4 的实现位于 ucm/store/cache/v2，已使用 myRank = deviceId % rankCount，并分别向 DataStrategy 传入 deviceId 和 myRank。

按 DP 隔离控制区且 rankCount=8 时，设备 0..7 和 8..15 都映射到各自控制区的 rank 0..7，可覆盖这种 DP2×TP8 部署。扩展时验证同一域内取模结果唯一即可；若每个 worker 的可见 deviceId 都是 0，则需额外适配。当前不单独重做 rank 接口。

## 实施与验证

在现有 A5 开发分支同步最新 feature_a5 → 隔离 HAL/公共接口 → 补齐 Memfd → 接入 NUMA 和 SDMA stream 配置 → 验证 A5 和非 A5 平台 → 后续单独处理 develop 合入。不要求新建 develop 功能分支。

扩展时核对旧配置，不能静默忽略；合入 develop 前完成商用兼容验证，重点包括私有/共享模式、容量及 FA/WA 拆分、DP/TP/PP 与多机部署、已有传输路径、初始化和失败清理。MLA dump 均匀分工与数据段实际均匀分布分别验证。

本地做代码和协议检查；设备注册、HAL、NUMA 放置、拷贝及端到端性能在 GPU/NPU 服务器验证。本次未实施或运行硬件验证。

codex/cache-rank-partition 不再作为实现基线，也不继续整理或整体迁移；仅保留为历史问题和回归测试参考。
