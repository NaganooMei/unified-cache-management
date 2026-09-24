# 基于稳定 A5 分支迁移 Develop Cache 的方案

更新日期：2026-09-24。

状态：方案已确认，等待 feature_a5 稳定可用后实施；当前只维护方案。

## 目标

直接复用稳定 A5 分支的 connector 和 Cache，实现 develop 的控制区/数据区分离、乐观无锁查找快路径。平台内存差异集中在 DataStrategy，避免维护两套缓存核心。

## 要做的工作

1. **迁移 connector 和 Cache。** 以稳定 A5 提交为基线，复用 Buffer、控制区及查找、分配、淘汰协议，同时迁移配套的 unique ID 和 MLA dump 按模均匀分工。迁移时核对 develop 已合入的变化，只补缺失部分。
2. **补普通内存 DataStrategy。** 非 A5 使用 memfd_create、ftruncate、mmap 创建共享数据内存，补齐 FD 传递、跨进程映射，以及传输路径需要的设备注册、Host/Device 地址和释放逻辑；A5 继续使用 HAL 数据实现。
3. **独立设计 NUMA 模块。** 模块负责节点选择、绑定及放置验证，由普通内存 DataStrategy 在创建数据段时调用。顺序为映射、NUMA 绑定、首次触页、设备注册、发布可用状态。明确拓扑或权限不足时的降级行为。
4. **支持配置 SDMA stream 数。** 配置贯通 connector、store 和 SDMA stream 创建/使用逻辑，核对 Load/Dump 的分配与同步。默认值调整根据独立性能验证决定。

## 已确认的 rank 处理

feature_a5@3bf8dec4 的实现位于 ucm/store/cache/v2，已使用 myRank = deviceId % rankCount，并分别向 DataStrategy 传入 deviceId 和 myRank。

按 DP 隔离控制区且 rankCount=8 时，设备 0..7 和 8..15 都映射到各自控制区的 rank 0..7，可覆盖这种 DP2×TP8 部署。迁移时验证同一域内取模结果唯一即可；若每个 worker 的可见 deviceId 都是 0，则需额外适配。当前不单独重做 rank 接口。

## 实施与验证

等待 A5 稳定 → 对照最新 develop 迁移 connector/Cache → 补普通内存 DataStrategy → 接入 NUMA 和 SDMA stream 配置 → 完成兼容性与性能验证。

商用迁移重点核对：私有/共享模式、容量及 FA/WA 拆分、DP/TP/PP 与多机部署、已有传输路径、初始化和失败清理。MLA dump 均匀分工与数据段实际均匀分布分别验证。

本地做代码和协议检查；设备注册、HAL、NUMA 放置、拷贝及端到端性能在 GPU/NPU 服务器验证。本次未实施或运行硬件验证。

codex/cache-rank-partition 不再作为实现基线，也不继续整理或整体迁移；仅保留为历史问题和回归测试参考。
