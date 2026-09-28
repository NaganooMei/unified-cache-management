# A5 Cache 扩展及后续合入 Develop 的方案

更新日期：2026-09-28。

状态：本地 `lzx/feature-a5` 已同步 upstream 至 `00291f98`（#1462、#1463）。上游已拆分 DataBackend，本方案的 Memfd、NUMA 和 SDMA 扩展尚未实现。详细约定见 [Cache 统一实现任务书（供编码 Agent 使用）](cache-platform-datastrategy-design.md)。

## 目标

直接保留 A5 分支已有的 connector、控制区/数据区分离和乐观无锁查找，在同一缓存核心上补齐普通内存、NUMA 和 SDMA 能力。初始化由 DataStrategy 统一编排，平台内存差异集中在 DataBackend；完成验证后再单独处理合入 develop。

## 要做的工作

1. **沿用上游 DataBackend 架构。** 保留统一 DataStrategy 的创建、发布、导入及 ready 屏障流程，平台差异放在 backend。保留 Buffer、控制区、缓存协议，以及 connector 的 unique ID 和 MLA dump 分工，不重复拆两套 DataStrategy。
2. **新增 MemfdDataBackend。** 用户确认保留 memfd_create 要求；在 #1462 接口下补 FD 传递、跨进程映射、注册和释放。已有 PosixShmDataBackend 保留作测试/参考；目标平台选择为 A5 → HAL、ascend/A3/CUDA/simu → Memfd。当前上游注释了 A5 HAL 开关且部分平台仍构建旧核心，需要显式修正构建入口。设备别名用新增接口提供，保留原有 Host/Device 地址互斥语义。
3. **统一自动 NUMA 分配。** DataStrategy 生成分配计划，HAL/Memfd backend 执行：检测到设备亲和性就使用亲和内存，检测不到则选择 `nodes[本机 worker rank % nodes.size()]`，整个本地数据段放在该节点；不提供用户 policy 配置。A5/A2/H100 对应亲和分配，A3 对应确定性轮转，最终以实际探测结果为准。Memfd 先绑定再触页、注册；A5 在 HAL 分配时落实目标节点。
4. **支持配置 SDMA stream 数。** 参考 develop 已有 SDMA Direct 适配接入 v2，配置贯通 connector、store 和 stream 创建/使用逻辑。普通拷贝默认 4、SDMA Direct 默认 16（Load/Dump 各 16 条），显式配置按指定数量执行；核对 Load/Dump 的分配与同步。

## 已确认的 rank 处理

feature_a5@3bf8dec4 的实现位于 ucm/store/cache/v2，已使用 myRank = deviceId % rankCount，并分别向 DataStrategy 传入 deviceId 和 myRank。

按 DP 隔离控制区且 rankCount=8 时，设备 0..7 和 8..15 都映射到各自控制区的 rank 0..7，可覆盖这种 DP2×TP8 部署。扩展时验证同一域内取模结果唯一即可；若每个 worker 的可见 deviceId 都是 0，则需额外适配。控制区 myRank 保持不变；NUMA fallback 单独传入本机跨 DP/PP/TP 的 worker rank，不能用 myRank 替代。

## 实施与验证

在现有 A5 开发分支同步最新 feature_a5 → 扩展已有 DataBackend 接口并新增 Memfd backend → 接入 NUMA 和 SDMA stream 配置 → 验证 A5 和非 A5 平台 → 后续单独处理 develop 合入。不要求新建 develop 功能分支。

保留 #1463 独立 Cache pipeline；独立 Store 调用也要传入 NUMA 需要的设备提示/本机 worker rank，不依赖 vLLM connector 必然存在。保留 POSIX backend 原测试，另补 Memfd 的 FD 传递、ready 屏障和清理测试。

扩展时核对旧配置，不能静默忽略；合入 develop 前完成商用兼容验证，重点包括私有/共享模式、容量及 FA/WA 拆分、DP/TP/PP 与多机部署、已有传输路径、初始化和失败清理。MLA dump 均匀分工与数据段实际均匀分布分别验证。

本地做代码和协议检查；设备注册、HAL、NUMA 放置、拷贝及端到端性能在 GPU/NPU 服务器验证。本次未实施或运行硬件验证。

codex/cache-rank-partition 不作为实现基线，不整体迁移；定向复用其 NUMA 探测、按本机 worker rank 取模选节点及绑定逻辑，其余作为历史问题和回归测试参考。
