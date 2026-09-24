# UCM Engineering Notes

该目录保存个人维护的 UCM 工程设计、功能状态、性能分析和开发记录。内容用于长期研究与方案评审，不等同于已经合入主仓或已经完成硬件验证的正式文档。

## 目录

- `architecture/`：架构与数据流说明。
- `features/active/`：正在设计、实现或验证的功能。
- `features/completed/`：已经完成并保留的功能记录。
- `investigation/`：问题分析、实验和结论。
- `performance/`：性能数据、指标定义和优化记录。
- `development/`：开发与验证流程。

## 当前文档

- [基于稳定 A5 分支迁移 Develop Cache 的方案](features/active/cache-unification-develop-proposal.md)
- [Cache 内存按 Rank 划分、NUMA 绑定与 Lookup 首层预取](features/active/cache_rank_partitioned_memory_bandwidth.md)
- [A5 Cache2 控制区布局：Bucket 与 SlotMeta](features/active/a5-cache2-control-layout.md)

功能实现应在独立 feature 分支完成。本分支只维护文档，并长期保留稳定的 GitHub 链接；需要合入主仓的正式文档再由对应功能 PR 选择性带入。
