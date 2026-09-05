# Rank-striped SHM 的 NUMA 均分

## 一个开关做对照

本实现将 sandbox 中验证成功的首次触页前 NUMA 绑定接入 UCM 的 rank-striped 数据段。
普通共享缓冲保留当前分支原有的单文件布局和 `MAP_POPULATE` 分配路径，不执行 NUMA 均分。

在已开启共享缓冲的现有 `ucm_connector_config` 中，只需增加一个字段。
两组实验只切换 `share_buffer_rank_striped`：

```yaml
# false：原有普通 SHM；true：rank-striped + NUMA 均分 + 原有段轮转。
share_buffer_rank_striped: true
```

没有指定 `share_buffer_numa_nodes` 或指定 `[]` 时，默认固定使用物理 NUMA 节点
`[0, 1, 2, 3, 4, 5, 6, 7]`。其他拓扑可通过该参数覆盖；rank-striped=false 时完全忽略。
所有 worker 必须使用同一节点列表、相同顺序和 shard 大小。

vLLM connector 对 MLA 模型默认开启 `share_buffer_enable`，并按 TP 大小填入
`local_rank_size`，因此现有 MLA TP16 实验不需要在 YAML 中重复填写这两个参数。
`cache_buffer_capacity_gb: 32` 表示 32 GiB 缓冲容量，沿用已有容量设置即可。
`timeout_ms: 120000` 表示 120 秒超时；它是为大容量初始化留时间的可选示例，
不是 NUMA 开关，未配置时仍使用原有 30000 毫秒默认值。

节点列表必须非负、无重复；本地 rank 数必须是 NUMA 节点数的整数倍。
无法均分、节点不允许访问、系统调用失败或实际落点不符都会令初始化失败，不退回默认分配。
本实现直接调用 Linux 系统调用，无需新增 libnuma 依赖或开启 `BUILD_NUMA`。

## 数据段布局

每个 rank 仍创建一个数据段，segment `s` 整体绑定到 `nodes[s % nodes.size()]`。
16 个 segment、8 个 NUMA 节点时，每个节点承担两段。
32 GiB 配置下，每段约 2 GiB，每节点约 4 GiB；原有容量按 shard 和段数取整的规则不变，
所以有效 payload 可能略小于配置容量。

每个数据段都按同样的系统页大小取整，因此相同数量的等大段对应相同的物理页容量。
metadata 文件不属于均分目标，仍使用默认内存策略。
每个 rank 映射并注册所有段，已有 block 分配偏好、Load/Dump 的段轮转、注册版本和 stream 数不变。

## 创建与同步顺序

metadata 的创建者初始化共享锁和节点元信息，并在追加区域记录 NUMA 节点列表和 shard 大小。
各数据段的 owner 执行：

1. 使用 `O_CREAT | O_EXCL` 创建数据文件，扩容后惰性映射，不带 `MAP_POPULATE`。
2. 对整个数据段设置单节点 `MPOL_BIND | MPOL_F_STATIC_NODES`。
3. 完整清零数据段，让物理页按绑定策略分配。
4. 使用 `move_pages(nodes=nullptr, flags=0)` 分批查询所有基页；不迁移页面。
5. 验证通过后，以 release 发布段 ready；失败则发布段失败状态。

其他 rank 以 acquire 等待所有段就绪，再映射其他段并执行现有 Host Register。
节点绑定和逐页验证只发生在创建阶段，不进入请求路径。重新加入已有数据段的进程不清零、不迁移数据。
系统调用的 `maxnode` 使用掩码容量加一，保留 sandbox 中修复的 Linux 掩码边界处理。

rank-striped metadata 加入进程先等待文件扩容，再惰性映射、等待 header ready，并检查配置一致性。
普通 SHM 的加入进程保留原先的映射和初始化行为。

## 兼容与退出

普通 SHM 的 header 布局和版本保持原样。rank-striped header 的 ready 版本改为 4，
NUMA 配置记录追加在原 metadata 区域之后，不改变原有索引、锁和 meta 节点的偏移。
升级 rank-striped 实现后需整体重启相关进程并使用新的 `unique_id`，不能与旧版本共享同一组文件。
两组对照实验也应分别重新启动服务，让各自创建新的 SHM。

rank-striped 进程只 unlink 自己创建的文件，配置不匹配的加入进程不会删除正在使用的 metadata 或兄弟段。
本实现没有新增支持 owner 独立重启的跨进程引用计数协议。

## 验证方法

开启 rank-striped 的启动日志应包含实际生效的 `Rank-striped SHM NUMA nodes`，
以及每个数据段不受日志限流影响的绑定与验证记录：

```text
SHM NUMA bind: file=... offset=0 bytes=... node=...
SHM NUMA verify: file=... expectedNode=... actualNode=... pages=... mismatches=0.
SHM NUMA verified: file=... ranges=1.
```

服务 ready 后，选一个实际 worker PID 复核数据段：

```bash
pid=12345  # 替换为实际 worker PID
grep -E 'file=.*uc_shm_cache_.*_rs_data_[0-9]+' /proc/$pid/numa_maps
```

16 rank、节点列表 0～7 时，应看到 segment 0/8 在 N0、1/9 在 N1，依此类推。
只统计一个 worker 中的映射，不要把不同 rank 对同一共享段的记录相加。
逐页验证发生在注册前，服务 ready 后的检查可以验证运行环境是否改变了落点。
页分布均衡不等于任意时刻的读流量均衡，带宽仍需用服务实际 load 指标比较。

## 测试与当前验证边界

测试覆盖节点掩码边界、页布局、rank 数不可均分、普通 SHM 忽略 NUMA 配置、
rank metadata 尚未扩容、配置不一致时的失败与文件保留，以及并发 rank 初始化和 watcher 访问。
真实 NUMA 系统调用测试需要显式指定节点；可以用 simu 后端验证真实 Linux SHM 而不依赖 NPU：

```bash
cmake -S . -B build/numa-test -DRUNTIME_ENVIRONMENT=simu -DBUILD_UNIT_TESTS=ON
cmake --build build/numa-test --target ucmstore.test -j
UCM_TEST_NUMA_NODES=0-7 ./build/numa-test/ucm/store/test/ucmstore.test \
  --gtest_filter='UCCacheShmNumaTest.*:UCCacheTransBufferSharedTest.RankStripedPlacesAndSharesSegments'
```

Windows 开发机已执行 portable 页布局和节点掩码测试；没有可用 Linux/Ascend 环境，
完整编译、真实并发 SHM 测试及 UCM 服务端性能尚未实测。
服务端需沿用现有 Ascend 构建安装流程重新编译 CacheStore C++ 库，仅修改 YAML 不会更新库文件。
