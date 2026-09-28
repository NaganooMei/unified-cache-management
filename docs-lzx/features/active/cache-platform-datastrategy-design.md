# Cache 统一实现任务书（供编码 Agent 使用）

更新：2026-09-28。本文是实现要求，不是已完成的代码说明。

## 1. 任务与基线

直接在现有 A5 开发分支 `lzx/feature-a5` 上实现，开始前同步最新 `upstream/feature_a5`。已同步基线 `00291f98`，包含 #1462（`be85ba41`，DataBackend 拆分）和 #1463（独立 Cache pipeline）。保留 connector 和 `ucm/store/cache/v2/`；不另建 develop 功能分支，不整体覆盖目录。

develop 只用于核对兼容性和参考已有传输实现（已核对 `2d24ab39`）。本轮完成 A5 上的功能扩展和验证，后续再单独处理合入 develop。

只实现四项：

1. 沿用 #1462 的统一 DataStrategy + DataBackend 架构，扩展初始化参数和地址接口，不再拆两套 DataStrategy。
2. 新增 MemfdDataBackend，通过构建时 PLATFORM 选择；保留 AscendHalDataBackend 和已有 PosixShmDataBackend。用户已确认普通平台仍使用 memfd_create，不以 shm_open 替代。
3. 增加独立 NUMA 模块，HAL/Memfd backend 共用：检测到设备亲和性就使用亲和内存，否则按本机 worker rank 在可用节点间轮转分配。
4. 支持配置 SDMA Direct stream 数。

不整体迁移 codex/cache-rank-partition；只复用其 NUMA 探测、按 rank 选节点及绑定逻辑。不重写缓存算法，不附带增加预取、线程池或 Posix 优化。

## 2. 文件与构建

以下路径相对仓库根目录；在现有文件上扩展，新增文件名按本表执行：

| 文件 | 修改内容 |
|---|---|
| `setup.py`、根 `CMakeLists.txt`、`ucm/store/cache/CMakeLists.txt` | PLATFORM → RUNTIME_ENVIRONMENT → backend 工厂选择；调整各平台 v2 构建入口 |
| `ucm/store/cache/v2/data_strategy.h` | 保留初始化编排和 backend_；传递 options，扩展设备别名访问 |
| `ucm/store/cache/v2/data_backend.h` | 扩展 BackendOptions 和 HostDeviceAddrOf；保持公共接口无 SDK 类型 |
| `ucm/store/cache/v2/data_backend_ascend.h/.cc` | 保留已有 HAL 实现，接入统一 NUMA 选择结果 |
| `ucm/store/cache/v2/data_backend_memfd.h/.cc` | 新增 MemfdDataBackend：FD 服务、映射、注册和释放 |
| `ucm/store/cache/v2/data_backend_posix.h/.cc`、`posix_shm.h` | 保留 #1462 的 named SHM 实现及测试，不作为本轮普通平台默认后端 |
| `ucm/store/cache/v2/numa/numa_policy.h/.cc` | 新增自动 NUMA 选择与绑定模块，不提供用户 policy 开关 |
| `ucm/shared/trans/ascend/hal/hal_memory.h/.cc` | 将 NUMA 选择结果传给 HAL 分配路径，具体参数核对目标 SDK |
| `ucm/store/cache/v2/global_config.h`、`cache_buffer.h` | 解析新配置，组装 DataOptions，调用数据区初始化 |
| `ucm/store/cache/v2/copy_stream.h`、Load/Dump 队列 | 接入所需传输模式、地址选择及 stream 数 |
| `ucm/integration/vllm/` | 保留已有 unique ID、MLA dump 分工；透传设备 NUMA 探测结果和本机 worker rank |

平台选择规则：

- `PLATFORM=ascend-a5`：工厂选择 AscendHalDataBackend。
- `PLATFORM=ascend/ascend-a3/cuda/simu`：工厂选择 MemfdDataBackend；simu 使用现有模拟注册接口。
- musa/maca：保留 A5 分支内已有的 `ucm/store/cache/cc/` 路径，不宣称支持新实现。

现状注意：#1462 将根 CMake 的 `set(UCM_RUNTIME_ASCEND_HAL ON)` 注释掉了，目前 A5 默认落到 POSIX 后端；cache/CMake 也尚未让 ascend/A3/CUDA 走 v2。实施时必须落实上述目标矩阵，不能把当前配置误认为已完成平台隔离。

多个 backend 类可以共存，但工厂默认选择必须唯一；HAL 源码及 SDK 依赖只在 A5 启用，不再要求“两份同名 DataStrategy 只能编译一份”。修改 PLATFORM 后重新构建，运行时不据此切换；直接 CMake 使用 RUNTIME_ENVIRONMENT，非法非空 PLATFORM 报错。

## 3. DataStrategy / DataBackend 接口

DataStrategy 负责校验、发布、导入和 ready 屏障；DataBackend 负责实际内存。沿用现有 `std::unique_ptr<DataBackend> backend_`，不再引入 Impl 或搬迁 HAL 文件。以下为需实现的 C++17 接口约定，其余方法保持现有签名。

```cpp
namespace UC::Cache2 {
struct DataOptions {
    std::string domainId;            // connector 的 uniqueId，不自行改共享范围
    int32_t deviceId{-1};            // 设备 ordinal
    size_t myRank{0};                // 控制区分区编号
    size_t slotSize{0};
    size_t slotsPerRank{0};
    size_t setupTimeoutMs{600000};   // 整次 Setup 总预算；0 表示不等待
    bool requireHostDeviceAddress{false}; // 由有效拷贝模式推导，不是用户开关
    std::optional<int32_t> deviceNumaNode; // 设备层探测结果，未知为空；不是用户配置
    std::optional<size_t> fallbackNumaRank; // connector 推导的本机 worker rank，不是 myRank
};
struct BackendOptions {
    int32_t deviceId;
    size_t rankCount;
    size_t rankBytes;
    Numa::Plan localPlacement;
    bool requireHostDeviceAddress;
    std::chrono::steady_clock::time_point deadline;
};
class DataBackend {
public:
    virtual Status Setup(const BackendOptions& options) = 0;
    virtual void* HostDeviceAddrOf(size_t rank) const { return nullptr; }
    // 析构、禁止复制及其他现有方法声明保持不变，此处省略。
};
class DataStrategy {
public:
    Status Setup(CtrlLayout& ctrl, const DataOptions& options);
    void* HostDeviceDataAt(size_t slotIdx) const; // 新增：Host 内存的设备别名
    // 保留 HostAccessibleOf / DataAt / DeviceDataAt、backend_ 及其他现有成员。
};
}
```

接口必须满足：

- scheduler 只加入控制区，不调用 DataStrategy::Setup。保留 #1457 从 Header 获取缺失 shardSize 的修复。
- worker 校验 rank 范围、非零尺寸、与 Header 一致及大小计算溢出。沿用 `Setup → BindLocal → PublishLocal → ImportPeers → MarkRankDataReady → WaitPeersReady → FinalizeSetup`，不删除 #1462 的 ready 字段和屏障。
- Setup 成功时本地和所有 peer 段均可用；等待句柄、FD 连接/接收及 ready 屏障共用同一 deadline，0 表示不等待。backend 的阻塞操作须遵守剩余预算。
- Setup 后地址查询不做连接、懒映射或注册；未初始化、越界或不具备相应访问能力时返回 false/nullptr。
- CtrlLayout 由 Buffer 持有，生命周期长于 DataStrategy。Setup 不可重复调用；失败清理本次资源并返回错误。

| 数据 | HostAccessibleOf | DataAt | DeviceDataAt | 新增 HostDeviceDataAt |
|---|---|---|---|---|
| A5 本地段 | true | Host 地址 | nullptr | nullptr |
| A5 peer 段 | false | nullptr | 设备地址 | nullptr |
| Memfd 本地/peer 段 | true | 本进程 Host 地址 | nullptr | 传输模式需要时必须提供设备别名，否则允许 nullptr |
| 现有 POSIX SHM 段 | true | 本进程 Host 地址 | nullptr | 默认 nullptr |

保留 #1462 原有 HostAddrOf/DeviceAddrOf 互斥语义；设备别名通过新增接口独立提供。HostAccessible 表示 CPU/后端 I/O 能否访问，不能代替拷贝模式判断；Host 地址与设备别名不保证数值相同。

## 4. MemfdDataBackend 实现流程

继承 DataBackend，每个 worker 创建一个数据段，并导入同一控制区的其他 worker 数据段。复用统一 DataStrategy 编排，不另建初始化协议：

`memfd_create → ftruncate → mmap → NUMA 绑定 → first touch → 设备注册 → 发布本地段 → 导入 peer 段`

具体要求：

- 本地段大小由 slotsPerRank × slotSize 计算；页对齐填充不增加逻辑 slot。不要再次拆分 connector 已拆分的 FA/WA 容量。
- 绑定前禁止 MAP_POPULATE 或提前清零。设备注册复用 Trans::Buffer 的平台接口，必要时获取设备别名。
- BindLocal 完成本地映射、NUMA、触页、注册并启动 FD 服务，成功后才允许 ExportLocal 发布。构造函数接收 domainId，与现有 POSIX backend 一样负责域隔离。
- ExportLocal 发布校验域/rank 的 uint64_t token；可参考 POSIX 的确定性名称 hash，不能把进程 fd 当共享 handle。ImportPeer 校验 token 后，通过派生的 Unix socket 使用 **SCM_RIGHTS 接收真实 FD**，检查 rank、布局和 fstat 长度。
- socket 名由 domainId 和 rank 派生并限制长度；HAL 继续导入驱动 handle，POSIX 继续按名称 shm_open，三者不能混用。连接和接收共用剩余 deadline，超时返回错误。
- peer 收到 FD 后 mmap，并在自己的设备上下文注册；不重新绑定或清零该段。各进程虚拟地址可以不同。
- 全部 rank ready 后，FinalizeSetup 停止 FD 服务并释放 socket 名，保留映射/注册；失败或析构由幂等 Reset 停止服务、解除注册、munmap、close。accept/recv 线程必须可唤醒退出。调用方先等 I/O 和设备拷贝结束，再释放 Handle 和 Buffer。
- 保留已有 POSIX 的“全部 ready 后 shm_unlink”语义，不把它套到 HAL handle 上。控制区保持最小改动，不新增复杂状态机；检查 #1462 前后布局差异，新旧二进制不得混用同一控制域。

## 5. NUMA 自动选择接口

不提供用户 policy、节点列表或开关。DataStrategy 在 BindLocal 前调用 Resolve，结果通过 BackendOptions 交给 HAL/Memfd，peer 导入不再次选节点：

| 探测结果 | 分配方式 | 当前部署对应关系 |
|---|---|---|
| 检测到设备 NUMA 亲和性 | 整个本地数据段放在亲和节点 | A5、A2、H100 |
| 检测不到设备 NUMA 亲和性 | 选择 nodes[本机 worker rank % nodes.size()]，整个本地数据段放在该节点 | A3 |

平台对应关系用于验收，执行时以实际探测结果为准，不按型号硬编码。分配粒度是 worker，采用确定性轮转，不在段内按页或块打散。走轮转路径的 worker 若可用节点列表相同且本机 rank 连续，各节点分配到的 worker 数相差不超过 1。

```cpp
namespace UC::Cache2::Numa {
enum class Placement { Affinity, RankRoundRobin }; // 内部选择结果，不是用户配置
struct Plan {
    Placement placement;
    int32_t node;
};
Expected<Plan> Resolve(std::optional<int32_t> deviceNode,
                       std::optional<size_t> fallbackNumaRank);
Status BindBeforeTouch(void* base, size_t bytes, const Plan&); // Memfd 路径
Status Verify(void* base, size_t bytes, const Plan&); // 有界采样并记录诊断
}
```

- connector/device 层将设备 ordinal 转为真实设备亲和节点，未知传空；NUMA 模块不依赖 torch、ACL、CUDA。已有 CPU 绑核逻辑按设备编号推算的节点不能冒充探测到的亲和性。
- Resolve 获取在线、有内存且当前进程允许访问的 NUMA 节点，按节点 ID 升序排列。亲和节点有效且可用时优先选择；检测不到亲和性时使用 nodes[fallbackNumaRank % nodes.size()]。缺少 fallbackNumaRank 时明确报错，不默认填 0；选定结果在数据段生命周期内不变。
- connector 沿用内部字段 cache_detected_numa_node、cache_fallback_numa_rank。后者是本机跨 DP/PP/TP 的 worker 编号，不使用控制区 myRank 或设备 ordinal 代替。单机模型并行组沿用 rank-partition 公式：localDpRank × (PP × TP) + modelParallelRank；跨机部署应由实际本机 worker 拓扑计算。FA/WA 使用同一 worker 编号。
- 已知亲和节点不可用、没有可用节点或绑定失败时明确报错，不能伪装成探测不到或静默退回系统默认分配。记录探测结果、选择方式和最终节点；Verify 只诊断，查询失败不影响服务。
- Memfd：创建者在首次触页/注册之前按 Plan 绑定整个本地数据段。peer 只导入并注册，不重新选择节点、绑定或清零。
- A5 HAL：同样先 Resolve，在 HAL 物理内存分配时落实目标节点；核对目标 SDK 的 NUMA 分配参数后扩展封装，不假设分配后 mbind 有效，也不把 deviceId 直接当 NUMA 节点。当前封装未暴露节点参数，需要补齐；实现受 SDK 限制时明确报告，不能跳过 A5 亲和分配。

另加 `cache_data_setup_timeout_ms=600000` 传入 DataOptions；它仅控制初始化超时，与 NUMA 策略无关。

复用参考：`codex/cache-rank-partition@25b2b596` 的 `device.py::get_numa_node`、`ucm_connector.py::_configure_numa_placement`、`shm_numa_layout.h::RankNode` 及 `shm_numa.h`。沿用探测与取模算法；新入口统一处理无亲和性场景，不照搬旧 connector 仅对 NPU 非 MLA 设置 fallback rank 的限制，也不迁入旧版按段拆分多节点的分支。A5 HAL 适配和有界 Verify 仍按本文补齐。

## 6. SDMA stream 数与地址选择

将 develop 已有的 SDMA Direct 适配接入 A5 的 Cache v2；A5 v2 当前只有普通 Setup，以下是需新增的接口，不是已有接口改名：

```cpp
Status SetupSdmaDirect(int32_t deviceId, size_t streamNumber, bool useGdr);
```

继续使用 cache_stream_number，范围 1..32。内部保留“用户是否显式配置”：未配置时普通拷贝为 4、SDMA Direct 为 16；显式设置时 SDMA 使用配置值。Load 和 Dump 各自创建 N 条 stream（SDMA Direct 默认各 16 条），轮转、事件等待及同步覆盖全部 N 条。IO aggregation 保持原有单聚合 stream 行为。

地址选择：普通 Memfd 拷贝使用 DataAt；Memfd SDMA Direct 使用新增 HostDeviceDataAt；A5 保持本地 Host、peer Device 路径。requireHostDeviceAddress 从传输模式推导。能力不支持或缺少所需地址时初始化报错，不静默切换模式。

## 7. 约束与交付

- 保留 unique ID、FA/WA 容量和 namespace、MLA dump 分工。当前 myRank=deviceId%rankCount 只适用于域内取模结果唯一的部署；同域重复 rank 必须在初始化 SlotMeta 前发现，不能覆盖已有分区。
- 非 A5 切换到 v2 时核对 share_buffer_enable、cache_load_backend_only 等旧配置，不静默忽略。未覆盖的模式保留现有 cc 路径并明确选择条件；不能兼容分发时应显式报错并列出限制，不宣称已满足 develop 商用兼容性。同库保留旧/新核心时只能有一个导出的 MakeCacheStore。
- 保留 #1463 独立 Cache pipeline。通过 pipeline/Store API 独立启动时，调用方也须传入设备亲和提示或本机 worker rank；不能假设一定经过 vLLM connector，不强制要求下层 store_backend。
- 先扩展既有 backend 接口并接入 Memfd，再补 NUMA、SDMA；按这些边界组织提交，不重复拆分核心。#1462 的 POSIX 测试保留为该 backend 的测试，另加 Memfd 多进程 FD/ready/清理测试，不把 POSIX shm_unlink 断言套到 Memfd 上。
- 本地检查：平台工厂选择和 HAL 依赖隔离、Linux 多进程 FD 共享与域隔离、地址/设备别名接口、失败清理和超时、NUMA 两条路径、stream 数配置、独立 Cache pipeline。
- 远端验收：A5 原有功能不回退；A5/A2/H100 验证亲和内存，A3 验证按本机 worker rank 轮转及数据正确；DP2TP8 与 FAWA 可运行，NUMA 和 stream 数分别做 A/B。
- 交付时列出修改文件、配置样例、已运行测试、未运行测试及远端命令；没有硬件结果不要宣称验证通过。
