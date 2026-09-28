# Cache 统一实现任务书（供编码 Agent 使用）

更新：2026-09-28。本文是实现要求，不是已完成的代码说明。

## 1. 任务与基线

直接在现有 A5 开发分支 `lzx/feature-a5` 上实现，开始前同步最新 `upstream/feature_a5`。保留已有 connector 和 `ucm/store/cache/v2/`，不另建 develop 功能分支，也不先搬迁核心。本文核对过的 A5 基线为 `9dc1189b`；同步时保留本地改动，禁止整目录覆盖丢失修复。

develop 只用于核对兼容性和参考已有传输实现（已核对 `2d24ab39`）。本轮完成 A5 上的功能扩展和验证，后续再单独处理合入 develop。

只实现四项：

1. 在现有 Cache v2 中抽出公共 DataStrategy 接口和 HAL 实现，保留 connector、控制区、Buffer、Handle 和缓存算法。
2. 增加普通内存 DataStrategy，通过构建时 PLATFORM 与 A5 HAL 实现隔离。
3. 增加独立 NUMA 模块，HAL/Memfd DataStrategy 共用：检测到设备亲和性就使用亲和内存，否则按本机 worker rank 在可用节点间轮转分配。
4. 支持配置 SDMA Direct stream 数。

不整体迁移 codex/cache-rank-partition；只复用其 NUMA 探测、按 rank 选节点及绑定逻辑。不重写缓存算法，不附带增加预取、线程池或 Posix 优化。

## 2. 文件与构建

以下路径相对仓库根目录；新增文件名按本表执行：

| 文件 | 修改内容 |
|---|---|
| `setup.py`、`ucm/store/cache/CMakeLists.txt` | 沿用 PLATFORM → RUNTIME_ENVIRONMENT；显式选择平台源文件 |
| `ucm/store/cache/v2/data_strategy.h` | 公共接口，不包含 HAL/ACL/CUDA 头；内部实现用 Impl 隐藏 |
| `ucm/store/cache/v2/data/hal/data_strategy.cc` | 移入当前 A5 HAL 实现，接入统一 NUMA 选择结果 |
| `ucm/store/cache/v2/data/memfd/data_strategy.cc` | 新增普通内存实现 |
| `ucm/store/cache/v2/numa/numa_policy.h/.cc` | 新增自动 NUMA 选择与绑定模块，不提供用户 policy 开关 |
| `ucm/shared/trans/ascend/hal/hal_memory.h/.cc` | 将 NUMA 选择结果传给 HAL 分配路径，具体参数核对目标 SDK |
| `ucm/store/cache/v2/global_config.h`、`cache_buffer.h` | 解析新配置，组装 DataOptions，调用数据区初始化 |
| `ucm/store/cache/v2/copy_stream.h`、Load/Dump 队列 | 接入所需传输模式、地址选择及 stream 数 |
| `ucm/integration/vllm/` | 保留已有 unique ID、MLA dump 分工；透传设备 NUMA 探测结果和本机 worker rank |

平台选择规则：

- `PLATFORM=ascend-a5`：编译 HAL DataStrategy。
- `PLATFORM=ascend/ascend-a3/cuda/simu`：编译 Memfd DataStrategy；simu 使用现有模拟注册接口。
- musa/maca：保留 A5 分支内已有的 `ucm/store/cache/cc/` 路径，不宣称支持新实现。

两份实现提供同一 DataStrategy 接口；CMake 只能编译其中一份，不能递归 glob 同时选中。普通平台不得引入 HAL SDK 依赖。修改 PLATFORM 后必须重新构建，运行时不读取它切换实现。直接使用 CMake 则显式指定 RUNTIME_ENVIRONMENT。非空且非法的 PLATFORM 应报错。

## 3. DataStrategy 接口

保留 A5 的地址访问方法，Setup 改用 options。以下为 C++17 接口约定，需补齐相应头文件；类禁止复制，析构负责资源清理。

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
class DataStrategy {
public:
    // 构造、析构、禁止复制声明沿用现有类。
    Status Setup(CtrlLayout& ctrl, const DataOptions& options);
    bool HostAccessibleOf(size_t slotIdx) const;
    void* DataAt(size_t slotIdx) const;
    void* DeviceDataAt(size_t slotIdx) const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
```

接口必须满足：

- scheduler 只加入控制区，不调用 DataStrategy::Setup。保留 #1457 从 Header 获取缺失 shardSize 的修复。
- worker 校验 rank 范围、非零尺寸、与 Header 一致及大小计算溢出。Setup 成功时，本地和所有 peer 段均已可用。
- Setup 后地址查询不做连接、懒映射或注册；未初始化、越界或不具备相应访问能力时返回 false/nullptr。
- CtrlLayout 由 Buffer 持有，生命周期长于 DataStrategy。Setup 不可重复调用；失败清理本次资源并返回错误。

| 数据 | HostAccessibleOf | DataAt | DeviceDataAt |
|---|---|---|---|
| A5 本地段 | true | Host 地址 | 保持现有 nullptr |
| A5 peer 段 | false | nullptr | 设备地址 |
| Memfd 本地/peer 段 | true | 本进程 Host 地址 | requireHostDeviceAddress=true 时必须提供设备别名，否则允许 nullptr |

HostAccessible 表示 CPU/后端 I/O 是否可访问，不能拿它代替拷贝模式判断。Host 地址与设备别名不保证数值相同。

## 4. 普通内存实现流程

每个 worker 创建一个数据段，并导入同一控制区的其他 worker 数据段：

`memfd_create → ftruncate → mmap → NUMA 绑定 → first touch → 设备注册 → 发布本地段 → 导入 peer 段`

具体要求：

- 本地段大小由 slotsPerRank × slotSize 计算；页对齐填充不增加逻辑 slot。不要再次拆分 connector 已拆分的 FA/WA 容量。
- 绑定前禁止 MAP_POPULATE 或提前清零。设备注册复用 Trans::Buffer 的平台接口，必要时获取设备别名。
- **FD 必须通过 Unix socket 的 SCM_RIGHTS 传递**。不能把 fd 整数写进共享 Header 后让其他进程直接 mmap。
- socket 名由 domainId 和 rank 派生，并处理长度限制；接收方检查 rank、布局和 fstat 得到的文件长度。HAL handle 继续使用现有 HAL 导入方式。
- 启动本地 FD 服务后再等待 peer，避免循环等待。连接、接收和等待 peer 共用剩余 deadline，超时返回错误。
- peer 收到 FD 后 mmap，并在自己的设备上下文注册；不重新绑定或清零该段。各进程虚拟地址可以不同。
- 失败/退出时停止 FD 服务，解除注册，再 munmap、close。调用方先等后端 I/O 和设备拷贝结束，再释放 Handle 和 Buffer。
- 保持控制区改动最小；本轮不要求新增复杂的 rank 生命周期状态机或热恢复机制。若必须修改共享布局，应升级版本并拒绝新旧布局混用。

## 5. NUMA 自动选择接口

不提供用户 policy、节点列表或开关。每个 worker 在创建本地数据段之前自动选择一次，HAL/Memfd 共用同一逻辑：

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

地址选择：普通 Memfd 拷贝使用 DataAt；Memfd SDMA Direct 使用 DeviceDataAt；A5 保持本地 Host、peer Device 路径。requireHostDeviceAddress 从传输模式推导。能力不支持或缺少所需地址时初始化报错，不静默切换模式。

## 7. 约束与交付

- 保留 unique ID、FA/WA 容量和 namespace、MLA dump 分工。当前 myRank=deviceId%rankCount 只适用于域内取模结果唯一的部署；同域重复 rank 必须在初始化 SlotMeta 前发现，不能覆盖已有分区。
- 非 A5 切换到 v2 时核对 share_buffer_enable、cache_load_backend_only 等旧配置，不静默忽略。未覆盖的模式保留现有 cc 路径并明确选择条件；不能兼容分发时应显式报错并列出限制，不宣称已满足 develop 商用兼容性。同库保留旧/新核心时只能有一个导出的 MakeCacheStore。
- 先在现有 v2 中隔离 HAL/公共接口，再补 Memfd、NUMA、SDMA；按这些边界组织可独立审查的提交，不重复迁移 connector 或核心。
- 本地检查：平台源文件互斥、Linux 多进程 FD 共享与域隔离、地址接口、失败清理和超时、NUMA 有/无亲和性的两条路径、stream 数配置。
- 远端验收：A5 原有功能不回退；A5/A2/H100 验证亲和内存，A3 验证按本机 worker rank 轮转及数据正确；DP2TP8 与 FAWA 可运行，NUMA 和 stream 数分别做 A/B。
- 交付时列出修改文件、配置样例、已运行测试、未运行测试及远端命令；没有硬件结果不要宣称验证通过。
