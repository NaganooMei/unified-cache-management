# A5 与 Develop Cache 统一：DataStrategy 和 NUMA 接口设计

日期：2026-09-28。状态：待实现的接口约定。

用户确认 A5 分支已比较稳定，开始准备向 develop 迁移。本文基于 `feature_a5@9dc1189b` 和 `develop@2d24ab39` 的代码检查，规定实现边界；不代表已经完成迁移或非 A5 硬件验证。

## 1. 目标与范围

复用 A5 connector、Cache v2 控制区和缓存算法，通过构建时 `PLATFORM` 选择数据区实现：A5 用 HAL，普通机器用 memfd/mmap。新增独立 NUMA 模块，并打通 SDMA stream 数配置。

统一的是 Buffer、Handle、CtrlLayout、查找/引用计数/owner/淘汰协议。数据区实现不参与 key 查找、填充 owner 选举或淘汰，不负责提交后端 I/O。旧 rank-partition 分支仅作为回归参考，不作为代码基线。

不把整个 feature_a5 分支直接覆盖到 develop；迁移目标文件并保留 develop 后续修复。首次交付以共享 Cache 为主，私有模式和其他既有路径在覆盖验证前保留旧实现，不静默改变旧配置含义。

## 2. PLATFORM：构建时选择，不在运行时切换

沿用已有链路：

```text
export PLATFORM=ascend-a5 / ascend-a3 / ascend / cuda
    → setup.py
    → CMake -DRUNTIME_ENVIRONMENT=...
    → 只编译一份 DataStrategy 实现
    → libcachestore.so
```

| PLATFORM | DataStrategy | 设备注册与传输 |
|---|---|---|
| ascend-a5 | HAL | 沿用 A5 HAL 分配、导入和映射，不调用 HostRegister |
| ascend-a3 | Memfd | 普通共享内存 + Ascend 注册；支持适配后的 SDMA Direct |
| ascend | Memfd | 普通共享内存 + Ascend 注册；保留普通/IO aggregation 路径 |
| cuda | Memfd | 普通共享内存 + CUDA 注册；保留已支持的 GDR 适配 |
| simu | Memfd | 真正测试 Linux memfd/FD 传递，设备注册使用已有模拟实现 |
| musa / maca | 暂保留 develop 原路径 | 不因名称属于非 A5 就宣称已支持新实现；后续适配后再加入 |

`PLATFORM` 是构建输入。安装后只修改环境变量不会切换已编译实现，必须重建安装；不同平台使用独立构建目录或清理旧构建产物。直接使用 CMake 时显式传 `-DRUNTIME_ENVIRONMENT=...`，不能假设 CMake 自动读取 shell 的 PLATFORM。

已有 setup.py 在未匹配 PLATFORM 时走 simu；本设计要求区分“未设置的 CI 默认行为”和“非空但拼写错误”，后者报错，避免生产包误编为 simu。

建议目录：

```text
ucm/store/cache/v2/
  data_strategy.h                 # 唯一公共接口，不包含 HAL/CUDA/ACL 头
  data_options.h                  # Setup 所需配置
  data/hal/data_strategy.cc        # A5 实现
  data/memfd/data_strategy.cc      # 普通内存实现
  numa/numa_policy.h
  numa/numa_policy.cc
```

CMake 显式选择源文件，不能继续用递归 glob 同时编译两份同名实现。公共 Cache 源文件与平台源文件分开列出；HAL SDK 依赖只进入 A5 构建。

```cmake
# 示意：公共源文件不包含 data/ 下的实现文件。
if(RUNTIME_ENVIRONMENT STREQUAL "ascend-a5")
    target_sources(cachestore PRIVATE v2/data/hal/data_strategy.cc)
elseif(RUNTIME_ENVIRONMENT MATCHES "^(ascend|ascend-a3|cuda|simu)$")
    target_sources(cachestore PRIVATE v2/data/memfd/data_strategy.cc
                                     v2/numa/numa_policy.cc)
endif()
```

首次迁移可让旧 Cache 继续服务尚未覆盖的模式；最终覆盖后公共核心只保留一套。DataStrategy 的选择始终由构建平台决定，不新增运行时 `data_strategy=hal/memfd` 配置。

若同一库暂时保留旧/新核心，必须只有一个导出的 `MakeCacheStore` 入口，由兼容适配层按既有模式选择内部 Store；旧/新工厂改用不同的内部符号，不能把两份同名 C 工厂直接链接。这个过渡路由不改变平台数据实现的构建选择。

## 3. DataStrategy 公共接口

保留 A5 已有的四个核心方法；把 Setup 参数收敛为 options，以传入域标识、NUMA 和设备地址要求。以下是拟实现接口，不是当前代码原样摘录。使用 C++17。

```cpp
namespace UC::Cache2 {

struct DataOptions {
    std::string domainId;           // 对应控制区 uniqueId，FA/WA 已隔离
    int32_t deviceId{-1};           // 当前进程设备 ordinal
    size_t myRank{0};               // 控制区中的本地分区编号
    size_t slotSize{0};             // 来自已验证的 CtrlLayout
    size_t slotsPerRank{0};         // 来自已验证的 CtrlLayout
    size_t setupTimeoutMs{600000};  // 整次数据区建立的总预算；0 表示不等待
    bool requireHostDeviceAddress{false}; // SDMA Direct 等路径需要 Host 的设备别名
    Numa::Options numa;             // 见第 6 节
    std::optional<int32_t> deviceNumaNode; // 已解析的设备亲和提示
};

class DataStrategy {
public:
    DataStrategy();
    ~DataStrategy();
    DataStrategy(const DataStrategy&) = delete;
    DataStrategy& operator=(const DataStrategy&) = delete;

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

两份 .cc 定义同一 DataStrategy 符号和各自的 Impl，不增加虚函数热路径。CtrlLayout 由 Buffer 持有，必须比 DataStrategy 活得更久。

### 3.1 方法契约

| 方法 | 必须满足的行为 |
|---|---|
| Setup | 仅 worker 调用。创建本地段、发布共享信息、导入所有 peer 段；全部可用后才返回成功。参数非法或布局不一致返回 InvalidParam；超时返回 Timeout；平台能力缺失返回 Unsupported |
| HostAccessibleOf | 表示此 slot 是否有本进程 CPU/后端 I/O 可用地址，不表示“本地段”，也不表示选择哪种加速拷贝 |
| DataAt | 返回 CPU 可读写地址；不可访问、未初始化或 slot 越界返回 nullptr |
| DeviceDataAt | 返回当前设备上下文可访问地址；无此映射返回 nullptr。不能假定与 Host 地址相等 |
| 析构 | 释放本进程导入、注册、映射、FD/句柄和服务线程；调用方须先结束全部 I/O 和拷贝，不允许边析构边访问 |

Setup 检查 `0 <= myRank < rankCount`、非零 slotSize/slotsPerRank、与 Header 一致、乘法/对齐溢出和域身份。调用方 Buffer 须已完成本 rank 的唯一认领和 SlotMeta 初始化。成功后禁止重复 Setup。失败后释放本次取得的资源，失败实例不在同一已发布 domain 内自动重试；由 Store 协调整组退出/重建。

Setup 成功后地址查询不得做 socket 连接、懒注册或阻塞等待，支持多线程只读访问。所有地址以“段内 slot 偏移”计算，不要求各进程虚拟地址相同，也不要求各段在普通内存实现中连续。

### 3.2 两种实现的访问能力

| 数据段 | HostAccessibleOf | DataAt | DeviceDataAt |
|---|---|---|---|
| A5 本地 HAL 段 | true | 有效 | 按当前 A5 契约为 nullptr |
| A5 peer HAL 段 | false | nullptr | 有效 |
| Memfd 本地/peer 段 | true | 有效 | requireHostDeviceAddress=true 时必须有效；否则允许 nullptr |

`requireHostDeviceAddress` 只要求 Host 可访问内存的设备别名，不要求 A5 本地段变成设备映射。A5 不支持的 Host 别名拷贝模式在 Store 配置校验时拒绝。simu 可用同一地址模拟别名，但不作为真实硬件证明。

普通内存注册复用 `Trans::Buffer::RegisterHostBuffer / UnregisterHostBuffer`，由已有平台实现提供 ACL/CUDA 差异。必须明确“注册成功但获取设备别名失败”的清理责任，避免泄漏或重复注销。

## 4. 控制区与数据区共享协议

控制区仍使用现有 CtrlStrategy 的 memfd 和 FD 传递。scheduler 最后加入，缺失 shardSize 时从 Header 读取 slotSize；不创建、映射或注册数据区。

HAL share handle 可跨进程导入；普通 memfd 的 fd 整数只在本进程有效，不能直接写入 RankDataDesc 后让其他进程拿这个整数 mmap。

规定：保留 `RankDataDesc.handle` 的 HAL 用途；新增后端种类与独立 readiness 状态，Memfd 通过派生 socket 名获取真正的 FD。新增字段需要升级控制区布局版本，不允许新旧二进制混用同一 domain。

```cpp
enum class DataBackend : uint32_t { Hal = 1, Memfd = 2 };
enum class SegmentState : uint32_t { Empty, Initializing, Ready, Failed };

// Header：在发布控制区 Ready 前填写 dataBackend。
// RankDataDesc：保留 handle，增加 atomic<uint32_t> state。
// Proposed CtrlLayout helpers:
Status ClaimRank(size_t rank); // CAS Empty -> Initializing，重复创建者报错
Status PublishRank(size_t rank, size_t handle); // Initializing -> Ready，release
void FailRank(size_t rank); // 发布 Failed，release
Expected<RankDataDesc> GetRankDesc(size_t rank) const; // acquire，按 state 判定
```

`GetRankDesc`：Empty/Initializing 返回 NotFound；Ready 返回描述符；Failed 返回明确错误。Memfd Ready 时 handle 不用于导入，可以保留 kInvalid；不得再把 handle 是否有效作为统一 readiness 判据。该变化只涉及共享生命周期，不修改 SlotMeta 算法。

Buffer 的顺序规定为 `ClaimRank → InitSlotRange → DataStrategy::Setup`，不能在认领之前重置 SlotMeta，否则重复 rank 会破坏已有分区。SlotMeta 初始化仍归 Buffer，DataStrategy 不负责控制元数据初始化；其成功发布 Ready，失败发布 Failed。

同一域参与者须校验布局版本、数据后端种类、rank 数及 slot 布局；HAL 与 Memfd 不能互相加入。domainId 继续来源于 connector 的完整 engine/store namespace，不由 DataStrategy 重新定义共享范围。

Memfd socket 名由 domainId 和 rank 稳定派生，限制在 Unix socket 名长度内。FD 服务握手回传协议版本、域标识摘要、rank、slotSize、slotsPerRank、segmentBytes；接收方核对后才采用 FD，并用 fstat 校验长度，避免错误映射。使用固定宽度编码，不直接传带 padding 的 C++ struct。

不假设 `rankCount` 个连接之后服务即可关闭：数据 FD 服务在该 DataStrategy 生命周期内持续存在，退出时停止 accept 并 join；错误请求不消耗固定参与者配额。Connect、握手和 RecvFd 都受同一个剩余 deadline 约束，不能只给循环重试设置超时。

## 5. 两份 DataStrategy 的初始化与清理

### Memfd

1. Buffer 完成本 rank 认领和 SlotMeta 初始化后，校验 options 和 Header；选定当前 deviceId。
2. 计算 segmentBytes，按页对齐并检查溢出；memfd_create、ftruncate、mmap，禁止提前 MAP_POPULATE。
3. NUMA Resolve/Bind，first touch，再按所选传输模式注册内存，必要时获取设备别名。
4. 启动本 rank FD 服务，确认服务可接收请求后 PublishRank。必须先发布本地段，再等待其他 rank，避免相互等待。
5. 对每个 peer 等待 Ready，接收 FD、校验、mmap、在当前设备上下文注册。导入者不清零、不重新绑定 NUMA。
6. 所有 peer 映射完成后 Setup 返回成功。没有无限期等待，0 timeout 允许即时尝试，但不等待 peer 就绪。

每个 rank 只创建一个物理段，但会映射和注册所有 peer 段；内存预算按唯一物理段总量计算，注册开销按进程单独评估。普通内存路径允许任意已映射段参与 Host I/O，不把其他 rank 的虚拟地址写入共享区。

失败：保留原始错误，发布 Failed（已 Claim 的 rank），停止本地 FD 服务，先解除设备注册，再 munmap 和 close FD。已经发给 peer 的 FD 不能撤销；其他 worker 检测 Failed 或在有界等待内失败，由上层停止该 domain。本轮不实现单 rank 热恢复。

### HAL

把现有 data_strategy.cc 的 HAL 部分移入 HAL 专用文件，保留大页失败后普通页重试、导出/导入和本地 Host/远端 Device 的地址语义。接入统一 options、readiness 和 deadline，不增加 HostRegister 调用。

HAL 分配粒度对 rankStride 的影响由实现内部处理，不改变逻辑 slotSize。每个 rank 对共享几何参数的理解必须一致；初始化时验证，不能因不同分配粒度静默计算不同 peer 段长度。

正常关闭由队列先等待后端写入/读取和所有 stream 完成，再释放 Handle，最后销毁 Buffer/DataStrategy。内存实现不能替代 Load/Dump 处理在途 I/O 的生命周期。

## 6. NUMA 模块接口

NUMA 只控制普通数据段的物理页放置，不影响控制区索引和缓存 key。设备到物理 NUMA 节点的探测在 connector/device 适配层完成；NUMA 模块只接收节点提示并结合 Linux allowed memory nodes 校验，不依赖 torch、ACL 或 CUDA。

```cpp
namespace UC::Cache2::Numa {
enum class Policy { Off, Auto, Bind };
struct Options {
    Policy policy{Policy::Off};
    std::vector<int32_t> nodes; // Bind 专用，按列表顺序为各 rank 分配节点
    bool verify{false};
};
struct Plan {
    Policy policy{Policy::Off};
    std::optional<int32_t> node; // 空表示保留系统默认放置
};
struct Verification {
    size_t sampledPages{0};
    size_t matchedPages{0};
    size_t unknownPages{0};
};
Expected<Plan> Resolve(const Options&, std::optional<int32_t> deviceNode, size_t myRank);
Status BindBeforeTouch(void* base, size_t bytes, const Plan&);
Expected<Verification> Verify(void* base, size_t bytes, const Plan&);
}
```

配置约定（拟新增）：

| 配置 | 默认值 | 语义 |
|---|---|---|
| cache_numa_policy | off | off / auto / bind |
| cache_numa_nodes | 不设置 | bind 时必填，段 r 绑定 nodes[r % nodes.size()]；单元素表示所有段绑同一节点 |
| cache_numa_verify | false | first touch 后做有界页采样，报告位置和不可查询页 |
| cache_detected_numa_node | connector 内部字段 | 当前设备的物理亲和节点，不用 deviceId 直接充当 NUMA node |
| cache_data_setup_timeout_ms | 600000 | 数据区初始化总 deadline，保留 A5 现有默认时长；0 不等待 |

Off 不依赖 NUMA sysfs 可用；Auto 优先有效且允许的亲和节点，缺失/权限不足时记录原因并保留系统放置，不自行猜测拓扑。Bind 要求 nodes 非空、无重复、每项在线且属于允许内存节点，失败就终止初始化。Off/Auto 配置非空 nodes 视为配置错误，避免看似生效实际被忽略。

BindBeforeTouch 只设页策略，不分配、不清零、不注册。DataStrategy 负责触页；Auto 的可恢复绑定失败由调用方显式降级并记录，Bind 失败不降级。Verify 仅诊断，不承诺所有页永远留在目标节点，也不因查询权限不足中止服务。

HAL 初期只支持 Off；明确请求 Auto/Bind 时返回 Unsupported，不声称已经绑定。simu 在 Linux 上可以实测 NUMA 模块，受权限限制的测试明确 skip。

## 7. CopyStream 与 SDMA stream 数

不能只增加 Memfd DataStrategy 就认为 SDMA Direct 已打通：A5 队列目前按 HostAccessible 选择 H2D/D2H 或 D2D；普通 memfd 全部 HostAccessible，而 SDMA Direct 需要设备别名。

规定拷贝选择顺序：先判断已选传输模式，再判断地址能力。HostAccessible 始终保留“CPU 可访问”含义。

| 传输模式/数据 | 使用地址 | 行为 |
|---|---|---|
| 普通 Memfd 拷贝 | DataAt | H2D / D2H |
| Memfd SDMA Direct | DeviceDataAt | 复用 develop SDMA Direct 适配和设备别名语义 |
| A5 本地 HAL | DataAt | 保留当前 H2D / D2H |
| A5 peer HAL | DeviceDataAt | 保留当前 D2D |

CopyStream 复用已有传输能力；SDMA 接口改为：

```cpp
Status SetupSdmaDirect(int32_t deviceId, size_t streamNumber, bool useGdr);
```

所有 N 条 stream 必须通过 MakeSdmaDirectStream 创建；轮转、事件等待和 Synchronize 覆盖实际 N 条，部分创建失败清理已创建资源。Load 与 Dump 各有自己的 N 条 stream，不是总计 N 条。

继续使用 `cache_stream_number`，配置范围 1..32。为保持 develop 的有效默认行为，内部记录是否显式设置：未设置时普通路径为 4、SDMA Direct 为 1；显式设置时 SDMA 不再强制覆盖为 1。IO aggregation 保持已有单聚合 stream 语义，不在此轮一起更改。A5 普通路径默认仍为 4。

requireHostDeviceAddress 从有效传输模式推导，不作为用户独立开关。缺少所需地址或 SDK 能力时 Setup 阶段报错，不在运行中静默改成另一种拷贝。GDR/IO aggregation 等兼容组合沿用 develop 的约束。

## 8. Connector 和兼容边界

connector 继续负责 uniqueId、FA/WA 分域、容量拆分和 MLA dump 分工；平台选择由构建决定，不再新增 Python DataStrategy 分支。

当前 A5 `myRank = deviceId % rankCount` 可以覆盖连续编号的 DP2TP8。初期保持该实现，但同一 domain 必须保证 rank 唯一；数据段 ClaimRank 对重复创建者报错，不允许覆盖活跃分区。每进程只可见一张卡导致 deviceId 都为 0 的部署，需要单独的拓扑适配才能启用新路径。

控制区总容量保持现有公式：`slotsPerRank = floor(floor(capacity / slotSize) / rankCount)`。rank 间等分剩余尾部不使用，页/HAL 对齐填充不增加可分配 slot。FA/WA 的容量在 connector 拆分一次，DataStrategy 不再拆分。

私有模式、跨节点 TP 和未覆盖的传输配置在适配完成前保持 develop 旧路径；共享 memfd/HAL domain 只支持同节点参与者。缓存核心统一不等于强制所有旧部署加入共享内存域。

新增接口不承诺 Store API 已全部对齐：批量 Lookup、cache_load_backend_only 等差异必须列入兼容清单。#1458 改脚本为 prefix lookup 只解决带宽脚本调用兼容，不代表补齐这些能力。不得静默忽略用户显式要求的 backend-only 语义。

## 9. 实现拆分与验收

1. 迁移稳定 A5 核心及 connector 必需变化；保留 #1457 scheduler 读 Header 的修复，对齐 develop 独有修复。
2. 整理公共 DataStrategy 接口与 CMake 平台选择；HAL 行为保持一致，增加 Memfd 数据区和生命周期协议。
3. 实现 NUMA 模块及 connector 拓扑提示。
4. 接入普通/SDMA 等已有 CopyStream 路径，支持配置 stream 数。
5. 完成商用兼容矩阵后扩大新核心覆盖，移除不再需要的旧实现。

| 验证项 | 验收要求 |
|---|---|
| 构建隔离 | HAL 与 Memfd 实现不重复链接；CUDA/simu 不包含 HAL SDK 头或库；切换 PLATFORM 确实改变编译实现 |
| 主机多进程 | 真正 SCM_RIGHTS 传 FD；各进程不同虚拟地址仍能读到同一内容；域隔离、重复 rank、错误 fd 长度和 deadline 可验证 |
| 地址接口 | 本地/peer/越界/未初始化行为与表一致；Setup 后查询不触发懒映射；注册失败完整回收 |
| 初始化失败 | 初始化中退出、peer Failed、缺少 peer 和阻塞握手都能有界结束；同域后端/版本不匹配被拒绝 |
| NUMA | 绑定早于首次触页；Auto 降级、Bind 失败、受限 cpuset 和页采样有明确结果 |
| SDMA | N=1、4、16 的创建、轮转和全部同步；非法 N 拒绝；缺省值保持旧行为 |
| 硬件回归 | A5 HAL、A3 注册+SDMA、CUDA 注册/拷贝；DP2TP8、FAWA、已支持模型/模式及冷/热命中数据正确 |

本地只做可用的主机测试和静态检查；硬件数据正确性及带宽、TTFT、启动耗时、注册开销在远端 GPU/NPU 验证。NUMA 与 stream 数分别做固定其他变量的 A/B。

## 10. 代码依据

- [A5 DataStrategy 当前接口](https://github.com/ModelEngine-Group/unified-cache-management/blob/9dc1189b/ucm/store/cache/v2/data_strategy.h)
- [A5 HAL 数据区实现](https://github.com/ModelEngine-Group/unified-cache-management/blob/9dc1189b/ucm/store/cache/v2/data_strategy.cc)
- [PLATFORM 到 CMake 的现有映射](https://github.com/ModelEngine-Group/unified-cache-management/blob/9dc1189b/setup.py)
- [当前 Cache 平台构建分支](https://github.com/ModelEngine-Group/unified-cache-management/blob/9dc1189b/ucm/store/cache/CMakeLists.txt)
- [develop stream 数与配置](https://github.com/ModelEngine-Group/unified-cache-management/blob/2d24ab39/ucm/store/cache/cc/global_config.h)
- [迁移方案摘要](cache-unification-develop-proposal.md)
