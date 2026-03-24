# gdr_copy 项目入门指南

这份文档面向刚接触 C++、CUDA 和 RDMA 的读者。

目标只有一个：让你先从宏观上看懂这个项目在做什么、为什么这样做、代码大概怎么组织，以及你该怎么用它。

---

## 1. 这个项目是干什么的

`gdr_copy` 是一个小型 C++ 库，目标是把部分 `cudaMemcpy` 替换成基于 GPUDirect RDMA 的数据拷贝。

更具体一点，它想优化下面两种常见的数据传输：

- `H2D`：Host to Device，CPU 内存拷到 GPU 内存
- `D2H`：Device to Host，GPU 内存拷回 CPU 内存

普通写法通常是：

```cpp
cudaMemcpy(d_ptr, h_ptr, bytes, cudaMemcpyHostToDevice);
cudaMemcpy(h_ptr, d_ptr, bytes, cudaMemcpyDeviceToHost);
```

这个项目想把它改成：

```cpp
auto ch = GDRCopyLib::open(0, "mlx5_0");

ch->memcpy(d_ptr, h_ptr, bytes, GDR_H2D);
ch->memcpy(h_ptr, d_ptr, bytes, GDR_D2H);
```

它不是为了替代所有 CUDA 拷贝，而是想在特定硬件条件下，把 H2D / D2H 的路径做得更轻、更快。

---

## 2. 它为什么要这么做

先记住一句最重要的话：

普通 `cudaMemcpy` 的主角是 CUDA 驱动。  
这个项目的主角是 RDMA 网卡。

### 2.1 普通 `cudaMemcpy` 的思路

你在用户代码里调用 `cudaMemcpy` 后，后面很多事情会由 CUDA Runtime 和驱动去安排：

- 检查参数
- 处理地址和设备上下文
- 可能涉及页锁定、DMA 调度、同步
- 最终把数据搬运到 GPU 或从 GPU 搬回来

你不用手动管这些，所以它好用、通用、稳定。

但它的软件路径相对更“厚”，在小数据量时，真正慢的有时不是“搬数据”本身，而是“调度这次搬运”的开销。

### 2.2 这个项目的思路

这个项目想做的是：

- 先把 RDMA 通道、内存注册、资源对象都准备好
- 后面每次传输时，由用户态程序直接给 RDMA 网卡下发“搬运任务”
- 真正搬数据的是 NIC 的 DMA 引擎

所以它的优势不是“世界上没有驱动了”，而是：

- 驱动和内核仍然存在
- 但很多重操作提前做完
- 后续单次传输时，CPU 和软件栈的参与更少

这也是 README 里一直强调“适合小 IO”的原因。

---

## 3. “没有用户态和内核态切换”这句话该怎么理解

这个说法不能按字面理解成“完全没有内核”。

更准确的理解是：

- 不是完全没有内核态参与
- 而是尽量避免每次拷贝都走一遍较重的内核控制路径

你可以把整个过程分成两部分。

### 3.1 准备阶段

这一步仍然依赖驱动和内核：

- 打开 CUDA 设备
- 打开 RDMA 设备
- 创建 QP、CQ、PD 等 RDMA 资源
- 注册内存
- 检查 GPU memory 能不能被 RDMA 注册

这些动作一般不会在每次传输都重复发生。

### 3.2 传输阶段

这时软件做的事情就少很多：

- 用户态代码提交一个 RDMA 请求
- NIC 根据已经准备好的资源去发起 DMA
- 数据直接在硬件之间流动

所以它不是“没有内核”，而是“把内核和驱动的重参与尽量前置，把每次拷贝的数据面做薄”。

---

## 4. 这个项目的核心前提

它不是一个“装上 CUDA 就一定能跑”的通用库，而是一个对硬件环境要求很强的项目。

至少需要这些条件：

- NVIDIA GPU
- 支持 RDMA 的 NIC，比如 ConnectX 系列
- CUDA Toolkit
- `libibverbs`
- `nvidia-peermem` 或同类模块
- GPU 和 NIC 拓扑尽量靠近，最好在同一个 PCIe switch 下

如果条件不满足，这个项目不会神奇地继续走 RDMA，而是会退回到 `cudaMemcpy`。

这件事在代码里是明确做了的。

---

## 5. 宏观理解一次数据传输

### 5.1 H2D：Host -> GPU

普通理解：

```text
用户内存 -> cudaMemcpy -> GPU
```

这个项目的实际思路更像：

```text
用户内存
  -> 先 memcpy 到一块 pinned host buffer
  -> RDMA NIC 发起 RDMA WRITE
  -> GPU memory
```

也就是：

- 先拷到固定页的主机缓冲区
- 再由网卡 DMA 到 GPU

### 5.2 D2H：GPU -> Host

普通理解：

```text
GPU -> cudaMemcpy -> 用户内存
```

这个项目的思路是：

```text
GPU memory
  -> RDMA NIC 发起 RDMA READ
  -> pinned host buffer
  -> 再 memcpy 到用户目标地址
```

也就是说，这里通常会经过一个中转站，而不是直接随便写到任意 host 指针。

### 5.3 为什么要 pinned memory

因为 DMA / RDMA 需要稳定的物理页。

普通可分页内存对硬件来说不够稳定，传输时页面可能被操作系统换出或重映射。  
所以项目里先准备了 pinned host buffer，保证地址可被网卡安全访问。

---

## 6. 项目的整体代码结构

这个仓库规模不大，结构很集中：

```text
gdr/
├─ CMakeLists.txt
├─ README.md
├─ include/
│  ├─ gdr_copy.h
│  ├─ mr_cache.h
│  └─ pinned_pool.h
├─ src/
│  ├─ gdr_copy.cpp
│  └─ demo.cpp
├─ bench/
│  └─ bench.cpp
└─ scripts/
   ├─ build.sh
   └─ install_deps.sh
```

可以把它理解成 5 层。

### 6.1 API 层

文件：`include/gdr_copy.h`

这是对外接口，用户最需要关心的头文件。

主要包含：

- `GDRCopyKind`
- `GDRStats`
- `GDRCopyChannel`
- `GDRCopyLib`

如果你只是“调用这个库”，大部分时候先读这个头文件就够了。

### 6.2 MR 缓存层

文件：`include/mr_cache.h`

`MR` 是 memory region，也就是 RDMA 注册过的一段内存。

GPU 内存注册很贵，所以这里做了一个 LRU 缓存：

- key 是地址和长度
- value 是 `ibv_mr*`
- 命中就复用
- 满了就淘汰最久没用的

作用很简单：避免每次传输都重新注册 GPU 内存。

### 6.3 Pinned buffer 池

文件：`include/pinned_pool.h`

这里维护了一组预先分配好的 pinned host buffer：

- 默认 8 个 slot
- 每个 slot 4 MiB
- 总共约 32 MiB

作用：

- H2D 时作为 RDMA WRITE 的本地源缓冲区
- D2H 时作为 RDMA READ 的本地目标缓冲区

这是一种典型的“用空间换时间”的做法。

### 6.4 RDMA 实现层

文件：`src/gdr_copy.cpp`

这是整个项目的核心实现。

这里做了几件大事：

- 初始化 CUDA 设备
- 打开 RDMA 设备
- 创建 `PD`、`CQ`、`QP`
- 把 QP 建成 loopback RC QP
- 初始化 pinned buffer 池
- 尝试注册一小块 GPU memory，判断 GPUDirect RDMA 是否可用
- 根据拷贝方向执行 `H2D`、`D2H` 或 fallback

如果只看这一份文件，你会发现它本质上就是“资源初始化 + 三条拷贝路径 + 资源回收”。

### 6.5 示例与基准层

- `src/demo.cpp`：最小可运行示例
- `bench/bench.cpp`：性能对比程序

这两个文件很适合入门时阅读：

- `demo.cpp` 帮你理解“怎么调用”
- `bench.cpp` 帮你理解“项目想优化的到底是什么”

---

## 7. 代码里的关键对象，分别是什么

第一次看 RDMA 代码时，最容易被各种缩写吓到。这里先只记最重要的几个：

### 7.1 `GDRCopyLib`

这是库的入口，像一个工厂。

职责：

- 根据 `(gpu_id, nic_name)` 打开通道
- 缓存已经打开的通道
- 统一关闭所有通道

你可以把它理解成“库级别的管理者”。

### 7.2 `GDRCopyChannel`

这是用户真正拿来做传输的对象。

职责：

- 执行 `memcpy`
- 返回统计信息
- 暴露所属 GPU 和 NIC

你可以把它理解成“一条已经搭好的传输通道”。

### 7.3 `PinnedPool`

这是 pinned host buffer 的资源池。

职责：

- 提前申请好几块固定页主机内存
- 每次传输时拿一个 slot 出来用
- 用完再放回池里

### 7.4 `MRCache`

这是 GPU memory registration 的缓存。

职责：

- 记录已经注册过的 GPU 地址范围
- 避免重复 `ibv_reg_mr`

### 7.5 `QP / CQ / PD`

这三个是 RDMA 的基础资源：

- `PD`：protection domain，资源归属域
- `QP`：queue pair，提交 RDMA 请求的队列对象
- `CQ`：completion queue，收完成事件

对初学者来说，不必一开始就死记定义。  
你可以先把它们理解成：

- `QP` 负责“发任务”
- `CQ` 负责“等完成”
- `PD` 负责“管理一组可互相配合的资源”

---

## 8. 一次 `open()` 大概做了什么

当你调用：

```cpp
auto ch = GDRCopyLib::open(0, "mlx5_0");
```

内部大概按这个顺序做事：

1. 设置 CUDA 设备
2. 找到对应 RDMA 网卡
3. 打开 RDMA device context
4. 分配 PD
5. 创建 CQ
6. 创建 RC 类型的 QP
7. 把 QP 连接成 loopback
8. 初始化 pinned host buffer 池
9. 申请一小块 GPU memory，尝试 `ibv_reg_mr`
10. 如果成功，说明 GPUDirect RDMA 可用；否则标记为 fallback

所以 `open()` 不是轻量函数，它是在“建通道”。

---

## 9. 一次 `memcpy()` 大概做了什么

### 9.1 H2D

当你调用：

```cpp
ch->memcpy(d_ptr, h_ptr, bytes, GDR_H2D);
```

大概流程是：

1. 查找或注册目标 GPU 地址对应的 MR
2. 从 `PinnedPool` 拿一个 slot
3. 把用户源数据先 `memcpy` 到 slot
4. 发一个 `RDMA_WRITE`
5. 轮询 CQ，等完成
6. 释放 slot
7. 如果数据太大，就分块重复

### 9.2 D2H

当你调用：

```cpp
ch->memcpy(h_ptr, d_ptr, bytes, GDR_D2H);
```

大概流程是：

1. 查找或注册源 GPU 地址对应的 MR
2. 从 `PinnedPool` 拿一个 slot
3. 发一个 `RDMA_READ`，把 GPU 数据拉到 slot
4. 轮询 CQ，等完成
5. 再 `memcpy` 到用户目标地址
6. 释放 slot
7. 如果数据太大，就分块重复

### 9.3 D2D

这个项目没有自己实现 GPU 到 GPU 的 RDMA 路径。  
`D2D` 直接回退到普通 `cudaMemcpyDeviceToDevice`。

---

## 10. 为什么代码里会有 fallback

这是这个项目里很重要的一点。

作者并没有假设“只要你调用了这个库，就一定成功走 RDMA”。  
相反，代码专门考虑了现实情况：

- 可能没装 `nvidia-peermem`
- 可能 GPU memory 注册失败
- 可能硬件拓扑不合适

这时候就退回到 `cudaMemcpy`。

所以这个库的实际行为是：

- 能走 RDMA 就走 RDMA
- 走不了就尽量保持功能正确

从工程角度看，这比“直接崩掉”更友好。

---

## 11. 你应该怎么阅读这个项目

如果你是初学者，我建议按这个顺序看：

### 第一步：先看 `src/demo.cpp`

目的：

- 先知道用户怎么调用这个库
- 不要一上来就掉进 RDMA 细节里

重点看这几件事：

- 如何 `open`
- 如何调用 `memcpy`
- 如何看 `stats`
- 如何 `shutdown`

### 第二步：看 `include/gdr_copy.h`

目的：

- 理解公开 API
- 建立“外部视角”

### 第三步：看 `src/gdr_copy.cpp`

这时重点别放在每个细节 API 上，而是先抓住主线：

- 构造函数做初始化
- `memcpy()` 分派到 `do_h2d` / `do_d2h` / `do_d2d`
- `rdma_write` / `rdma_read` 是真正提交 RDMA 请求的地方
- `poll_cq` 是等待完成的地方

### 第四步：再看 `mr_cache.h` 和 `pinned_pool.h`

这时你会发现，它们不是额外复杂度，而是在给主流程做性能支撑。

---

## 12. 如何构建和运行

项目里已经给了脚本。

### 12.1 安装依赖

```bash
sudo bash scripts/install_deps.sh
```

### 12.2 编译

```bash
bash scripts/build.sh
```

### 12.3 运行 demo

```bash
sudo ./build/demo 0 mlx5_0
```

含义：

- `0`：GPU 编号
- `mlx5_0`：RDMA 网卡名字

### 12.4 运行 benchmark

```bash
sudo ./build/bench 0 mlx5_0
```

它会对不同大小的数据做 H2D / D2H 测试，并与普通 `cudaMemcpy` 对比。

---

## 13. 最小使用方法

如果你只是想接入这个库，可以先记住最小模板：

```cpp
#include "gdr_copy.h"

auto ch = GDRCopyLib::open(0, "mlx5_0");

ch->memcpy(d_ptr, h_ptr, bytes, GDR_H2D);
ch->memcpy(h_ptr, d_ptr, bytes, GDR_D2H);

GDRStats s = ch->stats();

GDRCopyLib::shutdown();
```

你可以把它理解成“先建一条通道，然后反复用这条通道传数据”。

---

## 14. 看懂统计信息意味着什么

`GDRStats` 里比较关键的是这几个字段：

- `last_latency_us`
- `avg_latency_us`
- `total_ops`
- `rdma_ops`
- `fallback_ops`

最值得关注的是：

- 如果 `rdma_ops > 0`，说明至少有操作走到了 RDMA 路径
- 如果 `fallback_ops > 0` 并且 `rdma_ops == 0`，通常说明 RDMA 路径没真正启用

也就是说，判断“我到底有没有吃到这个项目的优化”，最直接的方法就是看统计信息。

---

## 15. 这个项目的优点和局限

### 15.1 优点

- 目标清晰，项目边界明确
- API 简单，接近 `cudaMemcpy`
- 有 demo 和 benchmark，容易上手
- 有 fallback 机制，不至于一失败就完全不能用
- 代码结构集中，适合学习一条完整的数据路径

### 15.2 局限

- 对硬件和驱动环境要求高
- 不是通用方案，只适合特定拓扑
- 异步接口目前只是占位实现
- 更像原型或 PoC，不是完全生产化的库
- 如果读者不懂 CUDA / RDMA / 内存注册，第一次看会觉得术语很多

---

## 16. 对初学者最重要的三个结论

如果你现在只想先记住最核心的内容，记这三条就够了。

### 结论 1

这个项目不是在“取消 CUDA”，而是在“替换一部分 `cudaMemcpy` 的数据路径”。

### 结论 2

它不是“完全没有内核态”，而是“把很多重操作前置，然后尽量让后续传输更多由 RDMA 硬件完成”。

### 结论 3

它的主线非常简单：

- 建好 RDMA 通道
- 准备 pinned host buffer
- 注册 GPU memory
- H2D 用 `RDMA_WRITE`
- D2H 用 `RDMA_READ`
- 不满足条件就 fallback 到 `cudaMemcpy`

---

## 17. 下一步你最适合看什么

如果你准备继续学这个项目，建议按下面顺序深入：

1. 先把 `src/demo.cpp` 逐行看懂
2. 再把 `include/gdr_copy.h` 的 API 看懂
3. 然后只盯着 `src/gdr_copy.cpp` 里的这几个函数：
   `GDRCopyChannelImpl` 构造函数、`memcpy()`、`do_h2d()`、`do_d2h()`
4. 最后再去理解 `MRCache` 和 `PinnedPool`

如果你按这个顺序走，理解负担会小很多。

---

## 18. 一句话总结

`gdr_copy` 是一个面向特定硬件环境的 C++ 实验型库，它尝试把 GPU 和主机之间的数据搬运从 “CUDA 驱动主导” 改成 “RDMA 网卡主导”，从而在 H2D/D2H 场景下减少单次传输的软件开销。
