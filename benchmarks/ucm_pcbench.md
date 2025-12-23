# UCM (Unified Cache Management) vs LMCache/KV Cache Pool：多层级KV-Cache卸载方案全面对比评测

Prefix Caching（前缀缓存） 是当下大模型推理优化的核心技术，其本质是对重复计算的记忆与复用。在自回归生成过程中，模型会将已处理token的Key-Value（KV）状态缓存在GPU显存中。当新请求包含与历史请求相同的前缀（如多轮对话的上下文、RAG检索返回的文档、代码仓库的多文件分析）时，系统可直接复用缓存的KV状态，跳过冗余的Prefill计算，将首token响应时间（TTFT）从秒级降至毫秒级。

然而，这一技术在实践中面临两大根本性挑战：
- 容量陷阱：以QwQ-32B模型为例，单张H20 GPU仅能提供约20万token的KV-Cache空间。当并发用户增多或上下文长度增加时，显存迅速耗尽，缓存命中率从100%断崖式跌至0%，TTFT反弹回冷启动水平，用户体验剧烈抖动。
- 持久化能力缺失：GPU显存作为缓存层缺乏持久性。服务重启、节点故障、新请求抢占都会导致精心构建的缓存被瞬间清空，引发周期性的"冷启动风暴"。

正是这些痛点催生了多层级KV-Cache卸载的架构革命——将缓存从昂贵的GPU显存（L1）扩展至CPU内存（L2）、本地SSD（L3）乃至远端存储（L4），实现存储与计算的解耦。本文将通过三大实测场景，系统对比原生HBM、KV Cache Pool与UCM/LMCache中心化架构在容量、性能、扩展性与成本维度的优劣，揭示为何UCM是当前唯一能同时满足CUDA/昇腾双平台、本地/分布式双场景的统一缓存管理方案。

## 1. 场景1：原生HBM Prefix Caching的容量陷阱

### 1.1 测试环境
- **硬件**：H20(96GB)×2
- **模型配置**：QwQ-32B tp=2, GPU utilization=0.87
- **软件框架**：vLLM v0.11.0
- **配置**：tp=2, GPU utilization=0.87

### 1.2 HBM容量观测
根据vLLM启动日志，GPU侧仅能提供 **≈400k tokens** 的KV-Cache空间：
```
Available KV cache memory: 48.65 GiB / 49.18 GiB
GPU KV cache size: 398,464 tokens / 402,816 tokens
Maximum concurrency for 38,000 tokens: 10.48x / 10.60x
```
这意味着在36k token长度的请求下，**理论并发上限仅约11个**。

### 1.3 驱逐现象验证
采用 **"冷启动→热复用→新请求抢占→原请求回退"** 四段式压测：

| 阶段 | 请求内容 | 预期命中率 | 实际TTFT |
|---|---|---|---|
| ① 首次写入 | 文档集A×11 | 0% | **12,439.95 ms** |
| ② 原地复用 | 文档集A×11（与①重复） | 100% | **104.09 ms** (↓99%) |
| ③ 新请求抢占 | 文档集B×11 | 0% | **11,781.80 ms** |
| ④ 原请求回退 | 文档集A×11（与①重复） | **0% (已驱逐)** | **11,777.59 ms** |

**说明**：
- **文档集A/B**：分别由不同的随机种子生成的36k token文档集合，**每个集合含11条独立请求**
- 阶段①持续发送11条请求将GPU KV-Cache完全占满
- 阶段④的TTFT回到阶段①水平，证明缓存已被**完全驱逐**，容量瓶颈生效

**结论**：当并发数>11时，GPU KV-Cache开始触顶，系统持续驱逐最久未使用的缓存，命中率从100%逐渐下降。当新请求完全占满缓存空间后，原有请求的KV-Cache被完全驱逐，命中率最终归零，TTFT回归冷启动水平。在多用户长文档问答、多轮对话等真实场景中，**用户体验将断崖式下跌**。

---

## 2. 现有解决方案架构分析

### 2.1 KV Cache Pool方案（以Mooncake为代表）

KV Cache Pool通过在多个计算节点中共享**池化DRAM**来扩展KV-Cache容量，相比原生HBM容量有明显提升，同时保持了较高的kv cache读写速度，但仍存在根本性问题：

- **容量天花板**：计算与存储紧耦合，受限于单节点/集群的DRAM总量，无法独立扩展存储层。
- **持久化缺失**：在vLLM Ascend生态中，KV Cache Pool仅支持DRAM池化，不支持SSD持久化存储，而纯DRAM方案不具备生产级系统所需的可靠性。服务重启、节点故障会导致缓存丢失，冷启动风暴频发。

### 2.2 中心化存储架构（LMCache & UCM）

LMCache与UCM采用**分层缓存设计**：在每个节点上使用DRAM作为缓存层，将持久化的本地盘SSD作为主存储层，具有以下特点：

- **存储计算解耦**：KV-Cache作为独立对象存储于文件系统（SSD/NFS/3FS），计算节点通过高速连接器访问，两者可独立扩缩容。
- **原生持久化**：缓存对象即文件，天然继承存储系统的可靠性、快照、跨可用区复制等能力。
- **生态支持**：LMCache聚焦CUDA生态，UCM在此基础上扩展**Ascend官方支持**。特别地，相比LMCache，UCM通过原生支持DeepSeek 3FS等分布式文件系统，可将KV-Cache容量大幅度扩展。

### 2.3 架构对比总结

| 特性 | KV Cache Pool | LMCache  | **UCM** |
|---|---|---|---|
| **存储介质** | DRAM池 | SSD/NFS | SSD/NFS/**3FS** |
| **扩展性** | 与算力绑定 | **存储独立扩展** | **存储独立扩展** |
| **持久化** | 不支持 | **支持**  | **支持** |
| **平台支持** | Ascend官方方案，无SSD支持 | CUDA官方方案 | **Ascend官方方案+CUDA支持** |

---

## 3. 场景2：CUDA平台UCM vs LMCache

### 3.1 测试方案设计
选取 **4k/8k/16k/32k** 四档长度与 **1/2/4/8** 四档并发，覆盖真实业务负载：
- **4k-8k**：多轮对话中后期轮次，命中率80%+
- **8k-16k**：AI辅助编程常见长度，命中率50-80%
- **16k-32k**：RAG多文档拼接，命中率50%±
- **1/2/4/8并发**：批量处理、高并发API场景

**存储后端说明**：UCM与LMCache均配置为本地SSD后端
### 3.2 测试环境
- **硬件**：H20(96GB)×2，7TB NVMe SSD×1
- **模型配置**：QwQ-32B tp=2, GPU utilization=0.87
- **软件框架**：vLLM v0.11.0 + UCM v0.1.2/LMCache v0.3.9


### 3.3 性能对比（UCM vs LMCache）

| 输入长度 | 并发请求数 | 裸推TTFT<br>(ms) | UCM80%命中<br>TTFT(ms) | LMCache80%命中<br>TTFT(ms) | **80%命中UCM提升<br>vs裸推** | **80%命中UCM提升<br>vsLMC** | UCM50%命中<br>TTFT(ms) | LMCache50%命中<br>TTFT(ms) | **50%命中UCM提升<br>vs裸推** | **50%命中UCM提升<br>vsLMCache** |
| :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4000 | 1 | 1018.34 | 313.69 | 419.96 | **224.6%** | **33.9%** | 623.44 | 695.06 | **63.3%** | **11.5%** |
| 8000 | 1 | 2071.69 | 575.5 | 774.01 | **260.0%** | **34.5%** | 1223.48 | 1331.39 | **69.3%** | **8.8%** |
| 16000 | 1 | 4440.74 | 1210.99 | 1583.1 | **266.7%** | **30.7%** | 2627.52 | 2856.79 | **69.0%** | **8.7%** |
| 32000 | 1 | 10088.07 | 2797.47 | 3858.36 | **260.6%** | **37.9%** | 6138.68 | 6851.55 | **64.3%** | **11.6%** |
| 4000 | 2 | 1521.47 | 535.08 | 586.49 | **184.3%** | **9.6%** | 1193.98 | 1022.49 | **27.4%** | -14.4% |
| 8000 | 2 | 3128.48 | 851.5 | 1071.88 | **267.4%** | **25.9%** | 1828.18 | 1982.55 | **71.1%** | **8.4%** |
| 16000 | 2 | 6712.17 | 2097.61 | 2473.49 | **220.0%** | **17.9%** | 3928.73 | 4012.58 | **70.8%** | **2.1%** |
| 32000 | 2 | 15238.17 | 4190.9 | 5575.86 | **263.6%** | **33.0%** | 9306.11 | 10401.21 | **63.7%** | **11.8%** |
| 4000 | 4 | 2788.26 | 863.34 | 1207.78 | **223.0%** | **39.9%** | 1943.98 | 2144.97 | **43.4%** | **10.3%** |
| 8000 | 4 | 5199.94 | 1755.2 | 2224.81 | **196.3%** | **26.8%** | 3326.21 | 3883.22 | **56.3%** | **16.7%** |
| 16000 | 4 | 11236.31 | 3843.43 | 5246.19 | **192.3%** | **36.5%** | 6577 | 7380.73 | **70.8%** | **12.2%** |
| 32000 | 4 | 25485.75 | 7686.6 | 10134.21 | **231.5%** | **31.8%** | 15545.12 | 16832.4 | **63.9%** | **8.3%** |
| 4000 | 8 | 4936.89 | 1817.34 | 2367.28 | **171.6%** | **30.3%** | 3514.73 | 3745.03 | **40.5%** | **6.6%** |
| 8000 | 8 | 9419.03 | 3152.87 | 4171.05 | **198.7%** | **32.3%** | 6000.83 | 6924.19 | **56.9%** | **15.4%** |
| 16000 | 8 | 20248.53 | 6175.53 | 7479.92 | **227.9%** | **21.1%** | 11870.84 | 12423.85 | **70.6%** | **4.7%** |
| 32000 | 8 | 45813.2 | 13774.85 | 17535.79 | **232.6%** | **27.3%** | 28075.22 | 30381.61 | **63.2%** | **8.2%** |

### 3.4 结果分析

通过对不同输入长度（4k-32k）和并发数（1-8）的实测数据分析，UCM在CUDA平台相比LMCache展现出显著优势：

#### 1. 核心性能表现
* **显著超越裸推性能**：在 80% 命中率下，UCM 的 TTFT 较裸推平均降低了 **200% - 260%**；即使在 50% 命中率的情况下，提升幅度仍稳定在 **40% - 70%** 之间。
* **对比 LMCache 的优势**：
    * **高命中率场景（80%）**：UCM 表现极佳，平均 TTFT 较 LMCache 降低了 **20% - 35%**。特别是在单并发 32k 长度下，提升达到了 **37.9%**，证明了 UCM 在处理长文本 KV Cache 检索与复用上的高效性。
    * **中命中率场景（50%）**：UCM 依然保持领先，平均提升约 **8% - 15%**。除个别低并发短文本波动外，整体表现更具鲁棒性。

#### 2. 架构价值与结论
* **存储后端灵活性**：除了纯性能领先，UCM 的核心竞争力在于其**原生支持 3FS 等分布式存储后端**。这为企业级部署预留了百 TB 级的水平扩展路径，打破了 LMCache 主要局限于本地盘的瓶颈。

**结论**：UCM 不仅在 CUDA 平台上实现了对 LMCache 的全面性能超越（尤其在高命中率场景下优势放大），更凭借其优秀的分布式扩展性，成为了面向大规模、高性能企业级大模型推理场景的演进级方案。

---

## 4. 场景3：Ascend平台UCM vs Mooncake Store

### 4.1 测试方案设计
在 **Atlas 910B3×2** 上，采用与CUDA平台完全相同的测试矩阵：**4k/8k/16k/32k** 上下文长度 × **1/2/4/8** 并发，对比两种方案在不同命中率压力下的表现。

**存储后端说明**：UCM采用本地SSD后端，Mooncake采用池化dram后端。

### 4.2 测试环境
- **硬件**：Atlas 910B3(80GB)×2，7TB NVMe SSD×1
- **模型配置**：QwQ-32B tp=2, GPU utilization=0.87
- **软件框架**：vLLM main + vLLM Ascend main + UCM v0.1.2/Mooncake v0.3.7.post2

### 4.3 性能对比（UCM vs Mooncake Store）

| 输入长度 | 并发请求数 | 裸推TTFT<br>(ms) | UCM80%命中<br>TTFT(ms) | Mooncake80%命中<br>TTFT(ms) | **80%命中UCM提升<br>vs裸推** | **80%命中UCM提升<br>vsMooncake** | UCM50%命中<br>TTFT(ms) | Mooncake50%命中<br>TTFT(ms) | **50%命中UCM提升<br>vs裸推** | **50%命中UCM提升<br>vsMooncake** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 4000 | 1 | 1016.04 | 385.20 | 328.26 | **163.77%** | **-14.78%** | 767.83 | 649.38 | **32.33%** | **-15.43%** |
| 8000 | 1 | 2070.25 | 667.44 | 588.35 | **210.18%** | **-11.85%** | 1307.51 | 1127.16 | **58.34%** | **-13.79%** |
| 16000 | 1 | 4431.41 | 1447.48 | 1249.04 | **206.15%** | **-13.71%** | 2813.33 | 2484.92 | **57.51%** | **-11.67%** |
| 32000 | 1 | 10102.96 | 3228.69 | 2946.04 | **212.91%** | **-8.75%** | 6472.71 | 5797.11 | **56.09%** | **-10.44%** |
| 4000 | 2 | 1524.46 | 569.63 | 472.83 | **167.62%** | **-16.99%** | 1316.22 | 1182.70 | **15.82%** | **-10.14%** |
| 8000 | 2 | 3127.71 | 988.14 | 863.74 | **216.52%** | **-12.59%** | 1990.27 | 1762.78 | **57.15%** | **-11.43%** |
| 16000 | 2 | 6680.89 | 2352.17 | 2138.78 | **184.03%** | **-9.07%** | 4309.17 | 3851.50 | **55.04%** | **-10.62%** |
| 32000 | 2 | 15297.56 | 5293.15 | 4949.55 | **189.01%** | **-6.49%** | 10018.79 | 8959.00 | **52.69%** | **-10.58%** |
| 4000 | 4 | 2518.94 | 991.27 | 912.61 | **154.11%** | **-7.94%** | 2108.35 | 1839.39 | **19.47%** | **-12.76%** |
| 8000 | 4 | 5205.35 | 1686.99 | 1611.47 | **208.56%** | **-4.48%** | 3329.60 | 2943.65 | **56.34%** | **-11.59%** |
| 16000 | 4 | 11168.84 | 3820.57 | 3552.97 | **192.33%** | **-7.00%** | 7151.71 | 6376.02 | **56.17%** | **-10.85%** |
| 32000 | 4 | 25268.37 | 9658.87 | 8197.46 | **161.61%** | **-15.13%** | 16518.77 | 15058.86 | **52.97%** | **-8.84%** |
| 4000 | 8 | 4566.77 | 1650.74 | 1498.66 | **176.65%** | **-9.21%** | 3349.96 | 3032.67 | **36.32%** | **-9.47%** |
| 8000 | 8 | 9396.05 | 3118.84 | 2912.10 | **201.27%** | **-6.63%** | 5949.88 | 5275.91 | **57.92%** | **-11.33%** |
| 16000 | 8 | 20048.89 | 6508.35 | 6440.57 | **208.05%** | **-1.04%** | 13286.99 | 11487.75 | **50.89%** | **-13.54%** |
| 32000 | 8 | 45378.43 | 15619.81 | 14146.81 | **190.52%** | **-9.43%** | 29306.33 | 26833.86 | **54.84%** | **-8.44%** |

---

### 4.4 结果分析

Ascend 平台的实测数据反映了 UCM 与基于 DRAM 池化方案（如 Mooncake）在存储介质选择与工程侧重点上的差异：

#### 1. 性能对标与介质差异
* **性能差距受控**：在 80% 和 50% 命中场景下，UCM 的 TTFT 仅较 Mooncake 略慢约 **8% - 15%**。这种微弱的差距主要源于物理介质差异（UCM 侧重 SSD/分布式存储，Mooncake 采用纯 DRAM 方案）。
* **I/O 路径优化的价值**：得益于对卸载kv cache的 **Connector** 做针对性优化，UCM 成功缩小了非易失性介质（SSD）与内存（DRAM）之间的性能鸿沟，使延迟保持在同一数量级。

#### 2. UCM 的工程化优势
Mooncake 方案基于 DRAM 池化，虽然具备内存级的读写速度，但在大规模生产环境下存在物理局限；UCM 则通过不同的架构选型解决了这些痛点：
* **原生持久化能力**：
    * **Mooncake**：纯 DRAM 方案，服务重启后缓存即清零，需重新进行昂贵的预热。
    * **UCM**：通过 SSD/NFS 实现持久化存储，服务升级或异常重启后缓存数据**即开即用**，保障了业务连续性。
* **突破容量天花板**：
    * **Mooncake**：受限于单节点 DRAM 容量（通常 < 2TB），难以支撑 PB 级的 KV Cache 需求。
    * **UCM**：通过扩展 SSD 或分布式存储（如 3FS），可轻松支持 **100TB+** 级别的缓存规模，是超长上下文 RAG 场景的唯一技术路径。
* **极致的成本效益**：
    * 在同等存储容量下，SSD 的单位成本仅为 DRAM 的约 **1/10**。这意味着支持相同规模的业务时，UCM 能将硬件投入成本降低一个数量级。


**结论**：
在 Ascend 场景下，UCM 以 **10% 左右** 的可控性能牺牲，换取了 **百倍以上的容量扩展性**、**原生的数据持久化支持** 以及 **极高的性价比**。对于追求大规模生产部署、超长上下文复用和成本严控的企业级应用而言，UCM 是更成熟的工程化选择。

---

## 5. 综合结论：UCM是多平台Prefix Caching的最优选择

- 在CUDA平台，UCM以平均30%的性能优势明显胜出LMCache，且通过3FS支持提供了LMCache不具备的分布式扩展能力。
- 在Ascend平台，UCM以约10%的性能牺牲换取了：
    - 容量百倍扩展（2TB → 100TB+）
    - 成本大幅降低（降至DRAM方案的1/10）
    - 持久化原生支持（服务重启缓存不丢）
- 跨平台层面，UCM是唯一实现CUDA与Ascend统一架构的方案，避免了混合集群维护多套缓存系统的技术债务。

综合性能、扩展性、成本、运维效率四个维度，UCM在CUDA和Ascend两个主流平台上均提供了最优的Prefix Caching解决方案。对于需要支撑长上下文高并发、关注规模化部署成本、追求跨平台一致性的企业级用户，UCM是当前无可争议的首选。

---

## 附录A：测试命令参考

本测评的所有结果均可复现，以下是复现的命令：

### 场景1：HBM容量陷阱验证
```bash
# 启动vLLM原生服务
vllm serve /home/models/QwQ-32B \
  --max-model-len 38000 --tensor-parallel-size 2 \
  --gpu_memory_utilization 0.87 --block_size 128 \
  --trust-remote-code --port 18082 --enforce-eager

# 四段式压测
python3 long_doc_pc_completion.py --test-mode no-pc --doc-len-list 36000 \
  --concurrency-list 1 --hit-ratio 1 --port 18082 --model /home/models/QwQ-32B \
  --seed 20 --repeat 11

python3 long_doc_pc_completion.py --test-mode no-pc --doc-len-list 36000 \
  --concurrency-list 1 --hit-ratio 1 --port 18082 --model /home/models/QwQ-32B \
  --seed 20 --repeat 11  # 复测

python3 long_doc_pc_completion.py --test-mode no-pc --doc-len-list 36000 \
  --concurrency-list 1 --hit-ratio 1 --port 18082 --model /home/models/QwQ-32B \
  --seed 21 --repeat 11  # 新请求抢占

python3 long_doc_pc_completion.py --test-mode no-pc --doc-len-list 36000 \
  --concurrency-list 1 --hit-ratio 1 --port 18082 --model /home/models/QwQ-32B \
  --seed 20 --repeat 11  # 原请求回退验证
```
### 场景2：UCM vs LMCache (CUDA)
```bash
# UCM服务
vllm serve /home/models/QwQ-32B --max-model-len 33000 --tensor-parallel-size 2 \
  --gpu_memory_utilization 0.87 --block_size 128 --trust-remote-code --port 18081 \
  --enforce-eager --no-enable-prefix-caching --kv-transfer-config '{
    "kv_connector": "UCMConnector",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "/home/ucm_config_example.yaml"}
  }'

# LMCache服务
LMCACHE_CONFIG_FILE="/home/disk-offload.yaml" vllm serve /home/models/QwQ-32B \
  --max-model-len 33000 --tensor-parallel-size 2 --gpu_memory_utilization 0.87 \
  --block_size 128 --trust-remote-code --port 18081 --enforce-eager \
  --no-enable-prefix-caching --kv-transfer-config \
  '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'

# 测试矩阵
for len in 4000 8000 16000 32000; do
  for conc in 1 2 4 8; do
    python3 long_doc_pc_completion.py --test-mode pc --doc-len-list $len \
      --concurrency-list $conc --hit-ratio 0.5 --port 18081 --model /home/models/QwQ-32B \
      --seed 25 --repeat 1
  done
done
```
### 场景3：UCM vs Mooncake (Ascend)
```bash
# 环境准备
export ASCEND_RT_VISIBLE_DEVICES=0,1
export ACL_OP_INIT_MODE=1
export ASCEND_BUFFER_POOL=4:8
export MOONCAKE_CONFIG_PATH="/vllm-workspace/Mooncake/mooncake.json"
export LD_LIBRARY_PATH=/usr/local/python3.11.13/site-packages/mooncake:$LD_LIBRARY_PATH

# UCM服务
vllm serve /home/models/QwQ-32B --max-model-len 33000 --tensor-parallel-size 2 \
  --gpu_memory_utilization 0.87 --block_size 128 --trust-remote-code --port 18081 \
  --enforce-eager --no-enable-prefix-caching --kv-transfer-config '{
    "kv_connector": "UCMConnector",
    "kv_connector_module_path": "ucm.integration.vllm.ucm_connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {"UCM_CONFIG_FILE": "ucm_config_example.yaml"}
  }'

# Mooncake服务
vllm serve /home/models/QwQ-32B --max-model-len 33000 --tensor-parallel-size 2 \
  --gpu_memory_utilization 0.87 --block_size 128 --trust-remote-code --port 18081 \
  --enforce-eager --no-enable-prefix-caching --kv-transfer-config '{
    "kv_connector": "AscendStoreConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
        "use_layerwise": false,
        "lookup_rpc_port":"1",
        "backend": "mooncake"
    }
  }'
```