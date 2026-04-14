# Prefix Caching Benchmark

`long_doc_pc_completion.py` 用于复现长文档 Prefix Caching 测评，核心关注首 token 时延（TTFT）在不同文档长度、并发度和命中率下的变化。

详细测试背景、结果分析和图表说明见 [ucm_pcbench.md](./ucm_pcbench.md)。

## 1. 脚本功能

- 脚本路径：`benchmarks/long_doc_pc_completion.py`
- 请求接口：OpenAI-compatible `http://localhost:<port>/v1/completions`
- 测试流程：两轮 warmup -> 首轮写入 -> 等待 8 秒 -> 命中查询
- 输出文件：
  - `benchmark.log`：运行日志
  - `prefix_cache_benchmark.csv`：每组参数的 TTFT 结果

建议先进入 `benchmarks` 目录再执行脚本，这样日志和 CSV 会直接落在当前目录下：

```bash
cd benchmarks
```

## 2. 前置条件

1. 启动一个可用的推理服务，并暴露 OpenAI-compatible completions 接口。
2. 确认 `--port` 与服务监听端口一致。
3. 确认 `--model` 与服务端可识别的模型名或模型路径一致。
4. 本地安装 Python 依赖：

```bash
python3 -m pip install openai
```

## 3. 基线测试

`no-pc` 模式用于测量完全重算的 TTFT，可作为图表中的裸推基线。

```bash
python3 long_doc_pc_completion.py \
  --test-mode no-pc \
  --doc-len-list 4000 8000 16000 32000 \
  --concurrency-list 1 2 4 8 \
  --port 18081 \
  --model /home/models/QwQ-32B \
  --repeat 1
```

## 4. Prefix Caching 测试

`pc` 模式会先写入缓存，再按设定命中率发起查询请求。

100% 命中：

```bash
python3 long_doc_pc_completion.py \
  --test-mode pc \
  --doc-len-list 4000 8000 16000 32000 \
  --concurrency-list 1 2 4 8 \
  --hit-ratio 1.0 \
  --port 18082 \
  --model /home/models/QwQ-32B \
  --repeat 1
```

50% 命中：

```bash
python3 long_doc_pc_completion.py \
  --test-mode pc \
  --doc-len-list 4000 8000 16000 32000 \
  --concurrency-list 1 2 4 8 \
  --hit-ratio 0.5 \
  --port 18082 \
  --model /home/models/QwQ-32B \
  --repeat 1
```

30% 命中：

```bash
python3 long_doc_pc_completion.py \
  --test-mode pc \
  --doc-len-list 4000 8000 16000 32000 \
  --concurrency-list 1 2 4 8 \
  --hit-ratio 0.3 \
  --port 18082 \
  --model /home/models/QwQ-32B \
  --repeat 1
```

0% 命中：

```bash
python3 long_doc_pc_completion.py \
  --test-mode pc \
  --doc-len-list 4000 8000 16000 32000 \
  --concurrency-list 1 2 4 8 \
  --hit-ratio 0.0 \
  --port 18082 \
  --model /home/models/QwQ-32B \
  --repeat 1
```

## 5. 常用参数

| 参数 | 说明 |
| --- | --- |
| `--test-mode` | `pc` 表示 Prefix Caching 测试，`no-pc` 表示完全重算基线 |
| `--doc-len-list` | 一次测试多组输入长度，单位为 token |
| `--concurrency-list` | 一次测试多组并发度 |
| `--hit-ratio` | 命中率，取值范围为 `0.0` 到 `1.0` |
| `--output-len` | 每个请求生成的输出 token 数，默认值为 `1` |
| `--repeat` | 每组参数重复次数，脚本会把请求数扩展为 `concurrency * repeat` |
| `--port` | 推理服务端口 |
| `--model` | 传给推理服务的模型标识 |
| `--seed` | 随机种子，用于复现实验 |

## 6. 结果读取

- `no-pc` 模式会记录完全重算的平均 TTFT。
- `pc` 模式会分别记录写入阶段 TTFT 和命中查询阶段 TTFT。
- 所有结果都会追加写入 `prefix_cache_benchmark.csv`，便于后处理和画图。

如果需要查看完整评测结论、测试场景和图表解释，请继续阅读 [ucm_pcbench.md](./ucm_pcbench.md)。
