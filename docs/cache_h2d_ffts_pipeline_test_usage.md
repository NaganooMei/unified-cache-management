# CacheStore H2D FFTS Pipeline 测试脚本运行说明

## 目标

这份文档只说明三件事：

- 怎么编译带 FFTS pipeline 的 UCM。
- 怎么运行 Qwen32B TP8/TP4 baseline case。
- 测试脚本有哪些常用可选参数。

测试脚本：

`@ucm/store/test/e2e/cache_h2d_ffts_pipeline_test.py`

## 编译

在 Ascend 环境上执行。先加载 CANN 环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

推荐用 editable install 编译并安装当前仓库：

```bash
cd unified-cache-management

PLATFORM=ascend \
ENABLE_SPARSE=false \
UCM_ENABLE_ASCEND_FFTS_PIPELINE=1 \
ASCEND_ROOT=/usr/local/Ascend/ascend-toolkit/latest \
pip install -v -e . --no-build-isolation
```

关键点：

- `PLATFORM=ascend` 会按 Ascend runtime 编译 UCM。
- `UCM_ENABLE_ASCEND_FFTS_PIPELINE=1` 会打开 CacheStore H2D FFTS pipeline 编译开关。
- `ASCEND_ROOT` 指向 Ascend toolkit 路径，用来查找 `ascendcl`、FFTS header 和 `libruntime`。
- 如果没有打开 FFTS pipeline 编译开关，运行 `ffts_pipeline` case 时 CacheStore setup 会失败。

## 运行 Qwen32B Baseline

baseline 会依次跑：

```text
qwen32b_tp8_full: 128 * 32KiB = 4MiB per shard
qwen32b_tp4_full: 128 * 64KiB = 8MiB per shard
```

每个 case 都会对比：

```text
ce
ffts_pipeline
```

运行命令：

```bash
cd unified-cache-management

UCM_FFTS_MODEL_CASE=qwen32b_baseline \
UCM_FFTS_TORCH_DEVICE=npu \
UCM_FFTS_DEVICE_ID=0 \
python ucm/store/test/e2e/cache_h2d_ffts_pipeline_test.py
```

默认每次 load 使用 `16` 个 shard：

```text
qwen32b_tp8_full: 16 * 4MiB = 64MiB per load
qwen32b_tp4_full: 16 * 8MiB = 128MiB per load
```

脚本会输出每个 case 的配置、CE 结果、FFTS pipeline 结果，以及最终 summary 表：

```text
summary: case,transport,blocks,fragments,shard,bytes,avg_ms,median_ms,min_ms,gbps,ffts_vs_ce
```

## 只跑单个 Case

只跑 TP8：

```bash
UCM_FFTS_MODEL_CASE=qwen32b_tp8_full \
UCM_FFTS_TORCH_DEVICE=npu \
UCM_FFTS_DEVICE_ID=0 \
python ucm/store/test/e2e/cache_h2d_ffts_pipeline_test.py
```

只跑 TP4：

```bash
UCM_FFTS_MODEL_CASE=qwen32b_tp4_full \
UCM_FFTS_TORCH_DEVICE=npu \
UCM_FFTS_DEVICE_ID=0 \
python ucm/store/test/e2e/cache_h2d_ffts_pipeline_test.py
```

只跑 FFTS pipeline，不跑 CE baseline：

```bash
UCM_FFTS_MODEL_CASE=qwen32b_baseline \
UCM_FFTS_COMPARE_CE=0 \
UCM_FFTS_TORCH_DEVICE=npu \
UCM_FFTS_DEVICE_ID=0 \
python ucm/store/test/e2e/cache_h2d_ffts_pipeline_test.py
```

## 常用参数

### Case 选择

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `UCM_FFTS_MODEL_CASE` | unset | 选择一个 case 或 case group。 |
| `UCM_FFTS_MODEL_CASES` | unset | 选择多个 case，用逗号分隔；优先级高于 `UCM_FFTS_MODEL_CASE`。 |

当前内置 case：

| case | 含义 |
| --- | --- |
| `qwen32b_baseline` | 依次跑 `qwen32b_tp8_full` 和 `qwen32b_tp4_full`。 |
| `baseline` | `qwen32b_baseline` 的别名。 |
| `qwen32b_tp8_full` | 128 个 32KiB fragment，4MiB shard。 |
| `qwen32b_tp8` | `qwen32b_tp8_full` 的别名。 |
| `qwen32b_tp4_full` | 128 个 64KiB fragment，8MiB shard。 |
| `qwen32b_tp4` | `qwen32b_tp4_full` 的别名。 |

如果不设置 case，脚本走自定义 tensor shape：

```text
UCM_FFTS_FRAGMENT_COUNT
UCM_FFTS_FRAGMENT_BYTES
UCM_FFTS_TENSOR_SIZES
```

### 设备

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `UCM_FFTS_TORCH_DEVICE` | `cuda` | Torch 设备类型。Ascend 上设置为 `npu`。 |
| `UCM_FFTS_DEVICE_ID` | `0` | 设备号。 |

Ascend 推荐：

```bash
UCM_FFTS_TORCH_DEVICE=npu
UCM_FFTS_DEVICE_ID=0
```

### 数据规模

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `UCM_FFTS_BLOCK_NUM` | `16` | 一次 load task 中的 shard 数。 |
| `UCM_FFTS_WARMUP` | `2` | 计时前 warmup 次数。 |
| `UCM_FFTS_REPEAT` | `10` | 正式计时次数。 |

示例：只测 1 个 shard：

```bash
UCM_FFTS_BLOCK_NUM=1 \
UCM_FFTS_MODEL_CASE=qwen32b_baseline \
UCM_FFTS_TORCH_DEVICE=npu \
UCM_FFTS_DEVICE_ID=0 \
python ucm/store/test/e2e/cache_h2d_ffts_pipeline_test.py
```

### 自定义 Tensor Shape

不设置 `UCM_FFTS_MODEL_CASE` 时生效。

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `UCM_FFTS_FRAGMENT_COUNT` | `128` | fragment 数。 |
| `UCM_FFTS_FRAGMENT_BYTES` | `32768` | 每个 fragment 的字节数。 |
| `UCM_FFTS_TENSOR_SIZES` | unset | 显式指定每个 fragment 字节数，用逗号分隔；设置后覆盖前两个参数。 |

示例：128 个 32KiB fragment：

```bash
UCM_FFTS_FRAGMENT_COUNT=128 \
UCM_FFTS_FRAGMENT_BYTES=32768 \
UCM_FFTS_TORCH_DEVICE=npu \
UCM_FFTS_DEVICE_ID=0 \
python ucm/store/test/e2e/cache_h2d_ffts_pipeline_test.py
```

示例：混合 fragment：

```bash
UCM_FFTS_TENSOR_SIZES=131072,16384,4096,256 \
UCM_FFTS_TORCH_DEVICE=npu \
UCM_FFTS_DEVICE_ID=0 \
python ucm/store/test/e2e/cache_h2d_ffts_pipeline_test.py
```

### FFTS Pipeline 参数

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `UCM_FFTS_PIPELINE_DEPTH` | `2` | device staging slot 数。 |
| `UCM_FFTS_MAX_READY_LANES` | `8` | FFTS launch ready context 数上限。 |

示例：

```bash
UCM_FFTS_MODEL_CASE=qwen32b_baseline \
UCM_FFTS_PIPELINE_DEPTH=2 \
UCM_FFTS_MAX_READY_LANES=8 \
UCM_FFTS_TORCH_DEVICE=npu \
UCM_FFTS_DEVICE_ID=0 \
python ucm/store/test/e2e/cache_h2d_ffts_pipeline_test.py
```

### CacheStore 参数

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `UCM_FFTS_CACHE_STREAM_NUMBER` | `4` | CE 路径的 stream 数；FFTS Load executor 不使用这个 stream pool。 |
| `UCM_FFTS_CACHE_BUFFER_CAPACITY_GB` | auto | CacheStore buffer 容量。未设置时脚本按 case 自动选择，TP8 默认 4GB，TP4 默认 8GB。 |
| `UCM_FFTS_LOAD_EXCLUSIVE_BUFFER_NUMBER` | `64` | CacheStore load exclusive buffer 数。 |
| `UCM_FFTS_WAITING_QUEUE_DEPTH` | `64` | CacheStore waiting queue 深度。 |
| `UCM_FFTS_RUNNING_QUEUE_DEPTH` | `4096` | CacheStore running queue 深度。 |
| `UCM_FFTS_TIMEOUT_MS` | `30000` | CacheStore task timeout。 |
| `UCM_FFTS_SHARE_BUFFER_ENABLE` | `true` | 是否使用 shared buffer。 |

### 结果阈值

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `UCM_FFTS_COMPARE_CE` | `1` | 是否同时跑 CE baseline。 |
| `UCM_FFTS_MAX_SLOWDOWN` | `5.0` | 如果跑 CE，对 FFTS/CE 平均耗时比做 sanity check。 |
| `UCM_FFTS_MIN_GBPS` | `0.0` | 如果大于 0，检查 FFTS pipeline 带宽下限。 |

## 推荐命令

当前最常用 baseline 命令：

```bash
cd unified-cache-management

UCM_FFTS_MODEL_CASE=qwen32b_baseline \
UCM_FFTS_TORCH_DEVICE=npu \
UCM_FFTS_DEVICE_ID=0 \
UCM_FFTS_BLOCK_NUM=16 \
UCM_FFTS_WARMUP=2 \
UCM_FFTS_REPEAT=10 \
python ucm/store/test/e2e/cache_h2d_ffts_pipeline_test.py
```
