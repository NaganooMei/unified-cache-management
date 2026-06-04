# CacheStore H2D 测试脚本运行说明

## 目标

这份文档只说明三件事：

- 怎么编译带 FFTS pipeline 的 UCM。
- 怎么分别运行 GQA CE baseline、GQA FFTS pipeline 和 DSV4 FAWA case。
- 这些脚本有哪些常用参数。

测试脚本：

`@ucm/store/test/e2e/cache_h2d_ce_baseline_test.py`

`@ucm/store/test/e2e/cache_h2d_ffts_pipeline_test.py`

`@ucm/store/test/e2e/cache_h2d_dsv4_ffts_pipeline_test.py`

GQA 的 CE/FFTS 两个脚本每次都只跑一个 case，不会同时跑多个 TP case，也不会在一个脚本里同时比较 CE 和 FFTS。DSV4 FAWA 脚本会在同一轮请求中创建 FA store 和 WA store，用来模拟 DSV4 的双 store load 形态。当前这些性能脚本默认关闭数据正确性校验；如果要做功能校验，可以显式设置 `UCM_FFTS_VALIDATE=1`。

## 编译

在 Ascend 环境上执行。先加载 CANN 环境：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

推荐用 editable install 编译并安装当前仓库：

```bash
PLATFORM=ascend \
ENABLE_SPARSE=false \
ASCEND_ROOT=/usr/local/Ascend/ascend-toolkit/latest \
pip install -v -e . --no-build-isolation
```

关键点：

- `PLATFORM=ascend` 按 Ascend runtime 编译 UCM。
- CacheStore H2D FFTS pipeline 编译开关默认开启；如果需要显式关闭，可以设置 `UCM_ENABLE_ASCEND_FFTS_PIPELINE=0`。
- `ASCEND_ROOT` 指向 Ascend toolkit 路径。
- 如果显式关闭 FFTS pipeline 编译开关，运行 FFTS pipeline 脚本会在 CacheStore setup 阶段失败。

## 运行 CE Baseline

GQA TP4 full 当前实测命令：

```bash
UCM_FFTS_MODEL_CASE=qwen32b_tp4_full \
UCM_FFTS_TORCH_DEVICE=npu \
UCM_FFTS_DEVICE_ID=0 \
UCM_FFTS_BLOCK_NUM=100 \
UCM_FFTS_SHARE_BUFFER_ENABLE=0 \
UCM_FFTS_WARMUP=1 \
UCM_FFTS_REPEAT=100 \
python ucm/store/test/e2e/cache_h2d_ce_baseline_test.py
```

## 运行 FFTS Pipeline

GQA TP4 full，按 2MiB object 拆：

```bash
UCM_FFTS_MODEL_CASE=qwen32b_tp4_full \
UCM_FFTS_OBJECT_TARGET_BYTES=2097152 \
UCM_FFTS_TORCH_DEVICE=npu \
UCM_FFTS_DEVICE_ID=0 \
UCM_FFTS_BLOCK_NUM=100 \
UCM_FFTS_WARMUP=1 \
UCM_FFTS_REPEAT=100 \
UCM_FFTS_SHARE_BUFFER_ENABLE=0 \
python ucm/store/test/e2e/cache_h2d_ffts_pipeline_test.py
```

GQA TP4 full，不拆 shard：

```bash
UCM_FFTS_MODEL_CASE=qwen32b_tp4_full \
UCM_FFTS_OBJECT_TARGET_BYTES=0 \
UCM_FFTS_TORCH_DEVICE=npu \
UCM_FFTS_DEVICE_ID=0 \
UCM_FFTS_BLOCK_NUM=100 \
UCM_FFTS_WARMUP=1 \
UCM_FFTS_REPEAT=100 \
UCM_FFTS_SHARE_BUFFER_ENABLE=0 \
python ucm/store/test/e2e/cache_h2d_ffts_pipeline_test.py
```

GQA TP8 full，按 2MiB object 拆：

```bash
UCM_FFTS_MODEL_CASE=qwen32b_tp8_full \
UCM_FFTS_OBJECT_TARGET_BYTES=2097152 \
UCM_FFTS_TORCH_DEVICE=npu \
UCM_FFTS_DEVICE_ID=0 \
UCM_FFTS_BLOCK_NUM=300 \
UCM_FFTS_WARMUP=1 \
UCM_FFTS_REPEAT=100 \
UCM_FFTS_SHARE_BUFFER_ENABLE=0 \
python ucm/store/test/e2e/cache_h2d_ffts_pipeline_test.py
```

GQA TP8 full，不拆 shard：

```bash
UCM_FFTS_MODEL_CASE=qwen32b_tp8_full \
UCM_FFTS_OBJECT_TARGET_BYTES=0 \
UCM_FFTS_TORCH_DEVICE=npu \
UCM_FFTS_DEVICE_ID=0 \
UCM_FFTS_BLOCK_NUM=300 \
UCM_FFTS_WARMUP=1 \
UCM_FFTS_REPEAT=100 \
UCM_FFTS_SHARE_BUFFER_ENABLE=0 \
python ucm/store/test/e2e/cache_h2d_ffts_pipeline_test.py
```

按当前内置 case 定义，`qwen32b_tp8_full` 是 32KiB fragment，`qwen32b_tp4_full` 是 64KiB fragment。实测命令按 case 名称记录，具体 fragment size 以 case 表为准。

## 运行 DSV4 FAWA

DSV4 FAWA 脚本会创建两个 CacheStore：FA store 和 WA store。这里 `UCM_FFTS_BLOCK_NUM` 表示 external hit blocks，也就是 FA rows 数；WA store 固定 load 1 个 boundary row。V4 测试需要打开 shared buffer，以贴近真实 FAWA 双 store 路径。

DSV4 CE：

```bash
UCM_FFTS_BLOCK_NUM=100 \
UCM_FFTS_H2D_TRANSPORT=ce \
UCM_FFTS_TORCH_DEVICE=npu \
UCM_FFTS_DEVICE_ID=0 \
UCM_FFTS_WARMUP=1 \
UCM_FFTS_REPEAT=100 \
UCM_FFTS_SHARE_BUFFER_ENABLE=1 \
python ucm/store/test/e2e/cache_h2d_dsv4_ffts_pipeline_test.py
```

DSV4 FFTS pipeline，按 2MiB object 拆：

```bash
UCM_FFTS_BLOCK_NUM=100 \
UCM_FFTS_H2D_TRANSPORT=ffts_pipeline \
UCM_FFTS_OBJECT_TARGET_BYTES=2097152 \
UCM_FFTS_TORCH_DEVICE=npu \
UCM_FFTS_DEVICE_ID=0 \
UCM_FFTS_WARMUP=1 \
UCM_FFTS_REPEAT=100 \
UCM_FFTS_SHARE_BUFFER_ENABLE=1 \
python ucm/store/test/e2e/cache_h2d_dsv4_ffts_pipeline_test.py
```

DSV4 FFTS pipeline，不拆 shard：

```bash
UCM_FFTS_BLOCK_NUM=100 \
UCM_FFTS_H2D_TRANSPORT=ffts_pipeline \
UCM_FFTS_OBJECT_TARGET_BYTES=0 \
UCM_FFTS_TORCH_DEVICE=npu \
UCM_FFTS_DEVICE_ID=0 \
UCM_FFTS_WARMUP=1 \
UCM_FFTS_REPEAT=100 \
UCM_FFTS_SHARE_BUFFER_ENABLE=1 \
python ucm/store/test/e2e/cache_h2d_dsv4_ffts_pipeline_test.py
```

如果只想采 runtime 和 copy task profiling，可以保持校验关闭并只测 1 次：

```bash
UCM_FFTS_VALIDATE=0 \
UCM_FFTS_BLOCK_NUM=100 \
UCM_FFTS_H2D_TRANSPORT=ffts_pipeline \
UCM_FFTS_OBJECT_TARGET_BYTES=2097152 \
UCM_FFTS_TORCH_DEVICE=npu \
UCM_FFTS_DEVICE_ID=0 \
UCM_FFTS_WARMUP=1 \
UCM_FFTS_REPEAT=1 \
UCM_FFTS_SHARE_BUFFER_ENABLE=1 \
msprof --output=./prof_out/ucm_dsv4_ffts_runtime_copy_only \
  --ascendcl=on \
  --runtime-api=on \
  --task-time=on \
  python ucm/store/test/e2e/cache_h2d_dsv4_ffts_pipeline_test.py
```

## 输出含义

GQA CE/FFTS 脚本都会输出一行 `case_config`、一行结果，以及最终 `summary`。

summary 字段：

```text
case,transport,blocks,fragments,shard,object_target,objects_per_shard,max_object,max_object_fragments,bytes,avg_ms,median_ms,min_ms,gbps
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `case` | 当前单个 case 名称。 |
| `transport` | `ce` 或 `ffts_pipeline`。 |
| `blocks` | 一次 load task 里的 shard 数。 |
| `fragments` | 一个 shard 内的 device fragment 数。 |
| `shard` | 单个 CacheStore shard 大小。 |
| `object_target` | FFTS pipeline 目标 object 大小；CE 脚本固定为 `0B`。 |
| `objects_per_shard` | 一个 shard 被拆成几个 FFTS object；CE 为空。 |
| `max_object` | 拆分后单个 object 的最大大小；CE 为空。 |
| `max_object_fragments` | 拆分后单个 object 的最大 fragment 数；CE 为空。 |
| `bytes` | 一次计时 load 的有效 payload 字节数。 |
| `avg_ms` | repeat 样本平均耗时。 |
| `median_ms` | repeat 样本中位数耗时。 |
| `min_ms` | repeat 样本最小耗时。 |
| `gbps` | `bytes / avg_ms` 换算出的端到端有效业务带宽。 |

这里的耗时范围是 `load_data + wait`，不是裸 `aclrtMemcpyAsync` 耗时。

DSV4 FAWA 脚本还会额外输出 `dsv4_fa_config`、`dsv4_wa_config` 和 `dsv4_expected_io`。其中 `dsv4_expected_io` 用来对照 profiler 里的 CE fragment copy 数或 FFTS object 数。

## Case 参数

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `UCM_FFTS_MODEL_CASE` | unset | 选择一个 GQA case。只接受单个 case，不支持逗号分隔和 group。DSV4 FAWA 脚本不使用这个参数。 |

当前内置 case：

| case | 含义 |
| --- | --- |
| `qwen32b_tp8_full` | 128 个 32KiB fragment，4MiB shard。 |
| `qwen32b_tp8` | `qwen32b_tp8_full` 的别名。 |
| `qwen32b_tp8_2m` | 64 个 32KiB fragment，2MiB shard，用于模拟 TP8 2MiB object。 |
| `qwen32b_tp8_1m` | 32 个 32KiB fragment，1MiB shard，用于模拟 TP8 1MiB object。 |
| `qwen32b_tp4_full` | 128 个 64KiB fragment，8MiB shard。 |
| `qwen32b_tp4` | `qwen32b_tp4_full` 的别名。 |
| `qwen32b_tp4_2m` | 32 个 64KiB fragment，2MiB shard，用于模拟 TP4 2MiB object。 |
| `qwen32b_tp4_1m` | 16 个 64KiB fragment，1MiB shard，用于模拟 TP4 1MiB object。 |
| `qwen32b_tp2_full` | 128 个 128KiB fragment，16MiB shard。 |
| `qwen32b_tp2` | `qwen32b_tp2_full` 的别名。 |
| `qwen32b_tp2_2m` | 16 个 128KiB fragment，2MiB shard，用于模拟 TP2 2MiB object。 |
| `qwen32b_tp2_1m` | 8 个 128KiB fragment，1MiB shard，用于模拟 TP2 1MiB object。 |
| `qwen32b_tp1_full` | 128 个 256KiB fragment，32MiB shard。 |
| `qwen32b_tp1` | `qwen32b_tp1_full` 的别名。 |
| `qwen32b_tp1_2m` | 8 个 256KiB fragment，2MiB shard，用于模拟 TP1 2MiB object。 |
| `qwen32b_tp1_1m` | 4 个 256KiB fragment，1MiB shard，用于模拟 TP1 1MiB object。 |

如果不设置 `UCM_FFTS_MODEL_CASE`，脚本会走自定义 tensor shape：

```text
UCM_FFTS_FRAGMENT_COUNT
UCM_FFTS_FRAGMENT_BYTES
UCM_FFTS_TENSOR_SIZES
```

## 常用参数

### 设备

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `UCM_FFTS_TORCH_DEVICE` | `cuda` | Torch 设备类型。Ascend 上设置为 `npu`。 |
| `UCM_FFTS_DEVICE_ID` | `0` | 设备号。 |

### 数据规模

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `UCM_FFTS_BLOCK_NUM` | `16` | 一次 load task 中的 shard 数。 |
| `UCM_FFTS_WARMUP` | `2` | 计时前 warmup 次数。 |
| `UCM_FFTS_REPEAT` | `10` | 正式计时次数。 |

### 自定义 Tensor Shape

不设置 `UCM_FFTS_MODEL_CASE` 时生效。

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `UCM_FFTS_FRAGMENT_COUNT` | `128` | fragment 数。 |
| `UCM_FFTS_FRAGMENT_BYTES` | `32768` | 每个 fragment 的字节数。 |
| `UCM_FFTS_TENSOR_SIZES` | unset | 显式指定每个 fragment 字节数，用逗号分隔；设置后覆盖前两个参数。 |

### FFTS Pipeline 与 DSV4 Transport

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `UCM_FFTS_PIPELINE_DEPTH` | `2` | device staging slot 数。 |
| `UCM_FFTS_MAX_READY_LANES` | `8` | FFTS launch ready context 数上限。 |
| `UCM_FFTS_OBJECT_TARGET_BYTES` | `0` | 单 shard 内部固定聚合目标大小。`0` 表示保持整 shard object；非 0 时按 fragment 边界拆成多个 object。 |
| `UCM_FFTS_H2D_TRANSPORT` | `ffts_pipeline` | 只对 DSV4 FAWA 脚本有意义。取值为 `ce` 或 `ffts_pipeline`。 |

### CacheStore

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `UCM_FFTS_CACHE_STREAM_NUMBER` | `4` | CE 路径的 stream 数。FFTS Load executor 不使用这个 stream pool。 |
| `UCM_FFTS_CACHE_BUFFER_CAPACITY_GB` | auto | CacheStore buffer 容量。未设置时脚本按 case 自动选择。 |
| `UCM_FFTS_LOAD_EXCLUSIVE_BUFFER_NUMBER` | `64` | CacheStore load exclusive buffer 数。 |
| `UCM_FFTS_WAITING_QUEUE_DEPTH` | `64` | CacheStore waiting queue 深度。 |
| `UCM_FFTS_RUNNING_QUEUE_DEPTH` | `4096` | CacheStore running queue 深度。 |
| `UCM_FFTS_TIMEOUT_MS` | `30000` | CacheStore task timeout。 |
| `UCM_FFTS_SHARE_BUFFER_ENABLE` | `true` | 是否使用 shared buffer。 |
| `UCM_FFTS_VALIDATE` | `false` | 是否执行额外的 load + allclose 数据正确性校验。性能测试默认关闭；功能校验时可设为 `1`。 |
