# A5 Cache 带宽对比

实验分支：`a5-exp`，从最新 develop 整理。对比下面两条实际路径：

| 路径 | Cache 实现 | 内存 | O_DIRECT | share_buffer_enable |
| --- | --- | --- | --- | --- |
| mallocHost | cache/cc | aclrtMallocHost | 关闭 | false |
| mmap | cache/v2 | memfd mmap，不注册 host | 开启 | true |

Posix 两组都使用 aio，不清 Page Cache。Cache 命中先预填一轮，再重复读取同一组 key；
Posix 命中先预埋数据，再启动空 Cache，每轮使用不同 key，避免读成 UCM Cache 命中。

`/dev/shm` 通常是 tmpfs，文件数据就在内存中。这组结果用于比较内存文件系统上的
Posix→Cache→NPU 路径，不代表物理盘冷读性能。tmpfs 的 O_DIRECT 支持由内核决定；
即使支持，也不会把数据变成冷盘读取。脚本不会清缓存、修改挂载参数或更改系统配置。

## 路径与准备

以下示例假设服务器仓库在 `/workspace/unified-cache-management`。如果仓库目录不同，
替换命令中的这个前缀。脚本直接从仓库运行，不需要复制到 `/tmp`。
测试文件统一放在 `/dev/shm/ucm-a5-exp`。

```bash
cd /workspace/unified-cache-management
git fetch origin a5-exp
# 首次创建本地分支：
git switch -c a5-exp --track origin/a5-exp

mkdir -p /dev/shm/ucm-a5-exp/v4 /dev/shm/ucm-a5-exp/glm52
df -h /dev/shm
```

如果本地分支已经存在，使用 `git switch a5-exp`。
需要已安装匹配 A5 的 Torch/CANN 环境，逻辑设备 0–7 可用。

两组都是 `PLATFORM=ascend-a5`。本实验分支增加编译环境变量 `UCM_A5_CACHE_CORE`，
取值为 `cc` 或 `v2`，默认仍为 `cc`。切换后需要重新安装；脚本的 `--cache-core`
只适配参数，不会替换已编译的库。

## mallocHost：构建并运行

```bash
cd /workspace/unified-cache-management
export PLATFORM=ascend-a5
export UCM_A5_CACHE_CORE=cc
python -m pip install -v -e . --no-build-isolation
python -c "from ucm.store.cache.capabilities import cache_core_version; print(cache_core_version())"
```

版本应输出 `1`。下面的 `--worker-mode gqa` 用于关闭共享 key/Buffer；模型的
KV 字节布局仍由 `--model` 指定，GLM 不会因此换成 GQA 模型的数据布局。
`--no-direct-io` 让旧 Cache 选择 mallocHost，同时让 Posix 使用 buffered aio。
底层 PosixStore 支持这个组合，本分支移除了脚本之前过严的 aio/O_DIRECT 限制。

### V4 DP4TP1，4 stream

4 个进程对应逻辑 NPU 0–3，4 个 DP 的 Cache 域互相隔离。
总 Cache 64 GiB，每个 DP 16 GiB。
`dsv4` 是现有 FA-only 整存取 profile，不包含 WA，也不加载模型权重。
每个 DP 每次加载 512 个完整 FA block，单次约 1.52 GiB。

```bash
# Cache 命中
python /workspace/unified-cache-management/ucm/store/test/e2e/cache_bw.py \
  --mode cache-hit --cache-core legacy --model dsv4 \
  --dp-size 4 --tp-size 1 --worker-mode gqa \
  --blocks 512 --cache-capacity-gb 64 \
  --no-direct-io --cache-copy plain --cache-streams 4 \
  --posix-engine aio --epochs 30 --warmup 3 --epoch-interval-ms 15

# Posix 命中
python /workspace/unified-cache-management/ucm/store/test/e2e/cache_bw.py \
  --mode posix-hit --cache-core legacy --model dsv4 \
  --dp-size 4 --tp-size 1 --worker-mode gqa \
  --blocks 512 --cache-capacity-gb 64 \
  --no-direct-io --cache-copy plain --cache-streams 4 \
  --storage-backends /dev/shm/ucm-a5-exp/v4 \
  --posix-engine aio --epochs 10 --warmup 2 --epoch-interval-ms 15 \
  --max-storage-gb 500
```

### GLM5.2 DP1TP8，4 stream

8 个进程对应逻辑 NPU 0–7。每个 rank 都加载 512 block、78 层，
每个 block 每层为 `128K + 16K + 32K = 176 KiB`。
每层每 rank 约 88 MiB，每次完整请求每 rank 约 6.70 GiB。
总 Cache 64 GiB，mallocHost 每 rank 独立分配 8 GiB；8 个 rank 分别读写各自的数据。

```bash
# Cache 命中
python /workspace/unified-cache-management/ucm/store/test/e2e/cache_bw.py \
  --mode cache-hit --cache-core legacy --model glm-5.2 \
  --dp-size 1 --tp-size 8 --worker-mode gqa \
  --blocks 512 --layout layerwise --cache-capacity-gb 64 \
  --no-direct-io --cache-copy plain --cache-streams 4 \
  --posix-engine aio --epochs 30 --warmup 3 --epoch-interval-ms 15

# Posix 命中
python /workspace/unified-cache-management/ucm/store/test/e2e/cache_bw.py \
  --mode posix-hit --cache-core legacy --model glm-5.2 \
  --dp-size 1 --tp-size 8 --worker-mode gqa \
  --blocks 512 --layout layerwise --cache-capacity-gb 64 \
  --no-direct-io --cache-copy plain --cache-streams 4 \
  --storage-backends /dev/shm/ucm-a5-exp/glm52 \
  --posix-engine aio --epochs 5 --warmup 1 --epoch-interval-ms 15 \
  --max-storage-gb 500
```

## mmap：构建并运行

```bash
cd /workspace/unified-cache-management
export PLATFORM=ascend-a5
export UCM_A5_CACHE_CORE=v2
python -m pip install -v -e . --no-build-isolation
python -c "from ucm.store.cache.capabilities import cache_core_version; print(cache_core_version())"
```

版本应输出 `2`。启动日志应包含 `register_host=false`。
记录实际页类型；memfd 会按现有策略尝试大页并回退。
以下 `--worker-mode mla` 开启共享 key 和 owner 协调，`--direct-io` 开启 O_DIRECT。

下面的 mmap 命令显式加了 `--no-cache-numa-bind`，跳过 UCM 的 NUMA 探测和 mbind，
沿用进程/系统的内存策略。仍然触页，保留共享内存和原有注册策略，脚本不绑 CPU。
实际内存仍受容器 cpuset、进程继承的内存策略及大页池分布影响。
这个开关需要重新编译安装本分支的 C++ 库，不能只更新 Python 脚本。
启动日志应出现 `NumaBind to false` 和 `memfd NUMA binding disabled`。

程序默认仍启用绑定。不传 `--no-cache-numa-bind` 时，未指定节点就走 rank slice；
指定 `--cache-numa-nodes` 则走 affinity。关闭绑定时不能同时指定节点。
该开关只用于 v2；mallocHost 组保持旧 Cache 的分配方式。

### V4 DP4TP1，4 stream

每个 DP 仍是独立 Cache 域。TP=1，域内只有一个 rank，开启共享不会让 4 个 DP
共用同一份数据。这组主要对比分配和搬运路径，读写数据量与 mallocHost 组相同。

```bash
# Cache 命中
python /workspace/unified-cache-management/ucm/store/test/e2e/cache_bw.py \
  --mode cache-hit --cache-core v2 --model dsv4 \
  --no-cache-numa-bind \
  --dp-size 4 --tp-size 1 --worker-mode mla \
  --blocks 512 --cache-capacity-gb 64 \
  --direct-io --cache-copy plain --cache-streams 4 \
  --posix-engine aio --epochs 30 --warmup 3 --epoch-interval-ms 15

# Posix 命中
python /workspace/unified-cache-management/ucm/store/test/e2e/cache_bw.py \
  --mode posix-hit --cache-core v2 --model dsv4 \
  --no-cache-numa-bind \
  --dp-size 4 --tp-size 1 --worker-mode mla \
  --blocks 512 --cache-capacity-gb 64 \
  --direct-io --cache-copy plain --cache-streams 4 \
  --storage-backends /dev/shm/ucm-a5-exp/v4 \
  --posix-engine aio --epochs 10 --warmup 2 --epoch-interval-ms 15 \
  --max-storage-gb 500
```

### GLM5.2 DP1TP8，4 stream

8 个 rank 共用一个 Cache 域，总容量 64 GiB，物理数据区每 rank 8 GiB。
同一请求只预埋一份数据；每层 512 个 block 的读盘 owner 按 rank 打散，均分时
每个 rank 负责约 64 个 block。每个 rank 的 H2D 仍然是全量 512 个 block。
这组包含共享内存减少重复读盘的收益，属于两条实际方案的整体比较。

```bash
# Cache 命中
python /workspace/unified-cache-management/ucm/store/test/e2e/cache_bw.py \
  --mode cache-hit --cache-core v2 --model glm-5.2 \
  --no-cache-numa-bind \
  --dp-size 1 --tp-size 8 --worker-mode mla \
  --blocks 512 --layout layerwise --cache-capacity-gb 64 \
  --direct-io --cache-copy plain --cache-streams 4 \
  --posix-engine aio --epochs 30 --warmup 3 --epoch-interval-ms 15

# Posix 命中
python /workspace/unified-cache-management/ucm/store/test/e2e/cache_bw.py \
  --mode posix-hit --cache-core v2 --model glm-5.2 \
  --no-cache-numa-bind \
  --dp-size 1 --tp-size 8 --worker-mode mla \
  --blocks 512 --layout layerwise --cache-capacity-gb 64 \
  --direct-io --cache-copy plain --cache-streams 4 \
  --storage-backends /dev/shm/ucm-a5-exp/glm52 \
  --posix-engine aio --epochs 5 --warmup 1 --epoch-interval-ms 15 \
  --max-storage-gb 500
```

## mmap 经 mallocHost 中转后 H2D

在本分支按上面的 `UCM_A5_CACHE_CORE=v2` 重新编译安装。原有 v2 命令加上
`--cache-staging-blocks 64` 即可启用；不传这个参数仍是原来的 mmap 直接 H2D。
只支持 `--cache-core v2 --cache-copy plain`，Cache 命中和 Posix 命中都可以使用。

路径为 `共享 Cache → CPU memcpy → 本 rank 的 mallocHost 中转区 → NPU`。
中转区在启动时分配、逐块填充，满一批后等待全部 H2D stream 完成再复用；
每层结束也会等待 H2D 完成。保持 4 条 stream，不增加 CPU 工作线程，暂不使用双缓冲。
GLM 的 64 个中转 slot 每 rank 占 11 MiB，V4 则约 194 MiB；不改变原有 Cache 容量。
这一步只改变 Load 的 H2D 来源，Dump 仍使用原路径。

```bash
# GLM：共享 mmap + mallocHost 中转，DP1TP8
python /workspace/unified-cache-management/ucm/store/test/e2e/cache_bw.py \
  --mode cache-hit --cache-core v2 --model glm-5.2 \
  --no-cache-numa-bind --dp-size 1 --tp-size 8 --worker-mode mla \
  --blocks 512 --layout layerwise --cache-capacity-gb 64 \
  --direct-io --cache-copy plain --cache-streams 4 --cache-staging-blocks 64 \
  --epochs 3 --warmup 1 --epoch-interval-ms 15 --verify

# V4：mmap + mallocHost 中转，DP4TP1，FA-only
python /workspace/unified-cache-management/ucm/store/test/e2e/cache_bw.py \
  --mode cache-hit --cache-core v2 --model dsv4 \
  --no-cache-numa-bind --dp-size 4 --tp-size 1 --worker-mode mla \
  --blocks 512 --cache-capacity-gb 64 \
  --direct-io --cache-copy plain --cache-streams 4 --cache-staging-blocks 64 \
  --epochs 3 --warmup 1 --epoch-interval-ms 15 --verify
```

先用 `--verify` 验证数据；正式性能对比时各组都去掉 `--verify`，避免校验影响下一层的调度。
和 mmap 直接 H2D 对比时，只去掉 `--cache-staging-blocks 64`，其余参数保持一致。
Posix 场景在上面的 Posix 命令中添加同一参数，存储路径和轮数不变。
也可把 64 改为 512，比较减小批次同步次数后的效果。

`cache_load_duration_ms` 包含 CPU memcpy、中转区复用等待和 H2D 的完整耗时。
`cache_load_staging_memcpy_ms` 为每个 shard 的 CPU 拷贝时间；
`cache_load_staging_wait_ms` 为批次填满后的 H2D 等待时间，
任务末尾的等待仍记录在 `cache_h2d_sync_ms`。
逐轮输出的 `staging_memcpy`、`staging_wait` 是本 rank 整次请求的对应耗时之和，
已包含在 `request_cost` 内。开启后脚本会核对 memcpy 样本数；旧库未重新编译、
中转参数没有生效时会报错，不会输出一组标成中转但实际走原路径的数据。

## 空间与结果

按上述 Posix 轮数，预埋数据量如下；两组按顺序运行，不要同时启动多个测试：

| 场景 | 数据量 | 含预留元数据空间的估算 |
| --- | --- | --- |
| V4 DP4TP1，任一路径，10 测试＋2 预热 | 72.94 GiB | 74.44 GiB |
| GLM5.2 mallocHost，5 测试＋1 预热 | 321.75 GiB | 323.25 GiB |
| GLM5.2 mmap 共享，5 测试＋1 预热 | 40.22 GiB | 40.41 GiB |

500G 按十进制 500,000,000,000 字节计算。脚本启动前检查数据量和元数据预留空间；
还会检查现有文件系统用量加上本轮估算是否超过 `--max-storage-gb`，以及剩余空间是否足够。
超出则在预埋前报错，不会先写满再失败。修改轮数后，先在命令末尾加 `--dry-run`
查看 `seed_storage_bytes` 和 `storage_estimate_with_metadata_bytes`。

脚本只创建和清理自己的 `ucm-cache-bw-*` 子目录，正常结束及可捕获的异常退出后会清理。
`--keep-data` 或强制杀进程可能保留数据，下一轮会把它计入已有用量，不会擅自删除。
Cache 命中不创建这批 Posix 预埋文件。

主结果来自 C++ `cache_load_duration_ms` 和字节计数，比较 `task_avg`、`task_p99~`、
`bw` 及各 rank 数据。`~` 是 histogram 桶估计；`request_cost` 为各层 task 耗时之和，
不是模型 TTFT。ALL 的带宽是合并字节/耗时后的每 rank 有效率，不是所有 NPU 的聚合带宽。
`--verify` 可在计时外校验数据；脚本不运行模型 forward。

参考：[Linux tmpfs](https://docs.kernel.org/filesystems/tmpfs.html)、
[tmpfs 对 O_DIRECT 的支持](https://git.zx2c4.com/linux-rng/commit/?id=e88e0d366f9cfbb810b0c8509dc5d130d5a53e02)。
