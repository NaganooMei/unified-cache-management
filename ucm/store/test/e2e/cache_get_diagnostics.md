# Cache2 Get diagnostics

This opt-in probe profiles load dispatch without changing owner selection,
preallocation, shard order, barriers, backend IO, or transfer streams.
It applies to Cache2 (`--cache-core v2`), including A2. Rebuild/reinstall the
native library as well as updating the Python scripts. A Python-only update
cannot produce these metrics.

## Run on A2

Use the server's normal build procedure with `PLATFORM=ascend` for A2. A typical
editable installation is:

```bash
PLATFORM=ascend python3 -m pip install --no-build-isolation -e .
```

Run from this checkout with the same installed native library. The following
uses GLM's layerwise MLA shape on eight ranks; it does not load model weights.
Change `/mnt/ucm-get-diagnostics` to the actual filesystem under investigation.
With these arguments the Posix seed payload is about 20.1 GiB, plus filesystem
overhead. It uses a fresh run-specific directory and fresh keys for every Posix
request; `--warmup` does not make measured requests Cache hits.

```bash
python3 ucm/store/test/e2e/cache_bw.py \
  --mode cache-hit --cache-core v2 --model glm-5.2 --workers 8 \
  --blocks 512 --cache-capacity-gb 32 --epochs 2 --warmup 1 \
  --get-diagnostics --verify | tee cache-get.log

python3 ucm/store/test/e2e/cache_bw.py \
  --mode posix-hit --cache-core v2 --model glm-5.2 --workers 8 \
  --blocks 512 --cache-capacity-gb 32 --epochs 2 --warmup 1 \
  --storage-backends /mnt/ucm-get-diagnostics \
  --posix-engine psync --no-direct-io \
  --get-diagnostics --verify | tee posix-get.log
```

Match copy mode, stream count, backend engine, direct IO, and CPU/NUMA settings
to the case being investigated. The explicit buffered psync example above is
only one IO configuration. Use the same filesystem and PageCache procedure
across repeated Posix comparisons; the script does not drop PageCache.

First check correctness with `--verify`. For performance comparison, repeat
without `--verify` on both modes: verification is outside each timed load but
can change rank arrival/progress for subsequent layers. Repeat both modes with
`--get-diagnostics` omitted to quantify instrumentation overhead. No new
barriers or artificial rank delays are introduced by the diagnostic option.

## Output and meaning

Every request prints per-rank diagnostic deltas; the final summary also merges
histogram buckets and counters across measured requests, excluding warmup.
`ALL` sums work across ranks; it is not end-to-end elapsed time. p99 values are
histogram estimates, with buckets starting at 0.001 ms (1 microsecond).

| Histogram | Observation |
| --- | --- |
| `cache_load_get_target_duration_ms` | One Get on the preferred rank, or on a private buffer; includes all acquisition retries |
| `cache_load_get_peer_duration_ms` | One Get on a non-target rank; includes waiting for publication/claim |
| `cache_load_backend_call_duration_ms` | One cold owned shard's synchronous backend Load submission, excluding Get and queue push |
| `cache_load_running_push_duration_ms` | One shard's running queue Push, including queue-full waiting |
| `cache_load_slot_ready_wait_ms` | Transfer state polling without a backend task handle; includes already-Ready hits and is separate from Get |

Roles use `myRank == preferredRank`, not the returned handle's `Owner()`.
Ready hits normally return non-owner handles on the preferred rank too.

The `cache_load_get_*_total` counters distinguish outer acquisition retries,
peer lookup misses, peer observations of `Loading/reference=0`, pin failures,
claimed-slot spins, bucket TryLock failures, CLOCK scans, allocation failures,
and claims of idle Loading slots. Preallocation counters distinguish created,
existing, lock failure, and allocation failure outcomes; non-target ranks that
skip Prealloc do not count as failures. CLOCK scan counters exclude Prealloc.

Retry counters describe observations, not durations; nested pin spins and outer
retries are not mutually exclusive categories. Idle Loading claims normally
indicate preallocation reuse, but these aggregate counters cannot prove the
history of a particular evicted key. Metrics alone also cannot reconstruct a
cross-rank per-key timeline or distinguish CPU descheduling from active work.

High peer Get time with many unclaimed observations supports waiting for the
selected rank to claim an existing slot. High target Get time with many CLOCK
scans/lock retries points toward allocation contention. Backend submission and
queue push can delay a rank reaching later Gets, even though they are outside
the individual Get measurement. Compare each rank separately before `ALL`.

Diagnostics use caller-local retry counters and task-local duration samples,
then publish at the end of dispatch and next-shard preallocation. The marker
`cache_load_diagnostics_tasks_total` is published last. The benchmark waits for
both normal load completion and this marker, checks the Get sample count, and
rejects missing instrumentation instead of reporting zero. Diagnostics may
perturb rank progress; the flag defaults to off.

## Service collection

Set `cache_load_diagnostics: true` in the Cache store configuration and restart
with the rebuilt library. The default metrics definitions and example YAML
include all new metrics. A custom metrics configuration must include them too.
Enable the existing metrics exporter as usual, then run:

```bash
python3 scripts/collect_ucm_metrics.py --label posix-get
```

The CacheStore report includes the new histograms and counter deltas by rank.
For counter rows the `Samples` column is the event delta and `Unit` is `events`;
Avg/P90/P99 are empty. Do not attach the service's destructive C++ delta reader
to the benchmark process: `cache_bw.py` already owns that reader directly.
