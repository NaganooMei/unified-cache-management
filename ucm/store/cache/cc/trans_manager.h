/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 * */
#ifndef UNIFIEDCACHE_CACHE_STORE_CC_TRANS_MANAGER_H
#define UNIFIEDCACHE_CACHE_STORE_CC_TRANS_MANAGER_H

#include "dump_queue.h"
#include "load_queue.h"
#include "logger/logger.h"
#include "metrics_api.h"
#include "template/task_wrapper.h"
#include "trans_task.h"

namespace UC::CacheStore {

class TransManager : public Detail::TaskWrapper<TransTask, Detail::TaskHandle> {
    size_t shardSize_;
    int32_t deviceId_{-1};
    size_t bufferRank_{0};
    LoadQueue loadQ_;
    DumpQueue dumpQ_;

public:
    Status Setup(const Config& config, Buffer* buffer)
    {
        timeoutMs_ = config.timeoutMs;
        shardSize_ = config.shardSize;
        deviceId_ = config.deviceId;
        bufferRank_ = config.EffectiveBufferRank();
        auto s = loadQ_.Setup(config, &failureSet_, buffer);
        if (s.Failure()) [[unlikely]] { return s; }
        return dumpQ_.Setup(config, &failureSet_, buffer);
    }

protected:
    Status FailureStatus(const TaskPtr& task) const override { return task->FailureStatus(); }
    void Cancel(TaskPtr task) override
    {
        if (task->type != TransTask::Type::LOAD) { return; }
        UC_ERROR(
            "CACHE_LOAD_DIAG task_timeout task={} brief={} device={} buffer_rank={} shards={} "
            "dispatch_phase={} dispatched={} dispatch_original={} dispatch_shard={} "
            "dispatch_block_hash={} preferred_segment={} transfer_phase={} transferred={} "
            "transfer_original={} transfer_shard={} transfer_block_hash={} segment={} "
            "global_slot={} backend_task={} slot_state={} pending_owners={}",
            task->id, task->desc.brief, deviceId_, bufferRank_, task->desc.size(),
            LoadDispatchPhaseName(task->loadDispatchPhase.load(std::memory_order_acquire)),
            task->dispatchedShards.load(std::memory_order_relaxed),
            task->dispatchOriginalIndex.load(std::memory_order_relaxed),
            task->dispatchShardIndex.load(std::memory_order_relaxed),
            task->dispatchBlockHash.load(std::memory_order_relaxed),
            task->dispatchPreferredSegment.load(std::memory_order_relaxed),
            LoadTransferPhaseName(task->loadTransferPhase.load(std::memory_order_acquire)),
            task->transferredShards.load(std::memory_order_relaxed),
            task->transferOriginalIndex.load(std::memory_order_relaxed),
            task->transferShardIndex.load(std::memory_order_relaxed),
            task->transferBlockHash.load(std::memory_order_relaxed),
            task->transferSegment.load(std::memory_order_relaxed),
            task->transferGlobalSlot.load(std::memory_order_relaxed),
            task->transferBackendTask.load(std::memory_order_relaxed),
            task->transferSlotState.load(std::memory_order_relaxed),
            task->pendingOwnerShards.load(std::memory_order_relaxed));
    }
    void Dispatch(TaskPtr t, WaiterPtr w) override
    {
        const auto id = t->id;
        const auto& brief = t->desc.brief;
        const auto num = t->desc.size();
        const auto size = shardSize_ * num;
        const auto tp = w->startTp;
        const auto isLoad = t->type == TransTask::Type::LOAD;
        UC_DEBUG("Cache task({},{},{},{}) dispatching.", id, brief, num, size);
        w->SetEpilog([id, brief = std::move(brief), num, size, tp, isLoad] {
            auto cost = NowTime::Now() - tp;
            auto costMs = cost * 1e3;
            auto bwGbps = cost > 0 ? static_cast<double>(size) / cost / 1e9 : 0.0;
            UC_DEBUG("Cache task({},{},{},{}) finished, cost {:.3f}ms.", id, brief, num, size,
                     costMs);
            if (isLoad) {
                UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_duration_ms"), costMs);
                UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_bandwidth_gbps"), bwGbps);
                UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_bytes_total"),
                                         static_cast<double>(size));
            } else {
                UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_duration_ms"), costMs);
                UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_bandwidth_gbps"), bwGbps);
                UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_shards_total"),
                                         static_cast<double>(num));
                UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_bytes_total"),
                                         static_cast<double>(size));
            }
        });
        if (t->type == TransTask::Type::LOAD) {
            loadQ_.Submit(t, w);
        } else {
            dumpQ_.Submit(t, w);
        }
    }
};

}  // namespace UC::CacheStore

#endif
