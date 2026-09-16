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
#include "load_queue.h"
#include "logger/logger.h"
#include "metrics_api.h"
#include "thread/cpu_affinity.h"

namespace UC::CacheStore {

namespace {
constexpr double kSlowLoadPhaseMs = 100.0;
constexpr double kPeerWaitReportIntervalSeconds = 2.0;

size_t BlockHash(const Detail::BlockId& block) { return Detail::BlockIdHasher{}(block); }
}  // namespace

LoadQueue::~LoadQueue()
{
    stop_.store(true);
    if (dispatcher_.joinable()) { dispatcher_.join(); }
    if (transfer_.joinable()) { transfer_.join(); }
}

Status LoadQueue::Setup(const Config& config, TaskIdSet* failureSet, Buffer* buffer)
{
    failureSet_ = failureSet;
    buffer_ = buffer;
    backend_ = config.storeBackend;
    deviceId_ = config.deviceId;
    tensorSizes_ = config.tensorSizes;
    nShardPerBlock_ = config.blockSize / config.shardSize;
    streamNumber_ = config.EffectiveStreamNumber();
    useGdr_ = config.useGdr;
    cacheIOAggregation_ = config.cacheIOAggregation;
    cacheSdmaDirect_ = config.cacheSdmaDirect;
    shared_ = config.shareBufferEnable;
    cpuAffinityCores_ = config.cpuAffinityCores;
    segmentCount_ = buffer_->NumRanks();
    bufferRank_ = shared_ ? config.EffectiveBufferRank() : 0;
    stripeAcrossSegments_ = shared_ && config.localRankSize > 1;
    waiting_.Setup(config.waitingQueueDepth);
    running_.Setup(config.runningQueueDepth);
    holder_.reserve(1024);
    dispatcher_ = std::thread{&LoadQueue::DispatchStage, this};
    std::promise<Status> started;
    auto fut = started.get_future();
    transfer_ = std::thread{&LoadQueue::TransferStage, this, std::ref(started)};
    return fut.get();
}

void LoadQueue::Submit(TaskPtr task, WaiterPtr waiter)
{
    waiter->Up();
    auto success = waiting_.TryPush({task, waiter});
    if (success) { return; }
    UC_ERROR("Waiting queue full, submit load task({}) failed.", task->id);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_queue_full_total"), 1.0);
    RecordFailedShards(task->desc.size());
    failureSet_->Insert(task->id);
    waiter->Done();
}

void LoadQueue::DispatchStage()
{
    auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_load_disp");
    if (nameStatus.Failure()) {
        UC_WARN("Failed({}) to set UCM load dispatcher name.", nameStatus);
    }
    if (!cpuAffinityCores_.empty()) {
        auto s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
        if (s.Failure()) { UC_WARN("Failed({}) to set affinity.", s); }
    }
    waiting_.ConsumerLoop(stop_, &LoadQueue::DispatchOneTask, this);
}

static std::vector<size_t> RearrangeIndex(size_t n, size_t iProc, size_t nProc)
{
    std::vector<size_t> order;
    order.reserve(n);
    for (size_t r = 0; r < nProc; ++r) {
        size_t slice = (iProc + r) % nProc;
        for (size_t j = 0;; ++j) {
            size_t i = slice + j * nProc;
            if (i >= n) { break; }
            order.push_back(i);
        }
    }
    return order;
}

void LoadQueue::DispatchOneTask(TaskPair&& pair)
{
    auto& task = pair.first;
    auto& waiter = pair.second;
    if (failureSet_->Contains(task->id)) {
        waiter->Done();
        return;
    }
    auto tp = waiter->startTp;
    auto tpWait = NowTime::Now();
    const auto nShard = task->desc.size();
    size_t backendSubmitCount = 0;
    size_t waitShardCount = 0;
    const auto indexes =
        RearrangeIndex(nShard, bufferRank_, stripeAcrossSegments_ ? segmentCount_ : 1);
    struct PreallocHint {
        Detail::BlockId block;
        size_t shard;
        size_t segment;
    };
    std::vector<PreallocHint> preallocHints;
    preallocHints.reserve(nShard);
    for (size_t i = 0; i < nShard; i++) {
        const auto originalIndex = indexes[i];
        auto& shard = task->desc[originalIndex];
        ShardTask shardTask;
        shardTask.originalIndex = originalIndex;
        const auto preferredSegment =
            shared_ ? (stripeAcrossSegments_ ? originalIndex % segmentCount_ : bufferRank_)
                    : kInvalidIndex;
        const auto blockHash = BlockHash(shard.owner);
        task->loadDispatchPhase.store(LoadDispatchPhase::BufferGet, std::memory_order_relaxed);
        task->dispatchOriginalIndex.store(originalIndex, std::memory_order_relaxed);
        task->dispatchShardIndex.store(shard.index, std::memory_order_relaxed);
        task->dispatchPreferredSegment.store(preferredSegment, std::memory_order_relaxed);
        task->dispatchBlockHash.store(blockHash, std::memory_order_relaxed);
        auto tpGet = NowTime::Now();
        shardTask.bufferHandle = buffer_->Get(shard.owner, shard.index, true, preferredSegment);
        auto getMs = (NowTime::Now() - tpGet) * 1e3;
        if (!shardTask.bufferHandle) {
            task->loadDispatchPhase.store(LoadDispatchPhase::Failed, std::memory_order_relaxed);
            UC_ERROR(
                "CACHE_LOAD_DIAG buffer_get_failed task={} brief={} device={} buffer_rank={} "
                "dispatch={}/{} original={} shard={} block_hash={} preferred_segment={} "
                "cost={:.3f}ms",
                task->id, task->desc.brief, deviceId_, bufferRank_, i, nShard, originalIndex,
                shard.index, blockHash, preferredSegment, getMs);
            task->Fail(Status::Retry());
            failureSet_->Insert(task->id);
            waiter->Done();
            return;
        }
        if (getMs >= kSlowLoadPhaseMs) {
            UC_WARN(
                "CACHE_LOAD_DIAG slow_buffer_get task={} brief={} device={} buffer_rank={} "
                "dispatch={}/{} original={} shard={} block_hash={} preferred_segment={} "
                "actual_segment={} global_slot={} owner={} state={} references={} cost={:.3f}ms",
                task->id, task->desc.brief, deviceId_, bufferRank_, i, nShard, originalIndex,
                shard.index, blockHash, preferredSegment, shardTask.bufferHandle.Segment(),
                shardTask.bufferHandle.GlobalSlot(), shardTask.bufferHandle.Owner(),
                static_cast<int>(shardTask.bufferHandle.GetState()),
                shardTask.bufferHandle.ReferenceCount(), getMs);
        }
        shardTask.backendTaskHandle = 0;
        shardTask.fromPosix = !shardTask.bufferHandle.Ready();
        if (shardTask.fromPosix) { waitShardCount++; }
        if (shardTask.bufferHandle.Owner() && !shardTask.bufferHandle.Ready()) {
            if (shardTask.bufferHandle.Data() == nullptr) {
                shardTask.bufferHandle.MarkFailed();
                task->Fail(Status::Error("cache host mapping unavailable"));
                failureSet_->Insert(task->id);
                waiter->Done();
                return;
            }
            Detail::TaskDesc backendTask{
                Detail::Shard{shard.owner, shard.index, {shardTask.bufferHandle.Data()}}
            };
            backendTask.brief = "Backend2Cache";
            task->loadDispatchPhase.store(LoadDispatchPhase::BackendSubmit,
                                          std::memory_order_relaxed);
            auto tpBackendSubmit = NowTime::Now();
            auto res = backend_->Load(std::move(backendTask));
            auto backendSubmitMs = (NowTime::Now() - tpBackendSubmit) * 1e3;
            if (!res) [[unlikely]] {
                UC_ERROR("Failed({}) to submit load task({}) to backend.", res.Error(), task->id);
                UC::Metrics::UpdateStats(
                    NAME_TO_METRIC_ID("cache_backend_load_submit_errors_total"), 1.0);
                RecordLoadSourceShards(i + 1, waitShardCount);
                RecordFailedShards(nShard - i);
                shardTask.bufferHandle.MarkFailed();
                task->Fail(res.Error());
                failureSet_->Insert(task->id);
                waiter->Done();
                return;
            }
            shardTask.backendTaskHandle = res.Value();
            task->pendingOwnerShards.fetch_add(1, std::memory_order_relaxed);
            backendSubmitCount++;
            if (backendSubmitMs >= kSlowLoadPhaseMs) {
                UC_WARN(
                    "CACHE_LOAD_DIAG slow_backend_submit task={} brief={} device={} buffer_rank={} "
                    "dispatch={}/{} original={} shard={} block_hash={} segment={} global_slot={} "
                    "backend_task={} pending_owners={} cost={:.3f}ms",
                    task->id, task->desc.brief, deviceId_, bufferRank_, i, nShard, originalIndex,
                    shard.index, blockHash, shardTask.bufferHandle.Segment(),
                    shardTask.bufferHandle.GlobalSlot(), shardTask.backendTaskHandle,
                    task->pendingOwnerShards.load(std::memory_order_relaxed), backendSubmitMs);
            }
        }
        if (shard.index + 1 != nShardPerBlock_) {
            preallocHints.push_back(
                {shard.owner, shard.index + 1, shardTask.bufferHandle.Segment()});
        }
        shardTask.task = task;
        shardTask.shard = std::move(shard);
        shardTask.waiter = (i + 1 < nShard) ? nullptr : waiter;
        task->loadDispatchPhase.store(LoadDispatchPhase::RunningQueue, std::memory_order_relaxed);
        auto tpRunningPush = NowTime::Now();
        running_.Push(std::move(shardTask));
        auto runningPushMs = (NowTime::Now() - tpRunningPush) * 1e3;
        task->dispatchedShards.store(i + 1, std::memory_order_relaxed);
        if (runningPushMs >= kSlowLoadPhaseMs) {
            UC_WARN(
                "CACHE_LOAD_DIAG slow_running_push task={} brief={} device={} buffer_rank={} "
                "dispatch={}/{} original={} shard={} block_hash={} preferred_segment={} "
                "pending_owners={} cost={:.3f}ms",
                task->id, task->desc.brief, deviceId_, bufferRank_, i, nShard, originalIndex,
                task->dispatchShardIndex.load(std::memory_order_relaxed), blockHash,
                preferredSegment, task->pendingOwnerShards.load(std::memory_order_relaxed),
                runningPushMs);
    }
    auto tpDispatch = NowTime::Now();
    task->loadDispatchPhase.store(LoadDispatchPhase::Prealloc, std::memory_order_relaxed);
    auto tpPrealloc = NowTime::Now();
    for (const auto& hint : preallocHints) {
        buffer_->Prealloc(hint.block, hint.shard, true, hint.segment);
    }
    auto preallocMs = (NowTime::Now() - tpPrealloc) * 1e3;
    task->loadDispatchPhase.store(LoadDispatchPhase::Done, std::memory_order_release);
    if (preallocMs >= kSlowLoadPhaseMs) {
        UC_WARN(
            "CACHE_LOAD_DIAG slow_prealloc task={} brief={} device={} buffer_rank={} hints={} "
            "dispatched={}/{} pending_owners={} cost={:.3f}ms",
            task->id, task->desc.brief, deviceId_, bufferRank_, preallocHints.size(),
            task->dispatchedShards.load(std::memory_order_relaxed), nShard,
            task->pendingOwnerShards.load(std::memory_order_relaxed), preallocMs);
    }
    UC_DEBUG("Cache task({}) dispatch shards({}), wait={:.3f}ms, cost={:.3f}ms.", task->id, nShard,
             (tpWait - tp) * 1e3, (tpDispatch - tpWait) * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_queue_wait_duration_ms"),
                             (tpWait - tp) * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_backend_submit_duration_ms"),
                             (tpDispatch - tpWait) * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_backend_shards_total"),
                             static_cast<double>(backendSubmitCount));
    RecordLoadSourceShards(nShard, waitShardCount);
}

void LoadQueue::TransferStage(std::promise<Status>& started)
{
    auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_load_xfer");
    if (nameStatus.Failure()) { UC_WARN("Failed({}) to set UCM load transfer name.", nameStatus); }
    CopyStream stream;
    auto s = Status::OK();
    if (cacheIOAggregation_) {
        s = stream.SetupIoAggregation(deviceId_, useGdr_);
    } else if (cacheSdmaDirect_) {
        s = stream.SetupSdmaDirect(deviceId_, streamNumber_, useGdr_);
    } else {
        s = stream.Setup(deviceId_, streamNumber_, useGdr_);
    }
    started.set_value(s);
    if (s.Failure()) [[unlikely]] { return; }
    if (!cpuAffinityCores_.empty()) {
        s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
        if (s.Failure()) { UC_WARN("Failed({}) to set affinity.", s); }
    }
    running_.ConsumerLoop(stop_, &LoadQueue::TransferOneTask, this, stream);
}

void LoadQueue::TransferOneTask(CopyStream& stream, ShardTask&& task)
{
    auto parentTask = task.task;
    const auto taskHandle = parentTask->id;
    if (failureSet_->Contains(taskHandle)) {
        // Backend writes must finish before their pinned host storage is released.
        if (task.backendTaskHandle != 0) { WaitBackendTaskReady(task); }
        RecordFailedShards(1);
        if (task.waiter) {
            stream.Synchronize();
            holder_.clear();
            task.waiter->Done();
        }
        return;
    }

    auto s = Status::OK();
    auto waiter = task.waiter;
    parentTask->transferOriginalIndex.store(task.originalIndex, std::memory_order_relaxed);
    parentTask->transferShardIndex.store(task.shard.index, std::memory_order_relaxed);
    parentTask->transferGlobalSlot.store(task.bufferHandle.GlobalSlot(), std::memory_order_relaxed);
    parentTask->transferSegment.store(task.bufferHandle.Segment(), std::memory_order_relaxed);
    parentTask->transferBlockHash.store(BlockHash(task.shard.owner), std::memory_order_relaxed);
    parentTask->transferBackendTask.store(task.backendTaskHandle, std::memory_order_relaxed);
    parentTask->transferSlotState.store(static_cast<int32_t>(task.bufferHandle.GetState()),
                                        std::memory_order_relaxed);
    do {
        auto tpBackendWait = NowTime::Now();
        parentTask->loadTransferPhase.store(
            task.backendTaskHandle != 0 ? LoadTransferPhase::OwnerBackendWait
                                        : LoadTransferPhase::PeerReadyWait,
            std::memory_order_release);
        s = WaitBackendTaskReady(task);
        if (task.backendTaskHandle != 0) {
            parentTask->pendingOwnerShards.fetch_sub(1, std::memory_order_relaxed);
        }
        if (s.Failure()) [[unlikely]] {
            parentTask->loadTransferPhase.store(LoadTransferPhase::Failed,
                                                std::memory_order_relaxed);
            RecordShardResults(holder_, &task, false);
            break;
        }
        auto tpBackendReady = NowTime::Now();
        parentTask->transferSlotState.store(static_cast<int32_t>(task.bufferHandle.GetState()),
                                            std::memory_order_relaxed);
        auto backendWaitMs = (tpBackendReady - tpBackendWait) * 1e3;
        if (backendWaitMs >= kSlowLoadPhaseMs) {
            UC_WARN(
                "CACHE_LOAD_DIAG slow_ready_wait task={} brief={} device={} buffer_rank={} "
                "kind={} original={} shard={} block_hash={} segment={} global_slot={} "
                "backend_task={} state={} references={} dispatch_phase={} dispatched={}/{} "
                "pending_owners={} cost={:.3f}ms",
                taskHandle, parentTask->desc.brief, deviceId_, bufferRank_,
                task.backendTaskHandle != 0 ? "owner" : "peer", task.originalIndex,
                task.shard.index, parentTask->transferBlockHash.load(std::memory_order_relaxed),
                task.bufferHandle.Segment(), task.bufferHandle.GlobalSlot(),
                task.backendTaskHandle, static_cast<int>(task.bufferHandle.GetState()),
                task.bufferHandle.ReferenceCount(),
                LoadDispatchPhaseName(parentTask->loadDispatchPhase.load(std::memory_order_acquire)),
                parentTask->dispatchedShards.load(std::memory_order_relaxed),
                parentTask->desc.size(),
                parentTask->pendingOwnerShards.load(std::memory_order_relaxed), backendWaitMs);
        }
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_shard_backend_wait_ms"),
                                 backendWaitMs);

        auto* host = cacheSdmaDirect_ ? task.bufferHandle.DeviceData() : task.bufferHandle.Data();
        if (host == nullptr) {
            s = Status::Error("cache transfer mapping unavailable");
            break;
        }
        parentTask->loadTransferPhase.store(LoadTransferPhase::H2dSubmit,
                                            std::memory_order_release);
        s = HostToDeviceAsync(stream, host, task.shard.addrs.data());
        auto tpH2dSubmitted = NowTime::Now();
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to do H2D for task({}).", s, taskHandle);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_errors_total"), 1.0);
            RecordShardResults(holder_, &task, false);
            break;
        }
        auto h2dSubmitMs = (tpH2dSubmitted - tpBackendReady) * 1e3;
        parentTask->transferredShards.fetch_add(1, std::memory_order_relaxed);
        if (h2dSubmitMs >= kSlowLoadPhaseMs) {
            UC_WARN(
                "CACHE_LOAD_DIAG slow_h2d_submit task={} brief={} device={} buffer_rank={} "
                "original={} shard={} block_hash={} segment={} global_slot={} submitted={}/{} "
                "cost={:.3f}ms",
                taskHandle, parentTask->desc.brief, deviceId_, bufferRank_, task.originalIndex,
                task.shard.index, parentTask->transferBlockHash.load(std::memory_order_relaxed),
                task.bufferHandle.Segment(), task.bufferHandle.GlobalSlot(),
                parentTask->transferredShards.load(std::memory_order_relaxed),
                parentTask->desc.size(), h2dSubmitMs);
        }
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_submit_ms"),
                                 h2dSubmitMs);
        if (!waiter) {
            parentTask->loadTransferPhase.store(LoadTransferPhase::Idle,
                                                std::memory_order_relaxed);
            holder_.push_back(std::move(task));
            return;
        }
        auto tpH2dSyncStart = NowTime::Now();
        parentTask->loadTransferPhase.store(LoadTransferPhase::H2dSync,
                                            std::memory_order_release);
        s = stream.Synchronize();
        auto h2dSyncMs = (NowTime::Now() - tpH2dSyncStart) * 1e3;
        if (h2dSyncMs >= kSlowLoadPhaseMs) {
            UC_WARN(
                "CACHE_LOAD_DIAG slow_h2d_sync task={} brief={} device={} buffer_rank={} "
                "submitted={}/{} held={} pending_owners={} cost={:.3f}ms status={}",
                taskHandle, parentTask->desc.brief, deviceId_, bufferRank_,
                parentTask->transferredShards.load(std::memory_order_relaxed),
                parentTask->desc.size(), holder_.size() + 1,
                parentTask->pendingOwnerShards.load(std::memory_order_relaxed), h2dSyncMs, s);
        }
        RecordH2dSyncMetrics(h2dSyncMs);
        RecordShardResults(holder_, &task, s.Success());
        holder_.clear();
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to sync on stream for task({}).", s, taskHandle);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_errors_total"), 1.0);
            break;
        }
        parentTask->loadTransferPhase.store(LoadTransferPhase::Done, std::memory_order_release);
    } while (0);
    if (s.Failure()) [[unlikely]] {
        parentTask->loadTransferPhase.store(LoadTransferPhase::Failed, std::memory_order_relaxed);
        stream.Synchronize();
        holder_.clear();
        parentTask->Fail(s);
        failureSet_->Insert(taskHandle);
    }
    if (waiter) { waiter->Done(); }
}

Status LoadQueue::WaitBackendTaskReady(ShardTask& task)
{
    if (task.backendTaskHandle != 0) {
        auto s = backend_->Wait(task.backendTaskHandle);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to wait backend({}) for task({}).", s, task.backendTaskHandle,
                     task.task->id);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_backend_load_wait_errors_total"),
                                     1.0);
            task.bufferHandle.MarkFailed();
            return s;
        }
        task.bufferHandle.MarkReady();
        return Status::OK();
    }
    auto tpPeerWait = NowTime::Now();
    auto nextReport = tpPeerWait + kPeerWaitReportIntervalSeconds;
    for (;;) {
        auto state = task.bufferHandle.GetState();
        task.task->transferSlotState.store(static_cast<int32_t>(state),
                                           std::memory_order_relaxed);
        if (state == State::Ready) { return Status::OK(); }
        if (state == State::Failed) { return Status::Retry(); }
        if (failureSet_->Contains(task.task->id)) { return task.task->FailureStatus(); }
        auto now = NowTime::Now();
        if (now >= nextReport) {
            UC_WARN(
                "CACHE_LOAD_DIAG peer_stall task={} brief={} device={} buffer_rank={} "
                "original={} shard={} block_hash={} segment={} global_slot={} state={} "
                "references={} dispatch_phase={} dispatched={}/{} transferred={} "
                "pending_owners={} waited={:.3f}ms",
                task.task->id, task.task->desc.brief, deviceId_, bufferRank_, task.originalIndex,
                task.shard.index, BlockHash(task.shard.owner), task.bufferHandle.Segment(),
                task.bufferHandle.GlobalSlot(), static_cast<int>(state),
                task.bufferHandle.ReferenceCount(),
                LoadDispatchPhaseName(task.task->loadDispatchPhase.load(std::memory_order_acquire)),
                task.task->dispatchedShards.load(std::memory_order_relaxed),
                task.task->desc.size(),
                task.task->transferredShards.load(std::memory_order_relaxed),
                task.task->pendingOwnerShards.load(std::memory_order_relaxed),
                (now - tpPeerWait) * 1e3);
            nextReport = now + kPeerWaitReportIntervalSeconds;
        }
        std::this_thread::yield();
    }
}

Status LoadQueue::HostToDeviceAsync(CopyStream& stream, void* host, void** device)
{
    return stream.HostToDeviceAsync(host, device, tensorSizes_);
}

void LoadQueue::RecordShardResults(const std::vector<ShardTask>& tasks, const ShardTask* extra,
                                   bool success) const
{
    size_t cache = 0;
    size_t posix = 0;
    for (const auto& task : tasks) {
        if (task.fromPosix) {
            ++posix;
        } else {
            ++cache;
        }
    }
    if (extra != nullptr) {
        if (extra->fromPosix) {
            ++posix;
        } else {
            ++cache;
        }
    }
    if (!success) {
        RecordFailedShards(cache + posix);
        return;
    }
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_success_shards_total"),
                             static_cast<double>(cache));
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_posix_load_success_shards_total"),
                             static_cast<double>(posix));
}

void LoadQueue::RecordFailedShards(size_t count) const
{
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_failed_shards_total"),
                             static_cast<double>(count));
}

void LoadQueue::RecordLoadSourceShards(size_t total, size_t wait) const
{
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_shards_total"),
                             static_cast<double>(total));
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_wait_shards_total"),
                             static_cast<double>(wait));
}

void LoadQueue::RecordH2dSyncMetrics(double h2dSyncMs) const
{
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_sync_ms"), h2dSyncMs);
}

}  // namespace UC::CacheStore
