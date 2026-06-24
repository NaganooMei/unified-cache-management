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
#include <algorithm>
#include "logger/logger.h"
#include "metrics_api.h"
#include "thread/cpu_affinity.h"

namespace UC::CacheStore {

LoadQueue::~LoadQueue()
{
    stop_.store(true);
    if (dispatcher_.joinable()) { dispatcher_.join(); }
    if (transfer_.joinable()) { transfer_.join(); }
}

Status LoadQueue::Setup(const Config& config, TaskIdSet* failureSet, TransBuffer* buffer)
{
    failureSet_ = failureSet;
    buffer_ = buffer;
    backend_ = config.storeBackend;
    deviceId_ = config.deviceId;
    tensorSizes_ = config.tensorSizes;
    streamNumber_ = config.streamNumber;
    useGdr_ = config.useGdr;
    cacheSdmaDirect_ = config.cacheSdmaDirect;
    sdmaDirectMaxReadyLanes_ = config.sdmaDirectMaxReadyLanes;
    sdmaDirectLaunchGranularity_ = config.sdmaDirectLaunchGranularity;
    cpuAffinityCores_ = config.cpuAffinityCores;
    waiting_.Setup(config.waitingQueueDepth);
    running_.Setup(config.runningQueueDepth);
    holder_.reserve(1024);
    sdmaDirectBatchHolder_.reserve(kSdmaDirectLaunchBatchSize);
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
    failureSet_->Insert(task->id);
    waiter->Done();
}

void LoadQueue::DispatchStage()
{
    if (!cpuAffinityCores_.empty()) {
        auto s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
        if (s.Failure()) { UC_WARN("Failed({}) to set affinity.", s); }
    }
    waiting_.ConsumerLoop(stop_, &LoadQueue::DispatchOneTask, this);
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
    for (size_t i = 0; i < nShard; i++) {
        auto& shard = task->desc[i];
        ShardTask shardTask;
        shardTask.bufferHandle = buffer_->Get(shard.owner, shard.index, true, true);
        shardTask.backendTaskHandle = 0;
        if (shardTask.bufferHandle.Owner() && !shardTask.bufferHandle.Ready()) {
            Detail::TaskDesc backendTask{
                Detail::Shard{shard.owner, shard.index, {shardTask.bufferHandle.Data()}}
            };
            backendTask.brief = "Backend2Cache";
            auto res = backend_->Load(std::move(backendTask));
            if (!res) [[unlikely]] {
                UC_ERROR("Failed({}) to submit load task({}) to backend.", res.Error(), task->id);
                UC::Metrics::UpdateStats(
                    NAME_TO_METRIC_ID("cache_backend_load_submit_errors_total"), 1.0);
                failureSet_->Insert(task->id);
                waiter->Done();
                return;
            }
            shardTask.backendTaskHandle = res.Value();
            backendSubmitCount++;
        }
        shardTask.taskHandle = task->id;
        shardTask.shard = std::move(shard);
        shardTask.waiter = (i + 1 < nShard) ? nullptr : waiter;
        running_.Push(std::move(shardTask));
    }
    auto tpDispatch = NowTime::Now();
    auto dispatchMs = (tpDispatch - tpWait) * 1e3;
    auto queueWaitMs = (tpWait - tp) * 1e3;
    UC_DEBUG("Cache task({}) dispatch shards({}), wait={:.3f}ms, cost={:.3f}ms.", task->id, nShard,
             queueWaitMs, dispatchMs);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_queue_wait_duration_ms"), queueWaitMs);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_dispatch_duration_ms"), dispatchMs);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_backend_shards_total"),
                             static_cast<double>(backendSubmitCount));
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_shards_total"),
                             static_cast<double>(nShard));
}

void LoadQueue::TransferStage(std::promise<Status>& started)
{
    CopyStream stream;
    auto s = cacheSdmaDirect_
                 ? stream.SetupSdmaDirect(deviceId_, streamNumber_, useGdr_,
                                          sdmaDirectMaxReadyLanes_)
                 : stream.Setup(deviceId_, streamNumber_, useGdr_);
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
    if (!pipelineTrace_.active || pipelineTrace_.taskHandle != task.taskHandle) {
        ResetPipelineTrace(task.taskHandle);
    }
    if (failureSet_->Contains(task.taskHandle)) {
        if (task.waiter) {
            ClearSdmaDirectHolders();
            task.waiter->Done();
        }
        return;
    }
    auto s = Status::OK();
    const auto taskHandle = task.taskHandle;
    auto waiter = task.waiter;
    do {
        auto tpBackendWait = NowTime::Now();
        s = WaitBackendTaskReady(task);
        if (s.Failure()) [[unlikely]] { break; }
        auto tpBackendReady = NowTime::Now();
        task.backendReadyTp = tpBackendReady;
        RecordBackendWait(task, tpBackendReady);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_shard_backend_wait_ms"),
                                 (tpBackendReady - tpBackendWait) * 1e3);
        if (UseSdmaDirectTaskLaunch()) {
            holder_.push_back(std::move(task));
            if (!waiter) { return; }
            auto tpH2dSubmitStart = NowTime::Now();
            s = HostToDeviceTaskAsync(stream, holder_);
            auto tpH2dSubmitted = NowTime::Now();
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to do H2D for task({}).", s, taskHandle);
                UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_errors_total"), 1.0);
                break;
            }
            RecordH2dLaunch(tpH2dSubmitStart);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_shard_h2d_ms"),
                                     (tpH2dSubmitted - tpBackendReady) * 1e3);
            auto tpSyncStart = NowTime::Now();
            s = stream.Synchronize();
            auto tpSyncEnd = NowTime::Now();
            ClearSdmaDirectHolders();
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to sync on stream for task({}).", s, taskHandle);
                UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_errors_total"), 1.0);
                break;
            }
            RecordLoadSummary(tpSyncStart, tpSyncEnd);
            waiter->Done();
            return;
        }
        if (UseSdmaDirectBatchLaunch()) {
            sdmaDirectBatchHolder_.push_back(std::move(task));
            const bool shouldFlush =
                waiter || sdmaDirectBatchHolder_.size() >= kSdmaDirectLaunchBatchSize;
            if (!shouldFlush) { return; }
            auto tpH2dSubmitStart = NowTime::Now();
            s = FlushSdmaDirectBatch(stream);
            auto tpH2dSubmitted = NowTime::Now();
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to do H2D for task({}).", s, taskHandle);
                UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_errors_total"), 1.0);
                break;
            }
            RecordH2dLaunch(tpH2dSubmitStart);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_shard_h2d_ms"),
                                     (tpH2dSubmitted - tpBackendReady) * 1e3);
            if (!waiter) { return; }
            auto tpSyncStart = NowTime::Now();
            s = stream.Synchronize();
            auto tpSyncEnd = NowTime::Now();
            ClearSdmaDirectHolders();
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to sync on stream for task({}).", s, taskHandle);
                UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_errors_total"), 1.0);
                break;
            }
            RecordLoadSummary(tpSyncStart, tpSyncEnd);
            waiter->Done();
            return;
        }
        auto* host = cacheSdmaDirect_ ? task.bufferHandle.DeviceData() : task.bufferHandle.Data();
        auto tpH2dSubmitStart = NowTime::Now();
        s = HostToDeviceAsync(stream, host, task.shard.addrs.data());
        auto tpH2dSubmitted = NowTime::Now();
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to do H2D for task({}).", s, task.taskHandle);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_errors_total"), 1.0);
            break;
        }
        RecordH2dLaunch(tpH2dSubmitStart);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_shard_h2d_ms"),
                                 (tpH2dSubmitted - tpBackendReady) * 1e3);
        if (!task.waiter) {
            holder_.push_back(std::move(task));
            return;
        }
        auto tpSyncStart = NowTime::Now();
        s = stream.Synchronize();
        auto tpSyncEnd = NowTime::Now();
        holder_.clear();
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to sync on stream for task({}).", s, task.taskHandle);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_errors_total"), 1.0);
            break;
        }
        RecordLoadSummary(tpSyncStart, tpSyncEnd);
    } while (0);
    if (s.Failure()) [[unlikely]] {
        if (UseSdmaDirectBatchLaunch() && !holder_.empty()) {
            auto syncStatus = stream.Synchronize();
            if (syncStatus.Failure()) {
                UC_ERROR("Failed({}) to sync in-flight H2D for task({}).", syncStatus,
                         taskHandle);
            }
        }
        failureSet_->Insert(taskHandle);
    }
    if (UseSdmaDirectTaskLaunch() || UseSdmaDirectBatchLaunch()) {
        ClearSdmaDirectHolders();
    }
    if (waiter) { waiter->Done(); }
}

Status LoadQueue::WaitBackendTaskReady(ShardTask& task)
{
    if (task.backendTaskHandle != 0) {
        auto s = backend_->Wait(task.backendTaskHandle);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to wait backend({}) for task({}).", s, task.backendTaskHandle,
                     task.taskHandle);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_backend_load_wait_errors_total"),
                                     1.0);
            return s;
        }
        task.bufferHandle.MarkReady();
        return Status::OK();
    }
    while (!task.bufferHandle.Ready()) {
        if (failureSet_->Contains(task.taskHandle)) { return Status::Error(); }
        std::this_thread::yield();
    }
    return Status::OK();
}

Status LoadQueue::HostToDeviceAsync(CopyStream& stream, void* host, void** device)
{
    return stream.HostToDeviceAsync(host, device, tensorSizes_);
}

Status LoadQueue::HostToDeviceTaskAsync(CopyStream& stream, std::vector<ShardTask>& tasks)
{
    std::vector<void*> hosts;
    std::vector<void**> devices;
    hosts.reserve(tasks.size());
    devices.reserve(tasks.size());
    for (auto& task : tasks) {
        hosts.push_back(cacheSdmaDirect_ ? task.bufferHandle.DeviceData()
                                         : task.bufferHandle.Data());
        devices.push_back(task.shard.addrs.data());
    }
    return stream.HostToDeviceAsync(hosts, devices, tensorSizes_);
}

Status LoadQueue::FlushSdmaDirectBatch(CopyStream& stream)
{
    if (sdmaDirectBatchHolder_.empty()) { return Status::OK(); }
    auto s = HostToDeviceTaskAsync(stream, sdmaDirectBatchHolder_);
    if (s.Failure()) [[unlikely]] { return s; }
    for (auto& task : sdmaDirectBatchHolder_) { holder_.push_back(std::move(task)); }
    sdmaDirectBatchHolder_.clear();
    return Status::OK();
}

void LoadQueue::ClearSdmaDirectHolders() noexcept
{
    holder_.clear();
    sdmaDirectBatchHolder_.clear();
}

void LoadQueue::ResetPipelineTrace(Detail::TaskHandle taskHandle)
{
    pipelineTrace_ = {};
    pipelineTrace_.active = true;
    pipelineTrace_.taskHandle = taskHandle;
    pipelineTrace_.startTp = NowTime::Now();
    pipelineTrace_.backendLoadReadyTps.reserve(1024);
}

void LoadQueue::RecordBackendWait(const ShardTask& task, double readyTp)
{
    const auto isBackendLoad = task.backendTaskHandle != 0;
    pipelineTrace_.shardCount++;
    if (isBackendLoad) {
        pipelineTrace_.backendLoadShardCount++;
        pipelineTrace_.backendLoadReadyTps.push_back(readyTp);
    } else {
        pipelineTrace_.cacheBufferShardCount++;
    }
}

void LoadQueue::RecordH2dLaunch(double submitStartTp)
{
    if (pipelineTrace_.firstH2dSubmitStartTp <= 0.0) {
        pipelineTrace_.firstH2dSubmitStartTp = submitStartTp;
    }
    pipelineTrace_.lastH2dSubmitStartTp = submitStartTp;
}

void LoadQueue::RecordLoadSummary(double syncStartTp, double syncEndTp)
{
    const auto totalMs = (syncEndTp - pipelineTrace_.startTp) * 1e3;
    const auto tailSyncMs = (syncEndTp - syncStartTp) * 1e3;
    const auto firstH2dTp = pipelineTrace_.firstH2dSubmitStartTp;
    const auto lastH2dTp = pipelineTrace_.lastH2dSubmitStartTp;
    const auto s2hDoneBeforeFirstH2d = firstH2dTp > 0.0
                                           ? static_cast<size_t>(std::count_if(
                                                 pipelineTrace_.backendLoadReadyTps.begin(),
                                                 pipelineTrace_.backendLoadReadyTps.end(),
                                                 [firstH2dTp](double readyTp) {
                                                     return readyTp <= firstH2dTp;
                                                 }))
                                           : 0;
    const auto s2hDoneBeforeLastH2d =
        lastH2dTp > 0.0 ? static_cast<size_t>(std::count_if(
                               pipelineTrace_.backendLoadReadyTps.begin(),
                               pipelineTrace_.backendLoadReadyTps.end(),
                               [lastH2dTp](double readyTp) { return readyTp <= lastH2dTp; }))
                         : 0;
    layerSummary_.taskCount++;
    layerSummary_.shardCount += pipelineTrace_.shardCount;
    layerSummary_.backendLoadShardCount += pipelineTrace_.backendLoadShardCount;
    layerSummary_.cacheBufferShardCount += pipelineTrace_.cacheBufferShardCount;
    layerSummary_.s2hDoneBeforeFirstH2dSubmitShardCount += s2hDoneBeforeFirstH2d;
    layerSummary_.s2hDoneBeforeLastH2dSubmitShardCount += s2hDoneBeforeLastH2d;
    layerSummary_.transferTotalMs += totalMs;
    layerSummary_.h2dTailSyncMs += tailSyncMs;

    if (layerSummary_.taskCount < kLoadLayerSummaryLayers) { return; }

    const auto backendLoadShardCount =
        static_cast<double>(layerSummary_.backendLoadShardCount);
    const auto firstH2dS2hDoneRatio =
        backendLoadShardCount > 0
            ? static_cast<double>(layerSummary_.s2hDoneBeforeFirstH2dSubmitShardCount) /
                  backendLoadShardCount
            : 0.0;
    const auto lastH2dS2hDoneRatio =
        backendLoadShardCount > 0
            ? static_cast<double>(layerSummary_.s2hDoneBeforeLastH2dSubmitShardCount) /
                  backendLoadShardCount
            : 0.0;
    const auto avgLayerTransferMs =
        layerSummary_.transferTotalMs / static_cast<double>(layerSummary_.taskCount);
    const auto avgH2dTailWaitMs =
        layerSummary_.h2dTailSyncMs / static_cast<double>(layerSummary_.taskCount);
    const auto h2dTailExposedRatio =
        layerSummary_.transferTotalMs > 0.0
            ? layerSummary_.h2dTailSyncMs / layerSummary_.transferTotalMs
            : 0.0;
    const std::string launchGranularity =
        cacheSdmaDirect_ ? sdmaDirectLaunchGranularity_ : "copy";
    UC_INFO("[UCM_LOAD_LAYER_SUMMARY] device_id={} layers={} tasks={} granularity={} "
            "shards={} backend_load_shards={} cache_buffer_shards={} "
            "avg_layer_transfer_ms={:.3f} avg_h2d_tail_wait_ms={:.3f} "
            "h2d_tail_exposed_ratio={:.3f} "
            "s2h_done_before_first_h2d_submit_ratio={:.3f} "
            "s2h_done_before_last_h2d_submit_ratio={:.3f}.",
            deviceId_, kLoadLayerSummaryLayers, layerSummary_.taskCount, launchGranularity,
            layerSummary_.shardCount, layerSummary_.backendLoadShardCount,
            layerSummary_.cacheBufferShardCount, avgLayerTransferMs, avgH2dTailWaitMs,
            h2dTailExposedRatio, firstH2dS2hDoneRatio, lastH2dS2hDoneRatio);
    layerSummary_ = {};
}

bool LoadQueue::UseSdmaDirectTaskLaunch() const noexcept
{
    return cacheSdmaDirect_ && sdmaDirectLaunchGranularity_ == kSdmaDirectLaunchTask;
}

bool LoadQueue::UseSdmaDirectBatchLaunch() const noexcept
{
    return cacheSdmaDirect_ && sdmaDirectLaunchGranularity_ == kSdmaDirectLaunchBatch;
}

}  // namespace UC::CacheStore
