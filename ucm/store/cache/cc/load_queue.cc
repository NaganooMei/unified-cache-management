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
#include <numeric>
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
    uniqueId_ = config.uniqueId;
    tensorSizes_ = config.tensorSizes;
    transferBytesPerShard_ =
        std::accumulate(tensorSizes_.begin(), tensorSizes_.end(), size_t{0});
    streamNumber_ = config.streamNumber;
    useGdr_ = config.useGdr;
    cacheSdmaDirect_ = config.cacheSdmaDirect;
    cacheSdmaTrace_ = config.cacheSdmaTrace;
    cpuAffinityCores_ = config.cpuAffinityCores;
    localRankSize_ = config.localRankSize;
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
    const auto indexes = RearrangeIndex(nShard, deviceId_, localRankSize_);
    for (size_t i = 0; i < nShard; i++) {
        auto& shard = task->desc[indexes[i]];
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
                shardTask.bufferHandle.MarkFailed(res.Error());
                task->Fail(res.Error());
                failureSet_->Insert(task->id);
                waiter->Done();
                return;
            }
            shardTask.backendTaskHandle = res.Value();
            backendSubmitCount++;
        }
        shardTask.task = task;
        shardTask.shard = std::move(shard);
        shardTask.waiter = (i + 1 < nShard) ? nullptr : waiter;
        shardTask.shardIndex = indexes[i];
        running_.Push(std::move(shardTask));
    }
    auto tpDispatch = NowTime::Now();
    UC_DEBUG("Cache task({}) dispatch shards({}), wait={:.3f}ms, cost={:.3f}ms.", task->id, nShard,
             (tpWait - tp) * 1e3, (tpDispatch - tpWait) * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_queue_wait_duration_ms"),
                             (tpWait - tp) * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_backend_submit_duration_ms"),
                             (tpDispatch - tpWait) * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_backend_shards_total"),
                             static_cast<double>(backendSubmitCount));
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_shards_total"),
                             static_cast<double>(nShard));
    if (cacheSdmaTrace_) {
        UC_INFO(
            "[UCM_SDMA_TRACE] event=load_dispatch unique_id={} device={} task={} sdma={} "
            "streams={} shards={} bytes={} backend_shards={} queue_wait_ms={:.3f} "
            "dispatch_ms={:.3f}.",
            uniqueId_, deviceId_, task->id, cacheSdmaDirect_, streamNumber_, nShard,
            transferBytesPerShard_ * nShard, backendSubmitCount, (tpWait - tp) * 1e3,
            (tpDispatch - tpWait) * 1e3);
    }
}

void LoadQueue::TransferStage(std::promise<Status>& started)
{
    auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_load_xfer");
    if (nameStatus.Failure()) { UC_WARN("Failed({}) to set UCM load transfer name.", nameStatus); }
    CopyStream stream;
    auto s = cacheSdmaDirect_ ? stream.SetupSdmaDirect(deviceId_, streamNumber_, useGdr_)
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
    auto parentTask = task.task;
    const auto taskHandle = parentTask->id;
    if (failureSet_->Contains(taskHandle)) {
        if (task.waiter) {
            if (cacheSdmaTrace_) {
                UC_INFO(
                    "[UCM_SDMA_TRACE] event=load_aborted unique_id={} device={} task={} "
                    "sdma={} streams={} status={}.",
                    uniqueId_, deviceId_, taskHandle, cacheSdmaDirect_, streamNumber_,
                    parentTask->FailureStatus());
                taskTrace_ = TaskTrace{};
            }
            holder_.clear();
            task.waiter->Done();
        }
        return;
    }
    if (cacheSdmaTrace_ && taskTrace_.taskHandle != taskHandle) {
        ResetTaskTrace(taskHandle);
    }

    auto s = Status::OK();
    auto waiter = task.waiter;
    do {
        auto tpBackendWait = NowTime::Now();
        s = WaitBackendTaskReady(task);
        if (s.Failure()) [[unlikely]] { break; }
        auto tpBackendReady = NowTime::Now();
        const auto backendWaitMs = (tpBackendReady - tpBackendWait) * 1e3;
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_shard_backend_wait_ms"), backendWaitMs);
        if (cacheSdmaTrace_) {
            taskTrace_.backendWaitSumMs += backendWaitMs;
            if (backendWaitMs > taskTrace_.backendWaitMaxMs) {
                taskTrace_.backendWaitMaxMs = backendWaitMs;
                taskTrace_.backendWaitMaxShard = task.shardIndex;
            }
        }

        auto* host = cacheSdmaDirect_ ? task.bufferHandle.DeviceData() : task.bufferHandle.Data();
        size_t streamIndex = 0;
        s = HostToDeviceAsync(stream, host, task.shard.addrs.data(),
                              cacheSdmaTrace_ ? &streamIndex : nullptr);
        auto tpH2dSubmitted = NowTime::Now();
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to do H2D for task({}).", s, taskHandle);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_errors_total"), 1.0);
            break;
        }
        const auto submitMs = (tpH2dSubmitted - tpBackendReady) * 1e3;
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_submit_ms"), submitMs);
        if (cacheSdmaTrace_) {
            if (taskTrace_.firstSubmitTp == 0.0) { taskTrace_.firstSubmitTp = tpBackendReady; }
            taskTrace_.lastSubmitTp = tpH2dSubmitted;
            taskTrace_.submitSumMs += submitMs;
            taskTrace_.processedShards++;
            taskTrace_.streamShards[streamIndex]++;
            taskTrace_.streamBytes[streamIndex] += transferBytesPerShard_;
            taskTrace_.streamSubmitMs[streamIndex] += submitMs;
            if (submitMs > taskTrace_.submitMaxMs) {
                taskTrace_.submitMaxMs = submitMs;
                taskTrace_.submitMaxShard = task.shardIndex;
                taskTrace_.submitMaxStream = streamIndex;
            }
        }
        if (!waiter) {
            holder_.push_back(std::move(task));
            return;
        }
        auto tpH2dSyncStart = NowTime::Now();
        std::vector<double> streamSyncWaitMs;
        s = stream.Synchronize(cacheSdmaTrace_ ? &streamSyncWaitMs : nullptr);
        auto tpH2dSyncEnd = NowTime::Now();
        auto h2dSyncMs = (tpH2dSyncEnd - tpH2dSyncStart) * 1e3;
        RecordH2dSyncMetrics(h2dSyncMs);
        if (cacheSdmaTrace_) {
            LogTaskTrace(*parentTask, *waiter, s, tpH2dSyncStart, tpH2dSyncEnd,
                         streamSyncWaitMs);
            taskTrace_ = TaskTrace{};
        }
        holder_.clear();
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to sync on stream for task({}).", s, taskHandle);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_errors_total"), 1.0);
            break;
        }
    } while (0);
    if (s.Failure()) [[unlikely]] {
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
            task.bufferHandle.MarkFailed(s);
            return s;
        }
        task.bufferHandle.MarkReady();
        return Status::OK();
    }
    for (;;) {
        auto state = task.bufferHandle.GetState();
        if (state == TransBuffer::State::READY) { return Status::OK(); }
        if (state == TransBuffer::State::FAILED) { return task.bufferHandle.FailureStatus(); }
        if (failureSet_->Contains(task.task->id)) { return task.task->FailureStatus(); }
        std::this_thread::yield();
    }
}

Status LoadQueue::HostToDeviceAsync(CopyStream& stream, void* host, void** device,
                                    size_t* streamIndex)
{
    return stream.HostToDeviceAsync(host, device, tensorSizes_, streamIndex);
}

void LoadQueue::RecordH2dSyncMetrics(double h2dSyncMs) const
{
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_sync_ms"), h2dSyncMs);
}

void LoadQueue::ResetTaskTrace(Detail::TaskHandle taskHandle)
{
    taskTrace_ = TaskTrace{};
    taskTrace_.taskHandle = taskHandle;
    taskTrace_.streamShards.assign(streamNumber_, 0);
    taskTrace_.streamBytes.assign(streamNumber_, 0);
    taskTrace_.streamSubmitMs.assign(streamNumber_, 0.0);
}

void LoadQueue::LogTaskTrace(const TransTask& task, const Latch& waiter, const Status& status,
                             double syncStartTp, double syncEndTp,
                             const std::vector<double>& streamSyncWaitMs) const
{
    const auto submitSpanMs = taskTrace_.firstSubmitTp > 0.0
                                  ? (taskTrace_.lastSubmitTp - taskTrace_.firstSubmitTp) * 1e3
                                  : 0.0;
    const auto h2dWindowMs = taskTrace_.firstSubmitTp > 0.0
                                 ? (syncEndTp - taskTrace_.firstSubmitTp) * 1e3
                                 : 0.0;
    const auto syncTotalMs = (syncEndTp - syncStartTp) * 1e3;
    const auto totalMs = (syncEndTp - waiter.startTp) * 1e3;
    UC_INFO(
        "[UCM_SDMA_TRACE] event=load_complete unique_id={} device={} task={} sdma={} streams={} "
        "shards={} processed_shards={} bytes={} backend_wait_sum_ms={:.3f} "
        "backend_wait_max_ms={:.3f} backend_wait_max_shard={} submit_sum_ms={:.3f} "
        "submit_max_ms={:.3f} submit_max_shard={} submit_max_stream={} "
        "submit_span_ms={:.3f} sync_total_ms={:.3f} h2d_window_ms={:.3f} "
        "total_ms={:.3f} stream_shards={} stream_bytes={} stream_submit_ms={} "
        "stream_sync_wait_ms={} status={}.",
        uniqueId_, deviceId_, task.id, cacheSdmaDirect_, streamNumber_, task.desc.size(),
        taskTrace_.processedShards, transferBytesPerShard_ * taskTrace_.processedShards,
        taskTrace_.backendWaitSumMs, taskTrace_.backendWaitMaxMs,
        taskTrace_.backendWaitMaxShard, taskTrace_.submitSumMs, taskTrace_.submitMaxMs,
        taskTrace_.submitMaxShard, taskTrace_.submitMaxStream, submitSpanMs, syncTotalMs,
        h2dWindowMs, totalMs, taskTrace_.streamShards, taskTrace_.streamBytes,
        taskTrace_.streamSubmitMs, streamSyncWaitMs, status);
}

}  // namespace UC::CacheStore
