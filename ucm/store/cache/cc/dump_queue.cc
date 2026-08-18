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
#include "dump_queue.h"
#include <algorithm>
#include <atomic>
#include <memory>
#include <numeric>
#include "logger/logger.h"
#include "metrics_api.h"
#include "thread/cpu_affinity.h"

namespace UC::CacheStore {

DumpQueue::~DumpQueue()
{
    stop_.store(true);
    if (dispatcher_.joinable()) { dispatcher_.join(); }
    if (dumper_.joinable()) { dumper_.join(); }
}

Status DumpQueue::Setup(const Config& config, TaskIdSet* failureSet, TransBuffer* buffer)
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
    waiting_.Setup(config.waitingQueueDepth);
    dumping_.Setup(config.runningQueueDepth);
    dumper_ = std::thread{&DumpQueue::BackendDumpStage, this};
    std::promise<Status> started;
    auto fut = started.get_future();
    dispatcher_ = std::thread{&DumpQueue::DispatchStage, this, std::ref(started)};
    return fut.get();
}

void DumpQueue::Submit(TaskPtr task, WaiterPtr waiter)
{
    waiter->Up();
    auto success = waiting_.TryPush({task, waiter});
    if (success) { return; }
    UC_ERROR("Waiting queue full, submit dump task({}) failed.", task->id);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_queue_full_total"), 1.0);
    failureSet_->Insert(task->id);
    waiter->Done();
}

void DumpQueue::DispatchStage(std::promise<Status>& started)
{
    auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_dump_disp");
    if (nameStatus.Failure()) {
        UC_WARN("Failed({}) to set UCM dump dispatcher name.", nameStatus);
    }
    CopyStream stream;
    auto s = cacheSdmaDirect_ ? stream.SetupSdmaDirect(deviceId_, streamNumber_, useGdr_)
                              : stream.Setup(deviceId_, streamNumber_, useGdr_);
    started.set_value(s);
    if (s.Failure()) [[unlikely]] { return; }
    if (!cpuAffinityCores_.empty()) {
        s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
        if (s.Failure()) { UC_WARN("Failed({}) to set affinity.", s); }
    }
    waiting_.ConsumerLoop(stop_, &DumpQueue::DispatchOneTask, this, stream);
}

void DumpQueue::DispatchOneTask(CopyStream& stream, TaskPair&& pair)
{
    auto& task = pair.first;
    auto& waiter = pair.second;
    const auto queueWaitMs = (NowTime::Now() - waiter->startTp) * 1e3;
    UC_DEBUG("Cache task({}) start running, wait {:.3f}ms.", task->id, queueWaitMs);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_queue_wait_duration_ms"), queueWaitMs);
    if (!failureSet_->Contains(task->id)) {
        auto s = DumpOneTask(stream, task, queueWaitMs);
        if (s.Failure()) [[unlikely]] {
            if (s == Status::StoreUnhealthy()) { task->Fail(s); }
            failureSet_->Insert(task->id);
        }
    }
    waiter->Done();
}

Status DumpQueue::DumpOneTask(CopyStream& stream, TaskPtr task, double queueWaitMs)
{
    auto dumpStartTp = NowTime::Now();
    Detail::TaskDesc backendTaskDesc;
    backendTaskDesc.brief = "Cache2Backend";
    const auto nShard = task->desc.size();
    UC_DEBUG("Try to dump ({}) shards.", nShard);
    DumpCtx dumpCtx;
    dumpCtx.taskHandle = task->id;
    std::shared_ptr<std::atomic<double>> eventReadyTp;
    if (task->desc.prerequisiteHandle != 0) {
        auto s = stream.WaitEvent(reinterpret_cast<void*>(task->desc.prerequisiteHandle));
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to wait prerequisite event for dump task({}).", s, task->id);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_d2h_errors_total"), 1.0);
            return s;
        }
        eventReadyTp = std::make_shared<std::atomic<double>>(0.0);
        auto cbStatus = stream.AppendCallback([eventReadyTp](bool) {
            eventReadyTp->store(NowTime::Now(), std::memory_order_release);
        });
        if (cbStatus.Failure()) [[unlikely]] { eventReadyTp.reset(); }
    }
    size_t copiedShards = 0;
    std::vector<size_t> streamShards;
    std::vector<size_t> streamBytes;
    std::vector<double> streamSubmitMs;
    if (cacheSdmaTrace_) {
        streamShards.assign(streamNumber_, 0);
        streamBytes.assign(streamNumber_, 0);
        streamSubmitMs.assign(streamNumber_, 0.0);
    }
    double firstSubmitTp = 0.0;
    double lastSubmitTp = 0.0;
    double submitSumMs = 0.0;
    double submitMaxMs = 0.0;
    size_t submitMaxShard = 0;
    size_t submitMaxStream = 0;
    for (size_t i = 0; i < nShard; i++) {
        auto& shard = task->desc[i];
        auto handle = buffer_->Get(shard.owner, shard.index);
        if (!handle.Owner()) { continue; }
        if (!handle.Ready()) {
            auto* host = cacheSdmaDirect_ ? handle.DeviceData() : handle.Data();
            const auto tpSubmitStart = cacheSdmaTrace_ ? NowTime::Now() : 0.0;
            size_t streamIndex = 0;
            auto s = DeviceToHostAsync(stream, shard.addrs.data(), host,
                                       cacheSdmaTrace_ ? &streamIndex : nullptr);
            const auto tpSubmitted = cacheSdmaTrace_ ? NowTime::Now() : 0.0;
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to do D2H for task({}).", s, task->id);
                UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_d2h_errors_total"), 1.0);
                return s;
            }
            copiedShards++;
            if (cacheSdmaTrace_) {
                const auto submitMs = (tpSubmitted - tpSubmitStart) * 1e3;
                if (firstSubmitTp == 0.0) { firstSubmitTp = tpSubmitStart; }
                lastSubmitTp = tpSubmitted;
                submitSumMs += submitMs;
                streamShards[streamIndex]++;
                streamBytes[streamIndex] += transferBytesPerShard_;
                streamSubmitMs[streamIndex] += submitMs;
                if (submitMs > submitMaxMs) {
                    submitMaxMs = submitMs;
                    submitMaxShard = i;
                    submitMaxStream = streamIndex;
                }
            }
        }
        backendTaskDesc.push_back(Detail::Shard{shard.owner, shard.index, {handle.Data()}});
        dumpCtx.bufferHandles.push_back(std::move(handle));
    }
    auto tpMakeBuffer = NowTime::Now();
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_shards_total"),
                             static_cast<double>(nShard));
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_backend_shards_total"),
                             static_cast<double>(backendTaskDesc.size()));
    if (backendTaskDesc.empty()) {
        if (cacheSdmaTrace_) {
            UC_INFO(
                "[UCM_SDMA_TRACE] event=dump_complete unique_id={} device={} task={} sdma={} "
                "streams={} shards={} copied_shards=0 bytes=0 queue_wait_ms={:.3f} "
                "total_ms={:.3f} status=OK reason=no_owned_shards.",
                uniqueId_, deviceId_, task->id, cacheSdmaDirect_, streamNumber_, nShard,
                queueWaitMs, queueWaitMs + (NowTime::Now() - dumpStartTp) * 1e3);
        }
        return Status::OK();
    }
    const auto backendShardCount = backendTaskDesc.size();
    auto tpSyncStart = NowTime::Now();
    std::vector<double> streamSyncWaitMs;
    auto s = stream.Synchronize(cacheSdmaTrace_ ? &streamSyncWaitMs : nullptr);
    if (s.Failure()) [[unlikely]] {
        UC_ERROR("Failed({}) to sync on stream for task({}).", s, task->id);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_d2h_errors_total"), 1.0);
        return s;
    }
    auto tpSyncStream = NowTime::Now();
    auto tpBackendSubmitStart = NowTime::Now();
    for (auto& handle : dumpCtx.bufferHandles) { handle.MarkReady(); }
    auto res = backend_->Dump(std::move(backendTaskDesc));
    if (!res) [[unlikely]] {
        auto error = res.Error();
        if (error != Status::StoreUnhealthy()) {
            UC_ERROR("Failed({}) to submit dump task({}) to backend.", error, task->id);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_backend_dump_submit_errors_total"),
                                     1.0);
        }
        return error;
    }
    dumpCtx.backendTaskHandle = res.Value();
    dumping_.Push(std::move(dumpCtx));
    auto tpEnd = NowTime::Now();
    auto prereqWaitMs = 0.0;
    auto d2hMs = std::max(0.0, tpSyncStream - tpSyncStart) * 1e3;
    if (eventReadyTp) {
        auto ready = eventReadyTp->load(std::memory_order_acquire);
        if (ready > 0.0) {
            prereqWaitMs = std::max(0.0, ready - dumpStartTp) * 1e3;
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_prereq_wait_ms"), prereqWaitMs);
        }
    }
    if (copiedShards > 0 && d2hMs > 0.0) {
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_d2h_duration_ms"), d2hMs);
    }
    UC_DEBUG("Cache task({}) mk_buf={:.3f}ms, prereq={:.3f}ms, d2h={:.3f}ms, back={:.3f}ms.",
             task->id, (tpMakeBuffer - dumpStartTp) * 1e3, prereqWaitMs, d2hMs,
             (tpEnd - tpBackendSubmitStart) * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_mkbuf_duration_ms"),
                             (tpMakeBuffer - dumpStartTp) * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_backend_submit_duration_ms"),
                             (tpEnd - tpBackendSubmitStart) * 1e3);
    if (cacheSdmaTrace_) {
        const auto submitSpanMs = firstSubmitTp > 0.0 ? (lastSubmitTp - firstSubmitTp) * 1e3 : 0.0;
        const auto d2hWindowMs = firstSubmitTp > 0.0 ? (tpSyncStream - firstSubmitTp) * 1e3 : 0.0;
        UC_INFO(
            "[UCM_SDMA_TRACE] event=dump_complete unique_id={} device={} task={} sdma={} "
            "streams={} shards={} copied_shards={} backend_shards={} bytes={} "
            "queue_wait_ms={:.3f} mkbuf_ms={:.3f} prereq_wait_ms={:.3f} "
            "submit_sum_ms={:.3f} submit_max_ms={:.3f} submit_max_shard={} "
            "submit_max_stream={} submit_span_ms={:.3f} sync_total_ms={:.3f} "
            "d2h_window_ms={:.3f} backend_submit_ms={:.3f} total_ms={:.3f} "
            "stream_shards={} stream_bytes={} stream_submit_ms={} stream_sync_wait_ms={} "
            "status=OK.",
            uniqueId_, deviceId_, task->id, cacheSdmaDirect_, streamNumber_, nShard, copiedShards,
            backendShardCount, transferBytesPerShard_ * copiedShards, queueWaitMs,
            (tpMakeBuffer - dumpStartTp) * 1e3, prereqWaitMs, submitSumMs, submitMaxMs,
            submitMaxShard, submitMaxStream, submitSpanMs, d2hMs, d2hWindowMs,
            (tpEnd - tpBackendSubmitStart) * 1e3,
            queueWaitMs + (tpEnd - dumpStartTp) * 1e3,
            streamShards, streamBytes, streamSubmitMs, streamSyncWaitMs);
    }
    return Status::OK();
}

Status DumpQueue::DeviceToHostAsync(CopyStream& stream, void** device, void* host,
                                    size_t* streamIndex)
{
    return stream.DeviceToHostAsync(device, host, tensorSizes_, streamIndex);
}

void DumpQueue::BackendDumpStage()
{
    auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_dump_back");
    if (nameStatus.Failure()) { UC_WARN("Failed({}) to set UCM dump backend name.", nameStatus); }
    if (!cpuAffinityCores_.empty()) {
        auto s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
        if (s.Failure()) { UC_WARN("Failed({}) to set affinity.", s); }
    }
    dumping_.ConsumerLoop(stop_, [this](auto&& task) {
        if (task.backendTaskHandle > finishedBackendTaskHandle_) {
            auto tpWait = NowTime::Now();
            auto s = backend_->Wait(task.backendTaskHandle);
            const auto backendWaitMs = (NowTime::Now() - tpWait) * 1e3;
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_dump_backend_wait_duration_ms"),
                                     backendWaitMs);
            if (cacheSdmaTrace_) {
                UC_INFO(
                    "[UCM_SDMA_TRACE] event=dump_backend_complete unique_id={} device={} "
                    "task={} backend_task={} backend_wait_ms={:.3f} status={}.",
                    uniqueId_, deviceId_, task.taskHandle, task.backendTaskHandle, backendWaitMs,
                    s);
            }
            finishedBackendTaskHandle_ = task.backendTaskHandle;
            if (s.Failure()) {
                UC_ERROR("Failed({}) to wait backend({}) for task({}).", s, task.backendTaskHandle,
                         task.taskHandle);
                UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_backend_dump_wait_errors_total"),
                                         1.0);
                return;
            }
        }
    });
}

}  // namespace UC::CacheStore
