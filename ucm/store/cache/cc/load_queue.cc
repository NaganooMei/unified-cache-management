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
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sys/syscall.h>
#include <unistd.h>
#include "load_queue.h"
#include "logger/logger.h"
#include "metrics_api.h"
#include "thread/cpu_affinity.h"

namespace {

constexpr const char* S2H_TRACE_ENABLE_ENV = "UCM_CACHE_POSIX_S2H_TRACE";
constexpr const char* S2H_TRACE_EPOCH_ENV = "UCM_CACHE_POSIX_S2H_TRACE_EPOCH";
constexpr const char* S2H_TRACE_WORKER_ENV = "UCM_CACHE_POSIX_S2H_TRACE_WORKER";

bool S2hTraceEnabled()
{
    const auto* enabled = std::getenv(S2H_TRACE_ENABLE_ENV);
    return enabled != nullptr && std::strcmp(enabled, "1") == 0;
}

long S2hTraceValue(const char* name)
{
    const auto* value = std::getenv(name);
    return value == nullptr ? -1 : std::strtol(value, nullptr, 10);
}

std::uint64_t MonotonicNs()
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

}  // namespace

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
    sdmaDirectLaunchGranularity_ = config.sdmaDirectLaunchGranularity;
    cpuAffinityCores_ = config.cpuAffinityCores;
    waiting_.Setup(config.waitingQueueDepth);
    running_.Setup(config.runningQueueDepth);
    holder_.reserve(1024);
    transferTrace_.reserve(1024);
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
    const auto taskLaunch = UseSdmaDirectTaskLaunch();
    const auto traceS2h = S2hTraceEnabled();
    const auto traceEpoch = traceS2h ? S2hTraceValue(S2H_TRACE_EPOCH_ENV) : -1;
    const auto traceWorker = traceS2h ? S2hTraceValue(S2H_TRACE_WORKER_ENV) : -1;
    size_t backendSubmitCount = 0;
    std::vector<ShardTask> readyTasks;
    std::vector<ShardTask> pendingTasks;
    if (taskLaunch) {
        readyTasks.reserve(nShard);
        pendingTasks.reserve(nShard);
    }
    for (size_t i = 0; i < nShard; i++) {
        auto& shard = task->desc[i];
        ShardTask shardTask;
        shardTask.traceEnabled = traceS2h;
        shardTask.traceEpoch = traceEpoch;
        shardTask.traceWorker = traceWorker;
        shardTask.traceShardPosition = i;
        shardTask.traceShardNumber = nShard;
        shardTask.traceBlockHash = traceS2h ? Detail::BlockIdHasher{}(shard.owner) : 0;
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
        if (taskLaunch) {
            if (shardTask.bufferHandle.Ready()) {
                readyTasks.push_back(std::move(shardTask));
            } else {
                pendingTasks.push_back(std::move(shardTask));
            }
            continue;
        }
        shardTask.waiter = (i + 1 < nShard) ? nullptr : waiter;
        running_.Push(std::move(shardTask));
    }
    if (taskLaunch) {
        if (!readyTasks.empty()) {
            readyTasks.back().launchBoundary = true;
            if (pendingTasks.empty()) { readyTasks.back().waiter = waiter; }
        }
        if (!pendingTasks.empty()) {
            pendingTasks.back().launchBoundary = true;
            pendingTasks.back().waiter = waiter;
        }
        for (auto& shardTask : readyTasks) { running_.Push(std::move(shardTask)); }
        for (auto& shardTask : pendingTasks) { running_.Push(std::move(shardTask)); }
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
    if (traceS2h) {
        UC_INFO_UNLIMITED(
            "[S2H_TRACE] event=assign epoch={} worker={} pid={} tid={} cache_task={} "
            "backend_tasks={} total_shards={}",
            traceEpoch, traceWorker, getpid(), syscall(SYS_gettid), task->id, backendSubmitCount,
            nShard);
    }
}

void LoadQueue::TransferStage(std::promise<Status>& started)
{
    auto nameStatus = CpuAffinity::SetCurrentThreadName("ucm_load_xfer");
    if (nameStatus.Failure()) { UC_WARN("Failed({}) to set UCM load transfer name.", nameStatus); }
    CopyStream stream;
    auto s = cacheSdmaDirect_ ? stream.SetupSdmaDirect(deviceId_, useGdr_)
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
            ClearSdmaDirectHolders();
            task.waiter->Done();
            FlushTransferTrace();
        }
        return;
    }
    auto s = Status::OK();
    auto waiter = task.waiter;
    const auto traceS2h = task.traceEnabled;
    const auto backendOwner = task.backendTaskHandle != 0;
    std::uint64_t backendWaitStartNs = 0;
    std::uint64_t backendReadyNs = 0;
    auto traceTransfer = [&](std::uint64_t h2dStartNs, std::uint64_t h2dEndNs,
                             std::uint64_t syncStartNs, std::uint64_t syncEndNs,
                             const char* status) {
        if (!traceS2h) { return; }
        TransferTrace trace;
        trace.epoch = task.traceEpoch;
        trace.worker = task.traceWorker;
        trace.pid = getpid();
        trace.tid = syscall(SYS_gettid);
        trace.cacheTask = taskHandle;
        trace.shardPosition = task.traceShardPosition;
        trace.shardNumber = task.traceShardNumber;
        trace.blockHash = task.traceBlockHash;
        trace.shardIndex = task.shard.index;
        trace.backendOwner = backendOwner;
        trace.waitStartNs = backendWaitStartNs;
        trace.readyNs = backendReadyNs;
        trace.h2dStartNs = h2dStartNs;
        trace.h2dEndNs = h2dEndNs;
        trace.syncStartNs = syncStartNs;
        trace.syncEndNs = syncEndNs;
        trace.final = waiter != nullptr;
        trace.status = status;
        transferTrace_.push_back(trace);
    };
    do {
        auto tpBackendWait = NowTime::Now();
        if (traceS2h) { backendWaitStartNs = MonotonicNs(); }
        s = WaitBackendTaskReady(task);
        if (traceS2h) { backendReadyNs = MonotonicNs(); }
        if (s.Failure()) [[unlikely]] {
            traceTransfer(0, 0, 0, 0, "backend_error");
            break;
        }
        auto tpBackendReady = NowTime::Now();
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_shard_backend_wait_ms"),
                                 (tpBackendReady - tpBackendWait) * 1e3);
        if (UseSdmaDirectTaskLaunch()) {
            const auto launchBoundary = task.launchBoundary;
            holder_.push_back(std::move(task));
            if (!launchBoundary) { return; }
            s = FlushSdmaDirectTaskBatch(stream);
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to flush H2D task batch for task({}).", s, taskHandle);
                UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_errors_total"), 1.0);
                break;
            }
            if (waiter) { waiter->Done(); }
            return;
        }

        if (cacheSdmaDirect_) {
            const auto h2dStartNs = traceS2h ? MonotonicNs() : 0;
            s = HostToDeviceAsync(stream, task.bufferHandle.DeviceData(), task.shard.addrs.data());
            const auto h2dEndNs = traceS2h ? MonotonicNs() : 0;
            auto tpH2dSubmitted = NowTime::Now();
            if (s.Failure()) [[unlikely]] {
                traceTransfer(h2dStartNs, h2dEndNs, 0, 0, "h2d_error");
                UC_ERROR("Failed({}) to do H2D for task({}).", s, taskHandle);
                UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_errors_total"), 1.0);
                break;
            }
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_submit_ms"),
                                     (tpH2dSubmitted - tpBackendReady) * 1e3);
            if (!task.waiter) {
                traceTransfer(h2dStartNs, h2dEndNs, 0, 0, "ok");
                holder_.push_back(std::move(task));
                return;
            }
            auto tpH2dSyncStart = NowTime::Now();
            const auto syncStartNs = traceS2h ? MonotonicNs() : 0;
            s = stream.Synchronize();
            const auto syncEndNs = traceS2h ? MonotonicNs() : 0;
            auto h2dSyncMs = (NowTime::Now() - tpH2dSyncStart) * 1e3;
            RecordH2dSyncMetrics(h2dSyncMs);
            holder_.clear();
            if (s.Failure()) [[unlikely]] {
                traceTransfer(h2dStartNs, h2dEndNs, syncStartNs, syncEndNs, "sync_error");
                UC_ERROR("Failed({}) to sync on stream for task({}).", s, taskHandle);
                UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_errors_total"), 1.0);
                break;
            }
            traceTransfer(h2dStartNs, h2dEndNs, syncStartNs, syncEndNs, "ok");
            break;
        }

        const auto h2dStartNs = traceS2h ? MonotonicNs() : 0;
        s = HostToDeviceAsync(stream, task.bufferHandle.Data(), task.shard.addrs.data());
        const auto h2dEndNs = traceS2h ? MonotonicNs() : 0;
        auto tpH2dSubmitted = NowTime::Now();
        if (s.Failure()) [[unlikely]] {
            traceTransfer(h2dStartNs, h2dEndNs, 0, 0, "h2d_error");
            UC_ERROR("Failed({}) to do H2D for task({}).", s, taskHandle);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_errors_total"), 1.0);
            break;
        }
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_submit_ms"),
                                 (tpH2dSubmitted - tpBackendReady) * 1e3);
        if (!task.waiter) {
            traceTransfer(h2dStartNs, h2dEndNs, 0, 0, "ok");
            holder_.push_back(std::move(task));
            return;
        }
        auto tpH2dSyncStart = NowTime::Now();
        const auto syncStartNs = traceS2h ? MonotonicNs() : 0;
        s = stream.Synchronize();
        const auto syncEndNs = traceS2h ? MonotonicNs() : 0;
        auto h2dSyncMs = (NowTime::Now() - tpH2dSyncStart) * 1e3;
        RecordH2dSyncMetrics(h2dSyncMs);
        holder_.clear();
        if (s.Failure()) [[unlikely]] {
            traceTransfer(h2dStartNs, h2dEndNs, syncStartNs, syncEndNs, "sync_error");
            UC_ERROR("Failed({}) to sync on stream for task({}).", s, taskHandle);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_errors_total"), 1.0);
            break;
        }
        traceTransfer(h2dStartNs, h2dEndNs, syncStartNs, syncEndNs, "ok");
    } while (0);
    if (s.Failure()) [[unlikely]] {
        parentTask->Fail(s);
        failureSet_->Insert(taskHandle);
    }
    if (UseSdmaDirectTaskLaunch()) { ClearSdmaDirectHolders(); }
    if (waiter) {
        waiter->Done();
        FlushTransferTrace();
    }
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
            return s;
        }
        task.bufferHandle.MarkReady();
        return Status::OK();
    }
    while (!task.bufferHandle.Ready()) {
        if (failureSet_->Contains(task.task->id)) { return task.task->FailureStatus(); }
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

Status LoadQueue::FlushSdmaDirectTaskBatch(CopyStream& stream)
{
    if (holder_.empty()) { return Status::OK(); }
    auto tpH2dSubmitStart = NowTime::Now();
    auto s = HostToDeviceTaskAsync(stream, holder_);
    auto tpH2dSubmitted = NowTime::Now();
    if (s.Failure()) [[unlikely]] {
        ClearSdmaDirectHolders();
        return s;
    }
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_submit_ms"),
                             (tpH2dSubmitted - tpH2dSubmitStart) * 1e3);
    auto tpH2dSyncStart = NowTime::Now();
    s = stream.Synchronize();
    auto h2dSyncMs = (NowTime::Now() - tpH2dSyncStart) * 1e3;
    RecordH2dSyncMetrics(h2dSyncMs);
    ClearSdmaDirectHolders();
    return s;
}

void LoadQueue::RecordH2dSyncMetrics(double h2dSyncMs) const
{
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_h2d_sync_ms"), h2dSyncMs);
}

void LoadQueue::FlushTransferTrace()
{
    for (const auto& trace : transferTrace_) {
        UC_INFO_UNLIMITED(
            "[S2H_TRACE] event=xfer epoch={} worker={} pid={} tid={} cache_task={} "
            "shard_pos={} total_shards={} block_hash={} shard={} backend_owner={} "
            "wait_start_ns={} ready_ns={} wait_ns={} h2d_start_ns={} h2d_end_ns={} "
            "h2d_submit_ns={} sync_start_ns={} sync_end_ns={} sync_ns={} final={} status={}",
            trace.epoch, trace.worker, trace.pid, trace.tid, trace.cacheTask,
            trace.shardPosition, trace.shardNumber, trace.blockHash, trace.shardIndex,
            trace.backendOwner ? 1 : 0, trace.waitStartNs, trace.readyNs,
            trace.readyNs - trace.waitStartNs, trace.h2dStartNs, trace.h2dEndNs,
            trace.h2dEndNs - trace.h2dStartNs, trace.syncStartNs, trace.syncEndNs,
            trace.syncEndNs - trace.syncStartNs, trace.final ? 1 : 0, trace.status);
    }
    transferTrace_.clear();
}

void LoadQueue::ClearSdmaDirectHolders() noexcept { holder_.clear(); }

bool LoadQueue::UseSdmaDirectTaskLaunch() const noexcept
{
    return cacheSdmaDirect_ && sdmaDirectLaunchGranularity_ == kSdmaDirectLaunchTask;
}

}  // namespace UC::CacheStore
