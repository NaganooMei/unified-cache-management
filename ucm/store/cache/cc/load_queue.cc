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
#include <limits>
#include <memory>
#include <numeric>
#include "logger/logger.h"
#include "metrics_api.h"
#include "thread/cpu_affinity.h"

#ifndef UCM_ENABLE_ASCEND_FFTS_PIPELINE
#define UCM_ENABLE_ASCEND_FFTS_PIPELINE 0
#endif

#if UCM_ENABLE_ASCEND_FFTS_PIPELINE
#include "trans/ascend/ascend_h2d_ffts_pipeline.h"
#endif

namespace UC::CacheStore {

class H2DTransferExecutor {
public:
    virtual ~H2DTransferExecutor() = default;
    virtual Status Setup(const Config& config) = 0;
    virtual Status Submit(void* host, void** device) = 0;
    virtual Status Synchronize() = 0;
};

class CeH2DTransferExecutor : public H2DTransferExecutor {
    CopyStream stream_;
    std::vector<size_t> tensorSizes_{};

public:
    Status Setup(const Config& config) override
    {
        tensorSizes_ = config.tensorSizes;
        return stream_.Setup(config.deviceId, config.streamNumber, config.useGdr);
    }
    Status Submit(void* host, void** device) override
    {
        const auto number = tensorSizes_.size();
        for (size_t i = 0, offset = 0; i < number; i++) {
            auto pHost = (void*)(((int8_t*)host) + offset);
            auto pDevice = device[i];
            auto size = tensorSizes_[i];
            auto s = stream_.NextStream()->HostToDeviceAsync(pHost, pDevice, size);
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to do H2D({}) batch({}/{}) async.", s, size, i, number);
                return s;
            }
            offset += size;
        }
        return Status::OK();
    }
    Status Synchronize() override { return stream_.Synchronize(); }
};

#if UCM_ENABLE_ASCEND_FFTS_PIPELINE
class FftsPipelineH2DTransferExecutor : public H2DTransferExecutor {
    struct ObjectPlanItem {
        size_t hostOffset{0};
        size_t firstFragment{0};
        size_t objectBytes{0};
        std::vector<size_t> sizes{};
    };

    Trans::AscendH2DFftsPipeline pipeline_;

public:
    Status Setup(const Config& config) override
    {
        tensorSizes_ = config.tensorSizes;
        objectPlan_ = BuildObjectPlan(tensorSizes_, config.h2dFftsObjectTargetBytes);
        if (objectPlan_.empty()) {
            return Status::InvalidParam("invalid H2D FFTS object plan");
        }
        size_t objectBytes = 0;
        size_t maxFragments = 0;
        for (const auto& item : objectPlan_) {
            objectBytes = std::max(objectBytes, item.objectBytes);
            maxFragments = std::max(maxFragments, item.sizes.size());
        }
        if (config.h2dFftsMaxReadyLanes > std::numeric_limits<uint16_t>::max()) {
            return Status::InvalidParam("too many FFTS ready lanes({})",
                                        config.h2dFftsMaxReadyLanes);
        }
        Trans::AscendH2DFftsPipelineConfig pipelineConfig;
        pipelineConfig.deviceId = config.deviceId;
        pipelineConfig.streamNumber = config.streamNumber;
        pipelineConfig.pipelineDepth = config.h2dFftsPipelineDepth;
        pipelineConfig.maxReadyLanes = static_cast<uint16_t>(config.h2dFftsMaxReadyLanes);
        pipelineConfig.objectBytes = objectBytes;
        pipelineConfig.maxFragments = maxFragments;
        UC_INFO("Set H2D FFTS object plan target={}, streams={}, objects={}, maxObjectBytes={}, "
                "maxFragments={}.",
                config.h2dFftsObjectTargetBytes, config.streamNumber, objectPlan_.size(),
                objectBytes, maxFragments);
        auto s = pipeline_.Setup(pipelineConfig);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to setup H2D FFTS pipeline.", s);
        }
        return s;
    }
    Status Submit(void* host, void** device) override
    {
        for (const auto& item : objectPlan_) {
            auto* subHost = static_cast<void*>(static_cast<int8_t*>(host) + item.hostOffset);
            auto s = pipeline_.SubmitObject(subHost, device + item.firstFragment, item.sizes);
            if (s.Failure()) [[unlikely]] { return s; }
        }
        return Status::OK();
    }
    Status Synchronize() override { return pipeline_.Synchronize(); }

private:
    static std::vector<ObjectPlanItem> BuildObjectPlan(const std::vector<size_t>& tensorSizes,
                                                       size_t targetBytes)
    {
        std::vector<ObjectPlanItem> plan;
        if (tensorSizes.empty()) { return plan; }

        if (targetBytes == 0) {
            ObjectPlanItem item;
            item.hostOffset = 0;
            item.firstFragment = 0;
            item.sizes = tensorSizes;
            item.objectBytes = std::accumulate(tensorSizes.begin(), tensorSizes.end(), size_t(0));
            plan.push_back(std::move(item));
            return plan;
        }

        size_t hostOffset = 0;
        ObjectPlanItem item;
        item.hostOffset = 0;
        item.firstFragment = 0;
        for (size_t i = 0; i < tensorSizes.size(); ++i) {
            const auto size = tensorSizes[i];
            const auto exceedsTarget =
                item.objectBytes >= targetBytes || size > targetBytes - item.objectBytes;
            if (!item.sizes.empty() && exceedsTarget) {
                plan.push_back(std::move(item));
                item = ObjectPlanItem{};
                item.hostOffset = hostOffset;
                item.firstFragment = i;
            }
            item.sizes.push_back(size);
            item.objectBytes += size;
            hostOffset += size;
        }
        if (!item.sizes.empty()) { plan.push_back(std::move(item)); }
        return plan;
    }

    std::vector<size_t> tensorSizes_{};
    std::vector<ObjectPlanItem> objectPlan_{};
};
#endif

std::unique_ptr<H2DTransferExecutor> MakeH2DTransferExecutor(const Config& config)
{
#if UCM_ENABLE_ASCEND_FFTS_PIPELINE
    if (config.h2dTransport == "ffts_pipeline") {
        return std::make_unique<FftsPipelineH2DTransferExecutor>();
    }
#else
    (void)config;
#endif
    return std::make_unique<CeH2DTransferExecutor>();
}

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
    h2dTransport_ = config.h2dTransport;
    h2dFftsPipelineDepth_ = config.h2dFftsPipelineDepth;
    h2dFftsMaxReadyLanes_ = config.h2dFftsMaxReadyLanes;
    h2dFftsObjectTargetBytes_ = config.h2dFftsObjectTargetBytes;
    cpuAffinityCores_ = config.cpuAffinityCores;
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
    UC_DEBUG("Cache task({}) dispatch shards({}), wait={:.3f}ms, cost={:.3f}ms.", task->id, nShard,
             (tpWait - tp) * 1e3, (tpDispatch - tpWait) * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_queue_wait_duration_ms"),
                             (tpWait - tp) * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_dispatch_duration_ms"),
                             (tpDispatch - tpWait) * 1e3);
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_backend_shards_total"),
                             static_cast<double>(backendSubmitCount));
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_shards_total"),
                             static_cast<double>(nShard));
}

void LoadQueue::TransferStage(std::promise<Status>& started)
{
    Config transferConfig;
    transferConfig.deviceId = deviceId_;
    transferConfig.tensorSizes = tensorSizes_;
    transferConfig.streamNumber = streamNumber_;
    transferConfig.useGdr = useGdr_;
    transferConfig.h2dTransport = h2dTransport_;
    transferConfig.h2dFftsPipelineDepth = h2dFftsPipelineDepth_;
    transferConfig.h2dFftsMaxReadyLanes = h2dFftsMaxReadyLanes_;
    transferConfig.h2dFftsObjectTargetBytes = h2dFftsObjectTargetBytes_;

    auto executor = MakeH2DTransferExecutor(transferConfig);
    auto s = executor->Setup(transferConfig);
    started.set_value(s);
    if (s.Failure()) [[unlikely]] { return; }
    if (!cpuAffinityCores_.empty()) {
        s = CpuAffinity::SetCpuAffinity4CurrentThread(cpuAffinityCores_);
        if (s.Failure()) { UC_WARN("Failed({}) to set affinity.", s); }
    }
    running_.ConsumerLoop(stop_, &LoadQueue::TransferOneTask, this, *executor);
}

void LoadQueue::TransferOneTask(H2DTransferExecutor& executor, ShardTask&& task)
{
    if (failureSet_->Contains(task.taskHandle)) {
        if (task.waiter) { task.waiter->Done(); }
        return;
    }
    auto s = Status::OK();
    do {
        auto tpBackendWait = NowTime::Now();
        s = WaitBackendTaskReady(task);
        if (s.Failure()) [[unlikely]] { break; }
        auto tpBackendReady = NowTime::Now();
        s = executor.Submit(task.bufferHandle.Data(), task.shard.addrs.data());
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to do H2D batch async for task({}).", s, task.taskHandle);
            break;
        }
        auto tpH2dSubmitted = NowTime::Now();
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_shard_backend_wait_ms"),
                                 (tpBackendReady - tpBackendWait) * 1e3);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_shard_h2d_ms"),
                                 (tpH2dSubmitted - tpBackendReady) * 1e3);
        if (!task.waiter) {
            holder_.push_back(std::move(task));
            return;
        }
        s = executor.Synchronize();
        holder_.clear();
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to sync on stream for task({}).", s, task.taskHandle);
            break;
        }
    } while (0);
    if (s.Failure()) [[unlikely]] { failureSet_->Insert(task.taskHandle); }
    if (task.waiter) { task.waiter->Done(); }
}

Status LoadQueue::WaitBackendTaskReady(ShardTask& task)
{
    if (task.backendTaskHandle != 0) {
        auto s = backend_->Wait(task.backendTaskHandle);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to wait backend({}) for task({}).", s, task.backendTaskHandle,
                     task.taskHandle);
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

}  // namespace UC::CacheStore
