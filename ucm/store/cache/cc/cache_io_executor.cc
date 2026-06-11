/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "cache_io_executor.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <utility>
#include "copy_stream.h"
#include "logger/logger.h"

#ifndef UCM_ENABLE_ASCEND_IO_AGGREGATION
#define UCM_ENABLE_ASCEND_IO_AGGREGATION 0
#endif

#ifndef UCM_ENABLE_ASCEND_FFTS_DIRECT_H2D
#define UCM_ENABLE_ASCEND_FFTS_DIRECT_H2D 0
#endif

#if UCM_ENABLE_ASCEND_IO_AGGREGATION
#include "trans/ascend/ascend_shard_io_aggregator.h"
#endif

#if UCM_ENABLE_ASCEND_FFTS_DIRECT_H2D
#include <acl/acl.h>
#include "trans/ascend/ffts_d2d_dispatcher.h"
#endif

namespace UC::CacheStore {

class TensorIOExecutor : public CacheIOExecutor {
    CopyStream stream_;
    std::vector<size_t> tensorSizes_{};

public:
    Status Setup(const Config& config) override
    {
        tensorSizes_ = config.tensorSizes;
        return stream_.Setup(config.deviceId, config.streamNumber, config.useGdr);
    }
    Status WaitEvent(void* event) override { return stream_.WaitEvent(event); }
    Status HostToDevice(void* host, void** devices, const void*) override
    {
        const auto number = tensorSizes_.size();
        for (size_t i = 0, offset = 0; i < number; i++) {
            auto pHost = static_cast<void*>(static_cast<int8_t*>(host) + offset);
            auto pDevice = devices[i];
            auto size = tensorSizes_[i];
            auto s = stream_.NextStream()->HostToDeviceAsync(pHost, pDevice, size);
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to do H2D({}) tensor({}/{}).", s, size, i, number);
                return s;
            }
            offset += size;
        }
        return Status::OK();
    }
    Status DeviceToHost(void** devices, void* host) override
    {
        const auto number = tensorSizes_.size();
        for (size_t i = 0, offset = 0; i < number; i++) {
            auto pDevice = devices[i];
            auto pHost = static_cast<void*>(static_cast<int8_t*>(host) + offset);
            auto size = tensorSizes_[i];
            auto s = stream_.NextStream()->DeviceToHostAsync(pDevice, pHost, size);
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to do D2H({}) tensor({}/{}).", s, size, i, number);
                return s;
            }
            offset += size;
        }
        return Status::OK();
    }
    Status Synchronize() override { return stream_.Synchronize(); }
};

#if UCM_ENABLE_ASCEND_FFTS_DIRECT_H2D
class FftsDirectH2DExecutor : public CacheIOExecutor {
    enum class LaunchMode { SHARD, TASK };
    struct InFlightObject {
        std::vector<Trans::AscendFftsCopySpec> specs;
        Trans::FftsD2DDispatcher dispatcher;
    };

    CopyStream copyStream_;
    int32_t deviceId_{-1};
    aclrtStream fftsStream_{nullptr};
    std::vector<size_t> tensorSizes_{};
    uint16_t maxReadyLanes_{8};
    LaunchMode launchMode_{LaunchMode::SHARD};
    std::vector<Trans::AscendFftsCopySpec> pendingSpecs_{};
    std::vector<std::unique_ptr<InFlightObject>> inFlight_{};
    bool setup_{false};

public:
    ~FftsDirectH2DExecutor() override { Cleanup(); }

    Status Setup(const Config& config) override
    {
        Cleanup();
        if (config.deviceId < 0) { return Status::InvalidParam("invalid device id"); }
        if (config.fftsDirectH2DMaxReadyLanes > std::numeric_limits<uint16_t>::max()) {
            return Status::InvalidParam("too many FFTS direct H2D ready lanes({})",
                                        config.fftsDirectH2DMaxReadyLanes);
        }
        tensorSizes_ = config.tensorSizes;
        maxReadyLanes_ = static_cast<uint16_t>(config.fftsDirectH2DMaxReadyLanes);
        launchMode_ = config.fftsDirectH2DLaunchMode == "task" ? LaunchMode::TASK
                                                               : LaunchMode::SHARD;

        auto s = copyStream_.Setup(config.deviceId, config.streamNumber, config.useGdr);
        if (s.Failure()) { return s; }
        s = AclStatus(aclrtSetDevice(config.deviceId), "aclrtSetDevice");
        if (s.Failure()) { return s; }
        s = AclStatus(aclrtCreateStream(&fftsStream_), "aclrtCreateStream(ffts-direct-h2d)");
        if (s.Failure()) { return s; }

        deviceId_ = config.deviceId;
        setup_ = true;
        const auto objectBytes =
            std::accumulate(tensorSizes_.begin(), tensorSizes_.end(), static_cast<size_t>(0));
        UC_INFO("Set Cache FFTS direct H2D mode={}, streams={}, objectBytes={}, tensors={}, "
                "maxReadyLanes={}.",
                config.fftsDirectH2DLaunchMode, config.streamNumber, objectBytes,
                tensorSizes_.size(), maxReadyLanes_);
        return Status::OK();
    }

    Status WaitEvent(void* event) override
    {
        auto s = copyStream_.WaitEvent(event);
        if (s.Failure() || event == nullptr) { return s; }
        s = AclStatus(aclrtSetDevice(deviceId_), "aclrtSetDevice");
        if (s.Failure()) { return s; }
        return AclStatus(aclrtStreamWaitEvent(fftsStream_, static_cast<aclrtEvent>(event)),
                         "aclrtStreamWaitEvent(ffts-direct-h2d)");
    }

    Status HostToDevice(void*, void** devices, const void* deviceHost) override
    {
        std::vector<Trans::AscendFftsCopySpec> specs;
        auto s = BuildSpecs(deviceHost, devices, specs);
        if (s.Failure()) { return s; }
        if (launchMode_ == LaunchMode::TASK) {
            pendingSpecs_.insert(pendingSpecs_.end(), specs.begin(), specs.end());
            return Status::OK();
        }
        return LaunchSpecs(std::move(specs));
    }

    Status DeviceToHost(void** devices, void* host) override
    {
        const auto number = tensorSizes_.size();
        for (size_t i = 0, offset = 0; i < number; i++) {
            auto pDevice = devices[i];
            auto pHost = static_cast<void*>(static_cast<int8_t*>(host) + offset);
            auto size = tensorSizes_[i];
            auto s = copyStream_.NextStream()->DeviceToHostAsync(pDevice, pHost, size);
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to do D2H({}) tensor({}/{}).", s, size, i, number);
                return s;
            }
            offset += size;
        }
        return Status::OK();
    }

    Status Synchronize() override
    {
        auto s = LaunchPendingTask();
        if (s.Failure()) { return s; }
        if (fftsStream_ != nullptr) {
            s = AclStatus(aclrtSetDevice(deviceId_), "aclrtSetDevice");
            if (s.Failure()) { return s; }
            s = AclStatus(aclrtSynchronizeStream(fftsStream_),
                          "aclrtSynchronizeStream(ffts-direct-h2d)");
            if (s.Failure()) { return s; }
            inFlight_.clear();
        }
        return copyStream_.Synchronize();
    }

private:
    static Status AclStatus(aclError ret, const char* expr)
    {
        if (ret == ACL_SUCCESS) { return Status::OK(); }
        UC_ERROR("Failed({}) to call {}.", static_cast<int32_t>(ret), expr);
        return Status{static_cast<int32_t>(ret), expr};
    }

    Status BuildSpecs(const void* deviceHost, void** devices,
                      std::vector<Trans::AscendFftsCopySpec>& specs)
    {
        if (!setup_) { return Status::Error("FFTS direct H2D executor is not setup"); }
        if (deviceHost == nullptr || devices == nullptr) {
            return Status::InvalidParam("invalid FFTS direct H2D pointers");
        }

        const auto number = tensorSizes_.size();
        specs.reserve(number);
        for (size_t i = 0, offset = 0; i < number; i++) {
            auto size = tensorSizes_[i];
            auto* src = static_cast<const std::byte*>(deviceHost) + offset;
            specs.push_back({devices[i], src, size});
            offset += size;
        }
        return Status::OK();
    }

    Status LaunchSpecs(std::vector<Trans::AscendFftsCopySpec>&& specs)
    {
        if (specs.empty()) { return Status::OK(); }
        auto object = std::make_unique<InFlightObject>();
        object->specs = std::move(specs);
        uint16_t readyCount = 0;
        auto s = object->dispatcher.BuildCopies(object->specs, maxReadyLanes_, readyCount);
        if (s.Failure()) { return s; }
        s = AclStatus(aclrtSetDevice(deviceId_), "aclrtSetDevice");
        if (s.Failure()) { return s; }
        s = object->dispatcher.Launch(fftsStream_, readyCount);
        if (s.Failure()) { return s; }
        inFlight_.push_back(std::move(object));
        return Status::OK();
    }

    Status LaunchPendingTask()
    {
        if (pendingSpecs_.empty()) { return Status::OK(); }
        std::vector<Trans::AscendFftsCopySpec> specs;
        specs.swap(pendingSpecs_);
        return LaunchSpecs(std::move(specs));
    }

    void Cleanup() noexcept
    {
        if (fftsStream_ != nullptr) {
            if (deviceId_ >= 0) { (void)aclrtSetDevice(deviceId_); }
            (void)aclrtDestroyStream(fftsStream_);
            fftsStream_ = nullptr;
        }
        pendingSpecs_.clear();
        inFlight_.clear();
        setup_ = false;
        deviceId_ = -1;
    }
};
#else
class UnavailableFftsDirectH2DExecutor : public CacheIOExecutor {
public:
    Status Setup(const Config&) override
    {
        return Status::InvalidParam("Cache FFTS direct H2D is not compiled");
    }
    Status WaitEvent(void*) override
    {
        return Status::InvalidParam("Cache FFTS direct H2D is not compiled");
    }
    Status HostToDevice(void*, void**, const void*) override
    {
        return Status::InvalidParam("Cache FFTS direct H2D is not compiled");
    }
    Status DeviceToHost(void**, void*) override
    {
        return Status::InvalidParam("Cache FFTS direct H2D is not compiled");
    }
    Status Synchronize() override
    {
        return Status::InvalidParam("Cache FFTS direct H2D is not compiled");
    }
};
#endif

#if UCM_ENABLE_ASCEND_IO_AGGREGATION
class AggregatedIOExecutor : public CacheIOExecutor {
    Trans::AscendShardIOAggregator aggregator_;
    std::vector<size_t> tensorSizes_{};

public:
    Status Setup(const Config& config) override
    {
        tensorSizes_ = config.tensorSizes;
        const auto objectBytes =
            std::accumulate(tensorSizes_.begin(), tensorSizes_.end(), static_cast<size_t>(0));
        if (config.ioAggregationMaxReadyLanes > std::numeric_limits<uint16_t>::max()) {
            return Status::InvalidParam("too many FFTS ready lanes({})",
                                        config.ioAggregationMaxReadyLanes);
        }
        Trans::AscendShardIOAggregatorConfig aggregatorConfig;
        aggregatorConfig.deviceId = config.deviceId;
        aggregatorConfig.streamNumber = config.streamNumber;
        aggregatorConfig.pipelineDepth = config.ioAggregationPipelineDepth;
        aggregatorConfig.maxReadyLanes = static_cast<uint16_t>(config.ioAggregationMaxReadyLanes);
        aggregatorConfig.objectBytes = objectBytes;
        aggregatorConfig.maxFragments = tensorSizes_.size();
        UC_INFO("Set Cache IO aggregation streams={}, objectBytes={}, tensors={}.",
                config.streamNumber, objectBytes, tensorSizes_.size());
        auto s = aggregator_.Setup(aggregatorConfig);
        if (s.Failure()) [[unlikely]] { UC_ERROR("Failed({}) to setup Cache IO aggregator.", s); }
        return s;
    }
    Status WaitEvent(void* event) override { return aggregator_.WaitEvent(event); }
    Status HostToDevice(void* host, void** devices, const void*) override
    {
        return aggregator_.SubmitLoadObject(host, devices, tensorSizes_);
    }
    Status DeviceToHost(void** devices, void* host) override
    {
        return aggregator_.SubmitDumpObject(devices, host, tensorSizes_);
    }
    Status Synchronize() override { return aggregator_.Synchronize(); }
};
#else
class UnavailableAggregatedIOExecutor : public CacheIOExecutor {
public:
    Status Setup(const Config&) override
    {
        return Status::InvalidParam("Cache IO aggregation is not compiled");
    }
    Status WaitEvent(void*) override
    {
        return Status::InvalidParam("Cache IO aggregation is not compiled");
    }
    Status HostToDevice(void*, void**, const void*) override
    {
        return Status::InvalidParam("Cache IO aggregation is not compiled");
    }
    Status DeviceToHost(void**, void*) override
    {
        return Status::InvalidParam("Cache IO aggregation is not compiled");
    }
    Status Synchronize() override
    {
        return Status::InvalidParam("Cache IO aggregation is not compiled");
    }
};
#endif

std::unique_ptr<CacheIOExecutor> MakeCacheIOExecutor(const Config& config)
{
    if (config.cacheFftsDirectH2D) {
#if UCM_ENABLE_ASCEND_FFTS_DIRECT_H2D
        return std::make_unique<FftsDirectH2DExecutor>();
#else
        return std::make_unique<UnavailableFftsDirectH2DExecutor>();
#endif
    }
    if (!config.cacheIOAggregation) { return std::make_unique<TensorIOExecutor>(); }
#if UCM_ENABLE_ASCEND_IO_AGGREGATION
    return std::make_unique<AggregatedIOExecutor>();
#else
    return std::make_unique<UnavailableAggregatedIOExecutor>();
#endif
}

}  // namespace UC::CacheStore
