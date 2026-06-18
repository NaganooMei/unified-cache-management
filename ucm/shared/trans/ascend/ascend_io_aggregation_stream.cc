/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "ascend_io_aggregation_stream.h"
#include <limits>
#include <numeric>
#include "ascend_shard_io_aggregator.h"

namespace UC::Trans {

namespace {

Status Unsupported(const char* op)
{
    return Status::InvalidParam("Cache IO aggregation stream does not support {}", op);
}

}  // namespace

AscendIoAggregationStream::~AscendIoAggregationStream() = default;

Status AscendIoAggregationStream::Setup()
{
    return Status::InvalidParam("Cache IO aggregation stream requires stream options");
}

Status AscendIoAggregationStream::Setup(const StreamOptions& options)
{
    auto s = ValidateStreamOptions(options);
    if (s.Failure()) [[unlikely]] { return s; }
    if (!options.cacheIOAggregation) {
        return Status::InvalidParam("Cache IO aggregation option is not enabled");
    }
    if (options.tensorSizes.empty()) {
        return Status::InvalidParam("invalid tensor sizes for Cache IO aggregation");
    }
    if (options.ioAggregationMaxReadyLanes == 0 ||
        options.ioAggregationMaxReadyLanes >
            static_cast<size_t>(std::numeric_limits<uint16_t>::max())) {
        return Status::InvalidParam("invalid Cache IO aggregation max ready lanes({})",
                                    options.ioAggregationMaxReadyLanes);
    }

    const auto objectBytes =
        std::accumulate(options.tensorSizes.begin(), options.tensorSizes.end(), size_t{0});
    AscendShardIOAggregatorConfig config;
    config.deviceId = options.deviceId;
    config.streamNumber = options.streamNumber;
    config.pipelineDepth = options.ioAggregationPipelineDepth;
    config.maxReadyLanes = static_cast<uint16_t>(options.ioAggregationMaxReadyLanes);
    config.objectBytes = objectBytes;
    config.maxFragments = options.tensorSizes.size();
    auto aggregator = std::make_unique<AscendShardIOAggregator>();
    s = aggregator->Setup(config);
    if (s.Failure()) [[unlikely]] { return s; }
    aggregator_ = std::move(aggregator);
    return Status::OK();
}

Status AscendIoAggregationStream::DeviceToHost(void*, void*, size_t)
{
    return Unsupported("DeviceToHost");
}

Status AscendIoAggregationStream::DeviceToHost(void*[], void*[], size_t, size_t)
{
    return Unsupported("DeviceToHost");
}

Status AscendIoAggregationStream::DeviceToHost(void*[], void*, size_t, size_t)
{
    return Unsupported("DeviceToHost");
}

Status AscendIoAggregationStream::DeviceToHostAsync(void*, void*, size_t)
{
    return Unsupported("DeviceToHostAsync");
}

Status AscendIoAggregationStream::DeviceToHostAsync(void*[], void*[], size_t, size_t)
{
    return Unsupported("DeviceToHostAsync");
}

Status AscendIoAggregationStream::DeviceToHostAsync(void*[], void*, size_t, size_t)
{
    return Unsupported("DeviceToHostAsync");
}

Status AscendIoAggregationStream::HostToDevice(void*, void*, size_t)
{
    return Unsupported("HostToDevice");
}

Status AscendIoAggregationStream::HostToDevice(void*[], void*[], size_t, size_t)
{
    return Unsupported("HostToDevice");
}

Status AscendIoAggregationStream::HostToDevice(void*, void*[], size_t, size_t)
{
    return Unsupported("HostToDevice");
}

Status AscendIoAggregationStream::HostToDeviceAsync(void*, void*, size_t)
{
    return Unsupported("HostToDeviceAsync");
}

Status AscendIoAggregationStream::HostToDeviceAsync(void*[], void*[], size_t, size_t)
{
    return Unsupported("HostToDeviceAsync");
}

Status AscendIoAggregationStream::HostToDeviceAsync(void*, void*[], size_t, size_t)
{
    return Unsupported("HostToDeviceAsync");
}

Status AscendIoAggregationStream::HostToDeviceScatterAsync(void* host, void* hostDevicePtr,
                                                           void** device,
                                                           const std::vector<size_t>& sizes)
{
    (void)hostDevicePtr;
    if (!aggregator_) [[unlikely]] {
        return Status::Error("Cache IO aggregation stream is not setup");
    }
    return aggregator_->SubmitLoadObject(host, device, sizes);
}

Status AscendIoAggregationStream::DeviceToHostGatherAsync(void** device, void* host,
                                                          void* hostDevicePtr,
                                                          const std::vector<size_t>& sizes)
{
    (void)hostDevicePtr;
    if (!aggregator_) [[unlikely]] {
        return Status::Error("Cache IO aggregation stream is not setup");
    }
    return aggregator_->SubmitDumpObject(device, host, sizes);
}

Status AscendIoAggregationStream::AppendCallback(std::function<void(bool)> cb)
{
    (void)cb;
    return Unsupported("AppendCallback");
}

Status AscendIoAggregationStream::Synchronized()
{
    if (!aggregator_) { return Status::OK(); }
    return aggregator_->Synchronize();
}

Status AscendIoAggregationStream::WaitEvent(void* event)
{
    if (!aggregator_) { return Status::OK(); }
    return aggregator_->WaitEvent(event);
}

}  // namespace UC::Trans
