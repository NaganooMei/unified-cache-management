/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "ascend_io_aggregation_stream.h"
#include <limits>
#include "ascend_shard_io_aggregator.h"

namespace UC::Trans {

namespace {

Status Unsupported(const char* op)
{
    return Status::InvalidParam("Cache IO aggregation stream does not support {}", op);
}

}  // namespace

AscendIoAggregationStream::AscendIoAggregationStream() = default;

AscendIoAggregationStream::~AscendIoAggregationStream() = default;

Status AscendIoAggregationStream::Setup()
{
    return Status::InvalidParam("Cache IO aggregation stream requires config");
}

Status AscendIoAggregationStream::Setup(const IoAggregationStreamConfig& config)
{
    if (config.maxReadyLanes == 0 ||
        config.maxReadyLanes > static_cast<size_t>(std::numeric_limits<uint16_t>::max())) {
        return Status::InvalidParam("invalid Cache IO aggregation max ready lanes({})",
                                    config.maxReadyLanes);
    }

    AscendShardIOAggregatorConfig aggregatorConfig;
    aggregatorConfig.deviceId = config.deviceId;
    aggregatorConfig.streamNumber = config.laneNumber;
    aggregatorConfig.pipelineDepth = config.pipelineDepth;
    aggregatorConfig.maxReadyLanes = static_cast<uint16_t>(config.maxReadyLanes);
    aggregatorConfig.objectBytes = config.objectBytes;
    aggregatorConfig.maxFragments = config.maxFragments;
    auto aggregator = std::make_unique<AscendShardIOAggregator>();
    auto s = aggregator->Setup(aggregatorConfig);
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

Status AscendIoAggregationStream::HostToDeviceAsync(void* host, void* device[],
                                                    const std::vector<size_t>& sizes,
                                                    void* mappedHost)
{
    (void)mappedHost;
    if (!aggregator_) [[unlikely]] {
        return Status::Error("Cache IO aggregation stream is not setup");
    }
    return aggregator_->SubmitLoadObject(host, device, sizes);
}

Status AscendIoAggregationStream::DeviceToHostAsync(void* device[], void* host,
                                                    const std::vector<size_t>& sizes,
                                                    void* mappedHost)
{
    (void)mappedHost;
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
