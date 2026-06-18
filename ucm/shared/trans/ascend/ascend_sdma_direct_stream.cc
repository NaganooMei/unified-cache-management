/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "ascend_sdma_direct_stream.h"
#include <limits>
#include "ascend_sdma_direct_copier.h"

namespace UC::Trans {

namespace {

Status Unsupported(const char* op)
{
    return Status::InvalidParam("Cache SDMA Direct stream does not support {}", op);
}

}  // namespace

AscendSdmaDirectStream::~AscendSdmaDirectStream() = default;

Status AscendSdmaDirectStream::Setup()
{
    return Status::InvalidParam("Cache SDMA Direct stream requires stream options");
}

Status AscendSdmaDirectStream::Setup(const StreamOptions& options)
{
    auto s = ValidateStreamOptions(options);
    if (s.Failure()) [[unlikely]] { return s; }
    if (!options.cacheSdmaDirect) {
        return Status::InvalidParam("Cache SDMA Direct option is not enabled");
    }
    if (options.tensorSizes.empty()) {
        return Status::InvalidParam("invalid tensor sizes for Cache SDMA Direct");
    }
    if (options.sdmaDirectLaunchMode != "shard" && options.sdmaDirectLaunchMode != "task") {
        return Status::InvalidParam("invalid Cache SDMA Direct launch mode({})",
                                    options.sdmaDirectLaunchMode);
    }
    if (options.sdmaDirectMaxReadyLanes == 0 ||
        options.sdmaDirectMaxReadyLanes >
            static_cast<size_t>(std::numeric_limits<uint16_t>::max())) {
        return Status::InvalidParam("invalid Cache SDMA Direct max ready lanes({})",
                                    options.sdmaDirectMaxReadyLanes);
    }

    AscendSdmaDirectCopyConfig config;
    config.deviceId = options.deviceId;
    config.streamNumber = options.streamNumber;
    config.launchMode = options.sdmaDirectLaunchMode;
    config.maxReadyLanes = static_cast<uint16_t>(options.sdmaDirectMaxReadyLanes);
    auto copier = std::make_unique<AscendSdmaDirectCopier>();
    s = copier->Setup(config);
    if (s.Failure()) [[unlikely]] { return s; }
    copier_ = std::move(copier);
    return Status::OK();
}

Status AscendSdmaDirectStream::DeviceToHost(void*, void*, size_t)
{
    return Unsupported("DeviceToHost");
}

Status AscendSdmaDirectStream::DeviceToHost(void*[], void*[], size_t, size_t)
{
    return Unsupported("DeviceToHost");
}

Status AscendSdmaDirectStream::DeviceToHost(void*[], void*, size_t, size_t)
{
    return Unsupported("DeviceToHost");
}

Status AscendSdmaDirectStream::DeviceToHostAsync(void*, void*, size_t)
{
    return Unsupported("DeviceToHostAsync");
}

Status AscendSdmaDirectStream::DeviceToHostAsync(void*[], void*[], size_t, size_t)
{
    return Unsupported("DeviceToHostAsync");
}

Status AscendSdmaDirectStream::DeviceToHostAsync(void*[], void*, size_t, size_t)
{
    return Unsupported("DeviceToHostAsync");
}

Status AscendSdmaDirectStream::HostToDevice(void*, void*, size_t)
{
    return Unsupported("HostToDevice");
}

Status AscendSdmaDirectStream::HostToDevice(void*[], void*[], size_t, size_t)
{
    return Unsupported("HostToDevice");
}

Status AscendSdmaDirectStream::HostToDevice(void*, void*[], size_t, size_t)
{
    return Unsupported("HostToDevice");
}

Status AscendSdmaDirectStream::HostToDeviceAsync(void*, void*, size_t)
{
    return Unsupported("HostToDeviceAsync");
}

Status AscendSdmaDirectStream::HostToDeviceAsync(void*[], void*[], size_t, size_t)
{
    return Unsupported("HostToDeviceAsync");
}

Status AscendSdmaDirectStream::HostToDeviceAsync(void*, void*[], size_t, size_t)
{
    return Unsupported("HostToDeviceAsync");
}

Status AscendSdmaDirectStream::HostToDeviceScatterAsync(void* host, void* hostDevicePtr,
                                                        void** device,
                                                        const std::vector<size_t>& sizes)
{
    (void)host;
    if (!copier_) [[unlikely]] { return Status::Error("Cache SDMA Direct stream is not setup"); }
    return copier_->SubmitLoadObject(hostDevicePtr, device, sizes);
}

Status AscendSdmaDirectStream::DeviceToHostGatherAsync(void** device, void* host,
                                                       void* hostDevicePtr,
                                                       const std::vector<size_t>& sizes)
{
    (void)host;
    if (!copier_) [[unlikely]] { return Status::Error("Cache SDMA Direct stream is not setup"); }
    return copier_->SubmitDumpObject(device, hostDevicePtr, sizes);
}

Status AscendSdmaDirectStream::AppendCallback(std::function<void(bool)> cb)
{
    (void)cb;
    return Unsupported("AppendCallback");
}

Status AscendSdmaDirectStream::Synchronized()
{
    if (!copier_) { return Status::OK(); }
    return copier_->Synchronize();
}

Status AscendSdmaDirectStream::WaitEvent(void* event)
{
    if (!copier_) { return Status::OK(); }
    return copier_->WaitEvent(event);
}

}  // namespace UC::Trans
