/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "ascend_sdma_direct_copier.h"
#include <cstddef>
#include <utility>
#include "logger/logger.h"

namespace UC::Trans {

AscendSdmaDirectCopier::~AscendSdmaDirectCopier() { Cleanup(); }

Status AscendSdmaDirectCopier::Setup(const AscendSdmaDirectCopyConfig& config)
{
    Cleanup();
    if (config.deviceId < 0) { return Status::InvalidParam("invalid device id"); }
    if (config.maxReadyLanes == 0) {
        return Status::InvalidParam("invalid Cache SDMA Direct max ready lanes");
    }
    if (config.launchMode != "shard" && config.launchMode != "task") {
        return Status::InvalidParam("invalid Cache SDMA Direct launch mode({})",
                                    config.launchMode);
    }

    auto s = AclStatus(aclrtSetDevice(config.deviceId), "aclrtSetDevice");
    if (s.Failure()) { return s; }
    s = AclStatus(aclrtCreateStream(&fftsStream_), "aclrtCreateStream(sdma-direct)");
    if (s.Failure()) { return s; }

    deviceId_ = config.deviceId;
    maxReadyLanes_ = config.maxReadyLanes;
    launchMode_ = config.launchMode == "task" ? LaunchMode::TASK : LaunchMode::SHARD;
    setup_ = true;
    return Status::OK();
}

Status AscendSdmaDirectCopier::WaitEvent(void* event)
{
    if (event == nullptr) { return Status::OK(); }
    auto s = AclStatus(aclrtSetDevice(deviceId_), "aclrtSetDevice");
    if (s.Failure()) { return s; }
    return AclStatus(aclrtStreamWaitEvent(fftsStream_, static_cast<aclrtEvent>(event)),
                     "aclrtStreamWaitEvent(sdma-direct)");
}

Status AscendSdmaDirectCopier::SubmitLoadObject(const void* hostDevicePtr, void** devices,
                                                const std::vector<size_t>& sizes)
{
    std::vector<AscendFftsCopySpec> specs;
    auto s = BuildHostToDeviceSpecs(hostDevicePtr, devices, sizes, specs);
    if (s.Failure()) { return s; }
    if (launchMode_ == LaunchMode::TASK) {
        pendingSpecs_.insert(pendingSpecs_.end(), specs.begin(), specs.end());
        return Status::OK();
    }
    return LaunchSpecs(std::move(specs));
}

Status AscendSdmaDirectCopier::SubmitDumpObject(void** devices, void* hostDevicePtr,
                                                const std::vector<size_t>& sizes)
{
    std::vector<AscendFftsCopySpec> specs;
    auto s = BuildDeviceToHostSpecs(devices, hostDevicePtr, sizes, specs);
    if (s.Failure()) { return s; }
    if (launchMode_ == LaunchMode::TASK) {
        pendingSpecs_.insert(pendingSpecs_.end(), specs.begin(), specs.end());
        return Status::OK();
    }
    return LaunchSpecs(std::move(specs));
}

Status AscendSdmaDirectCopier::Synchronize()
{
    auto s = LaunchPendingTask();
    if (s.Failure()) { return s; }
    if (fftsStream_ == nullptr) { return Status::OK(); }
    s = AclStatus(aclrtSetDevice(deviceId_), "aclrtSetDevice");
    if (s.Failure()) { return s; }
    s = AclStatus(aclrtSynchronizeStream(fftsStream_), "aclrtSynchronizeStream(sdma-direct)");
    if (s.Failure()) { return s; }
    inFlight_.clear();
    return Status::OK();
}

void AscendSdmaDirectCopier::Cleanup() noexcept
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

Status AscendSdmaDirectCopier::BuildHostToDeviceSpecs(
    const void* hostDevicePtr, void** devices, const std::vector<size_t>& sizes,
    std::vector<AscendFftsCopySpec>& specs) const
{
    if (!setup_) { return Status::Error("Cache SDMA Direct copier is not setup"); }
    if (hostDevicePtr == nullptr || devices == nullptr) {
        return Status::InvalidParam("invalid Cache SDMA Direct H2D pointers");
    }

    specs.reserve(sizes.size());
    size_t offset = 0;
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (sizes[i] == 0 || devices[i] == nullptr) {
            return Status::InvalidParam("invalid Cache SDMA Direct H2D tensor({})", i);
        }
        auto* src = static_cast<const std::byte*>(hostDevicePtr) + offset;
        specs.push_back({devices[i], src, sizes[i]});
        offset += sizes[i];
    }
    return Status::OK();
}

Status AscendSdmaDirectCopier::BuildDeviceToHostSpecs(
    void** devices, void* hostDevicePtr, const std::vector<size_t>& sizes,
    std::vector<AscendFftsCopySpec>& specs) const
{
    if (!setup_) { return Status::Error("Cache SDMA Direct copier is not setup"); }
    if (hostDevicePtr == nullptr || devices == nullptr) {
        return Status::InvalidParam("invalid Cache SDMA Direct D2H pointers");
    }

    specs.reserve(sizes.size());
    size_t offset = 0;
    for (size_t i = 0; i < sizes.size(); ++i) {
        if (sizes[i] == 0 || devices[i] == nullptr) {
            return Status::InvalidParam("invalid Cache SDMA Direct D2H tensor({})", i);
        }
        auto* dst = static_cast<std::byte*>(hostDevicePtr) + offset;
        specs.push_back({dst, devices[i], sizes[i]});
        offset += sizes[i];
    }
    return Status::OK();
}

Status AscendSdmaDirectCopier::LaunchSpecs(std::vector<AscendFftsCopySpec>&& specs)
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

Status AscendSdmaDirectCopier::LaunchPendingTask()
{
    if (pendingSpecs_.empty()) { return Status::OK(); }
    std::vector<AscendFftsCopySpec> specs;
    specs.swap(pendingSpecs_);
    return LaunchSpecs(std::move(specs));
}

Status AscendSdmaDirectCopier::AclStatus(aclError ret, const char* expr)
{
    if (ret == ACL_SUCCESS) { return Status::OK(); }
    UC_ERROR("Failed({}) to call {}.", static_cast<int32_t>(ret), expr);
    return Status{static_cast<int32_t>(ret), expr};
}

}  // namespace UC::Trans
