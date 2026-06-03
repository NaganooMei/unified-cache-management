/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "ascend_h2d_ffts_pipeline.h"
#include <numeric>
#include "logger/logger.h"

namespace UC::Trans {

namespace {
Status AclStatus(aclError ret, const char* expr)
{
    if (ret == ACL_SUCCESS) { return Status::OK(); }
    UC_ERROR("Failed({}) to call {}.", static_cast<int32_t>(ret), expr);
    return Status{static_cast<int32_t>(ret), expr};
}
}  // namespace

AscendH2DFftsPipeline::~AscendH2DFftsPipeline() { Cleanup(); }

Status AscendH2DFftsPipeline::Setup(const AscendH2DFftsPipelineConfig& config)
{
    Cleanup();
    if (config.deviceId < 0) { return Status::InvalidParam("invalid device id"); }
    if (config.streamNumber == 0) { return Status::InvalidParam("invalid stream number"); }
    if (config.pipelineDepth == 0) { return Status::InvalidParam("invalid pipeline depth"); }
    if (config.maxReadyLanes == 0) { return Status::InvalidParam("invalid max ready lanes"); }
    if (config.objectBytes == 0) { return Status::InvalidParam("invalid object bytes"); }
    if (config.maxFragments == 0) { return Status::InvalidParam("invalid max fragments"); }

    deviceId_ = config.deviceId;
    streamNumber_ = config.streamNumber;
    pipelineDepth_ = config.pipelineDepth;
    maxReadyLanes_ = config.maxReadyLanes;
    objectBytes_ = config.objectBytes;
    maxFragments_ = config.maxFragments;
    nextObjectIndex_ = 0;

    auto s = AclStatus(aclrtSetDevice(deviceId_), "aclrtSetDevice");
    if (s.Failure()) { return s; }
    lanes_.resize(streamNumber_);
    for (auto& lane : lanes_) {
        s = AclStatus(aclrtCreateStream(&lane.h2dStream), "aclrtCreateStream(h2d)");
        if (s.Failure()) {
            Cleanup();
            return s;
        }
        s = AclStatus(aclrtCreateStream(&lane.fftsStream), "aclrtCreateStream(ffts)");
        if (s.Failure()) {
            Cleanup();
            return s;
        }

        lane.nextSlotIndex = 0;
        lane.stagingBuffers.assign(pipelineDepth_, nullptr);
        lane.slotReady.assign(pipelineDepth_, nullptr);
        lane.slotFree.assign(pipelineDepth_, nullptr);
        for (size_t slot = 0; slot < pipelineDepth_; ++slot) {
            s = AclStatus(aclrtMalloc(&lane.stagingBuffers[slot], objectBytes_,
                                      ACL_MEM_TYPE_HIGH_BAND_WIDTH),
                          "aclrtMalloc(staging)");
            if (s.Failure()) {
                Cleanup();
                return s;
            }
            s = AclStatus(aclrtCreateEvent(&lane.slotReady[slot]), "aclrtCreateEvent(slotReady)");
            if (s.Failure()) {
                Cleanup();
                return s;
            }
            s = AclStatus(aclrtCreateEvent(&lane.slotFree[slot]), "aclrtCreateEvent(slotFree)");
            if (s.Failure()) {
                Cleanup();
                return s;
            }
            s = AclStatus(aclrtRecordEvent(lane.slotFree[slot], lane.h2dStream),
                          "aclrtRecordEvent(slotFree)");
            if (s.Failure()) {
                Cleanup();
                return s;
            }
        }
    }

    setup_ = true;
    return Status::OK();
}

Status AscendH2DFftsPipeline::BuildCopySpecs(InFlightObject& object, void* staging,
                                             void** devices,
                                             const std::vector<size_t>& sizes) const
{
    if (staging == nullptr || devices == nullptr) {
        return Status::InvalidParam("invalid H2D FFTS pipeline pointers");
    }
    if (sizes.empty() || sizes.size() > maxFragments_) {
        return Status::InvalidParam("invalid H2D FFTS fragment number({})", sizes.size());
    }

    object.specs.clear();
    object.specs.reserve(sizes.size());
    auto* stagingBase = static_cast<uint8_t*>(staging);
    size_t offset = 0;
    for (size_t i = 0; i < sizes.size(); ++i) {
        const auto size = sizes[i];
        if (size == 0 || devices[i] == nullptr) {
            return Status::InvalidParam("invalid H2D FFTS fragment({})", i);
        }
        if (offset > objectBytes_ || size > objectBytes_ - offset) {
            return Status::InvalidParam("H2D FFTS object bytes overflow");
        }
        object.specs.push_back({devices[i], stagingBase + offset, size});
        offset += size;
    }
    if (offset == 0 || offset > objectBytes_) {
        return Status::InvalidParam("invalid H2D FFTS object bytes({})", offset);
    }
    return Status::OK();
}

Status AscendH2DFftsPipeline::SubmitObject(void* host, void** devices,
                                           const std::vector<size_t>& sizes)
{
    if (!setup_) { return Status::Error("H2D FFTS pipeline is not setup"); }
    if (host == nullptr) { return Status::InvalidParam("invalid H2D FFTS host pointer"); }

    auto& lane = lanes_[nextObjectIndex_ % streamNumber_];
    const auto slot = lane.nextSlotIndex % pipelineDepth_;
    auto* staging = lane.stagingBuffers[slot];
    auto object = std::make_unique<InFlightObject>();
    auto s = BuildCopySpecs(*object, staging, devices, sizes);
    if (s.Failure()) { return s; }

    const auto objectBytes =
        std::accumulate(sizes.begin(), sizes.end(), static_cast<size_t>(0));
    uint16_t readyCount = 0;
    s = object->dispatcher.BuildCopies(object->specs, maxReadyLanes_, readyCount);
    if (s.Failure()) { return s; }

    s = AclStatus(aclrtSetDevice(deviceId_), "aclrtSetDevice");
    if (s.Failure()) { return s; }
    s = AclStatus(aclrtStreamWaitEvent(lane.h2dStream, lane.slotFree[slot]),
                  "aclrtStreamWaitEvent(slotFree)");
    if (s.Failure()) { return s; }
    s = AclStatus(aclrtMemcpyAsync(staging, objectBytes_, host, objectBytes,
                                   ACL_MEMCPY_HOST_TO_DEVICE, lane.h2dStream),
                  "aclrtMemcpyAsync(H2D staging)");
    if (s.Failure()) { return s; }
    s = AclStatus(aclrtRecordEvent(lane.slotReady[slot], lane.h2dStream),
                  "aclrtRecordEvent(slotReady)");
    if (s.Failure()) { return s; }
    s = AclStatus(aclrtStreamWaitEvent(lane.fftsStream, lane.slotReady[slot]),
                  "aclrtStreamWaitEvent(slotReady)");
    if (s.Failure()) { return s; }

    s = object->dispatcher.Launch(lane.fftsStream, readyCount);
    if (s.Failure()) {
        auto release = AclStatus(aclrtRecordEvent(lane.slotFree[slot], lane.fftsStream),
                                 "aclrtRecordEvent(slotFree after failed launch)");
        return release.Failure() ? release : s;
    }
    s = AclStatus(aclrtRecordEvent(lane.slotFree[slot], lane.fftsStream),
                  "aclrtRecordEvent(slotFree)");
    if (s.Failure()) { return s; }

    lane.inFlight.push_back(std::move(object));
    ++lane.nextSlotIndex;
    ++nextObjectIndex_;
    return Status::OK();
}

Status AscendH2DFftsPipeline::Synchronize()
{
    if (!setup_) { return Status::OK(); }
    auto s = AclStatus(aclrtSetDevice(deviceId_), "aclrtSetDevice");
    if (s.Failure()) { return s; }
    for (auto& lane : lanes_) {
        for (auto event : lane.slotFree) {
            s = AclStatus(aclrtStreamWaitEvent(lane.h2dStream, event),
                          "aclrtStreamWaitEvent(slotFree)");
            if (s.Failure()) { return s; }
        }
        s = AclStatus(aclrtSynchronizeStream(lane.h2dStream), "aclrtSynchronizeStream(h2d)");
        if (s.Failure()) { return s; }
        s = AclStatus(aclrtSynchronizeStream(lane.fftsStream), "aclrtSynchronizeStream(ffts)");
        if (s.Failure()) { return s; }
        lane.inFlight.clear();
    }
    return Status::OK();
}

void AscendH2DFftsPipeline::Cleanup() noexcept
{
    if (deviceId_ >= 0) { (void)aclrtSetDevice(deviceId_); }
    for (auto& lane : lanes_) {
        for (auto event : lane.slotReady) {
            if (event != nullptr) { (void)aclrtDestroyEvent(event); }
        }
        for (auto event : lane.slotFree) {
            if (event != nullptr) { (void)aclrtDestroyEvent(event); }
        }
        for (auto* buffer : lane.stagingBuffers) {
            if (buffer != nullptr) { (void)aclrtFree(buffer); }
        }
        if (lane.h2dStream != nullptr) { (void)aclrtDestroyStream(lane.h2dStream); }
        if (lane.fftsStream != nullptr) { (void)aclrtDestroyStream(lane.fftsStream); }
    }

    setup_ = false;
    deviceId_ = -1;
    streamNumber_ = 0;
    pipelineDepth_ = 0;
    maxReadyLanes_ = 0;
    objectBytes_ = 0;
    maxFragments_ = 0;
    nextObjectIndex_ = 0;
    lanes_.clear();
}

}  // namespace UC::Trans
