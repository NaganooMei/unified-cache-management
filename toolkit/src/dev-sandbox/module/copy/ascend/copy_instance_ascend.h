/**
 * MIT License
 *
 * Copyright (c) 2026 Mag1c.H
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
#ifndef COPY_INSTANCE_ASCEND_H
#define COPY_INSTANCE_ASCEND_H

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>
#include "copy_buffer.h"
#include "copy_instance.h"
#include "copy_start_trace_ascend.h"
#include "copy_sync_mode.h"
#include "ascend_stream_start_gate.h"
#include "error_handle_ascend.h"
#include "glm_io_buffer_ascend.h"

struct AscendStreamContext {
    size_t deviceId = 0;
    aclrtStream stream = nullptr;
    aclrtEvent endEvent = nullptr;
    size_t size = 0;
    std::vector<void*> src;
    std::vector<void*> dst;
    std::vector<size_t> sizes;
};

class AscendCopyInstanceBase : public CopyInstance {
protected:
    std::vector<AscendStreamContext> contexts_;
    aclrtEvent totalStart_;
    aclrtEvent totalEnd_;

    void Prepare(const std::vector<const CopyBuffer*>& srcBuffers,
                 const std::vector<const CopyBuffer*>& dstBuffers) override
    {
        contexts_.clear();
        const auto bufferNumber = srcBuffers.size();
        for (size_t i = 0; i < bufferNumber; i++) {
            auto& src = *srcBuffers[i];
            auto& dst = *dstBuffers[i];
            ASSERT(src.Number() == dst.Number());
            ASSERT(src.Size() == dst.Size());

            AscendStreamContext ctx;
            ctx.deviceId = AffinityDeviceId(src, dst);
            ctx.size = src.Size();
            ASCEND_ASSERT(aclrtSetDevice(ctx.deviceId));
            ASCEND_ASSERT(aclrtCreateStream(&ctx.stream));
            ASCEND_ASSERT(aclrtCreateEventExWithFlag(&ctx.endEvent, ACL_EVENT_SYNC));
            ctx.src.reserve(src.Number());
            ctx.dst.reserve(dst.Number());
            for (size_t j = 0; j < src.Number(); j++) {
                ctx.src.push_back(src[j]);
                ctx.dst.push_back(dst[j]);
            }
            contexts_.push_back(std::move(ctx));
        }

        ASCEND_ASSERT(aclrtSetDevice(contexts_[0].deviceId));
        ASCEND_ASSERT(aclrtCreateEventExWithFlag(
            &totalStart_, static_cast<uint32_t>(ACL_EVENT_TIME_LINE | ACL_EVENT_SYNC)));
        ASCEND_ASSERT(aclrtCreateEventExWithFlag(&totalEnd_, ACL_EVENT_TIME_LINE));
    }

    void Cleanup() override
    {
        for (auto& ctx : contexts_) {
            ASCEND_ASSERT(aclrtSetDevice(ctx.deviceId));
            ASCEND_ASSERT(aclrtDestroyEvent(ctx.endEvent));
            ASCEND_ASSERT(aclrtDestroyStream(ctx.stream));
        }
        ASCEND_ASSERT(aclrtSetDevice(contexts_[0].deviceId));
        ASCEND_ASSERT(aclrtDestroyEvent(totalStart_));
        ASCEND_ASSERT(aclrtDestroyEvent(totalEnd_));
        contexts_.clear();
    }

    std::pair<size_t, size_t> DoCopyOnce() override
    {
        using namespace std::chrono;

        ASCEND_ASSERT(aclrtSetDevice(contexts_[0].deviceId));
        ASCEND_ASSERT(aclrtRecordEvent(totalStart_, contexts_[0].stream));

        for (size_t i = 1; i < contexts_.size(); i++) {
            ASCEND_ASSERT(aclrtSetDevice(contexts_[i].deviceId));
            ASCEND_ASSERT(aclrtStreamWaitEvent(contexts_[i].stream, totalStart_));
        }

        auto submitStart = steady_clock::now();
        for (auto& ctx : contexts_) {
            ASCEND_ASSERT(aclrtSetDevice(ctx.deviceId));
            CopyInternal(ctx);
        }
        auto submitCost = duration_cast<microseconds>(steady_clock::now() - submitStart).count();

        for (size_t i = 1; i < contexts_.size(); i++) {
            ASCEND_ASSERT(aclrtSetDevice(contexts_[i].deviceId));
            ASCEND_ASSERT(aclrtRecordEvent(contexts_[i].endEvent, contexts_[i].stream));
            ASCEND_ASSERT(aclrtSetDevice(contexts_[0].deviceId));
            ASCEND_ASSERT(aclrtStreamWaitEvent(contexts_[0].stream, contexts_[i].endEvent));
        }

        ASCEND_ASSERT(aclrtSetDevice(contexts_[0].deviceId));
        ASCEND_ASSERT(aclrtRecordEvent(totalEnd_, contexts_[0].stream));
        SynchronizeInternal(contexts_[0]);

        float copyCostMs = 0.f;
        ASCEND_ASSERT(aclrtEventElapsedTime(&copyCostMs, totalStart_, totalEnd_));
        size_t copyCost = static_cast<size_t>(copyCostMs * 1000);

        return {copyCost, submitCost};
    }

    virtual void CopyInternal(const AscendStreamContext& ctx) = 0;
    virtual void SynchronizeInternal(const AscendStreamContext& ctx) = 0;

public:
    AscendCopyInstanceBase(size_t iterations, bool affinitySrc)
        : CopyInstance(iterations, affinitySrc)
    {
    }
};

class H2DCECopyInstance : public AscendCopyInstanceBase {
protected:
    void CopyInternal(const AscendStreamContext& ctx) override
    {
        for (size_t i = 0; i < ctx.src.size(); i++) {
            ASCEND_ASSERT(aclrtMemcpyAsync(ctx.dst[i], ctx.size, ctx.src[i], ctx.size,
                                           ACL_MEMCPY_HOST_TO_DEVICE, ctx.stream));
        }
    }

    void SynchronizeInternal(const AscendStreamContext& ctx) override
    {
        ASCEND_ASSERT(aclrtSynchronizeStream(ctx.stream));
    }

public:
    H2DCECopyInstance(size_t iterations, bool affinitySrc)
        : AscendCopyInstanceBase(iterations, affinitySrc)
    {
    }

    std::string Name() const override { return "CE"; }
};

class H2DBatchCECopyInstance : public AscendCopyInstanceBase {
protected:
    size_t targetDevice_;

    void CopyInternal(const AscendStreamContext& ctx) override
    {
        aclrtMemcpyBatchAttr attr;
        std::memset(&attr, 0, sizeof(attr));
        attr.srcLoc.type = ACL_MEM_LOCATION_TYPE_HOST;
        attr.dstLoc.type = ACL_MEM_LOCATION_TYPE_DEVICE;
        attr.dstLoc.id = targetDevice_;

        std::vector<aclrtMemcpyBatchAttr> attrArray{attr};
        std::vector<size_t> attrIdxArray(ctx.src.size(), 0);
        std::vector<size_t> sizeArray(ctx.src.size(), ctx.size);
        size_t failureIdx = 0;

        ASCEND_ASSERT(aclrtMemcpyBatchAsync(const_cast<void**>(ctx.dst.data()), sizeArray.data(),
                                            const_cast<void**>(ctx.src.data()), sizeArray.data(),
                                            ctx.src.size(), attrArray.data(), attrIdxArray.data(),
                                            attrArray.size(), &failureIdx, ctx.stream));
    }

    void SynchronizeInternal(const AscendStreamContext& ctx) override
    {
        ASCEND_ASSERT(aclrtSynchronizeStream(ctx.stream));
    }

public:
    H2DBatchCECopyInstance(size_t iterations, bool affinitySrc, size_t targetDevice)
        : AscendCopyInstanceBase(iterations, affinitySrc), targetDevice_(targetDevice)
    {
    }

    std::string Name() const override { return "BatchCE"; }
};

class D2DCECopyInstance : public AscendCopyInstanceBase {
protected:
    void CopyInternal(const AscendStreamContext& ctx) override
    {
        for (size_t i = 0; i < ctx.src.size(); i++) {
            ASCEND_ASSERT(aclrtMemcpyAsync(ctx.dst[i], ctx.size, ctx.src[i], ctx.size,
                                           ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream));
        }
    }

    void SynchronizeInternal(const AscendStreamContext& ctx) override
    {
        ASCEND_ASSERT(aclrtSynchronizeStream(ctx.stream));
    }

public:
    D2DCECopyInstance(size_t iterations, bool affinitySrc)
        : AscendCopyInstanceBase(iterations, affinitySrc)
    {
    }

    std::string Name() const override { return "CE"; }
};

class H2DCEMultiStreamCopyInstance : public CopyInstance {
protected:
    std::vector<AscendStreamContext> contexts_;
    AscendStreamStartGate startGate_;
    aclrtEvent totalStart_;
    aclrtEvent totalEnd_;
    size_t streamCount_;
    CopySyncMode syncMode_;

    void Prepare(const std::vector<const CopyBuffer*>& srcBuffers,
                 const std::vector<const CopyBuffer*>& dstBuffers) override
    {
        contexts_.clear();
        const auto bufferNumber = srcBuffers.size();
        contexts_.reserve(bufferNumber * streamCount_);

        for (size_t i = 0; i < bufferNumber; i++) {
            auto& src = *srcBuffers[i];
            auto& dst = *dstBuffers[i];
            ASSERT(src.Number() == dst.Number());

            const auto* glmSrc = dynamic_cast<const GlmIoCopyBuffer*>(&src);
            const auto* glmDst = dynamic_cast<const GlmIoCopyBuffer*>(&dst);
            if (glmSrc != nullptr || glmDst != nullptr) {
                ASSERT(glmSrc != nullptr);
                ASSERT(glmDst != nullptr);
                const auto taskCount = glmSrc->Number();
                ASSERT(taskCount > 0);
                const auto activeStreamCount = std::min(streamCount_, taskCount);
                const auto firstContext = contexts_.size();
                const auto deviceId = AffinityDeviceId(src, dst);
                ASCEND_ASSERT(aclrtSetDevice(deviceId));

                for (size_t stream = 0; stream < activeStreamCount; ++stream) {
                    AscendStreamContext ctx;
                    ctx.deviceId = deviceId;
                    ASCEND_ASSERT(aclrtCreateStream(&ctx.stream));
                    if (syncMode_ == CopySyncMode::EVENT) {
                        ASCEND_ASSERT(
                            aclrtCreateEventExWithFlag(&ctx.endEvent, ACL_EVENT_SYNC));
                    }
                    contexts_.push_back(std::move(ctx));
                }

                for (size_t task = 0; task < taskCount; ++task) {
                    auto& streamContext = contexts_[firstContext + task % activeStreamCount];
                    for (size_t io = 0; io < GlmIoCopyBuffer::IoCount(); ++io) {
                        ASSERT(glmSrc->IoSize(io) == glmDst->IoSize(io));
                        streamContext.src.push_back(glmSrc->IoAt(task, io));
                        streamContext.dst.push_back(glmDst->IoAt(task, io));
                        streamContext.sizes.push_back(glmSrc->IoSize(io));
                    }
                }
                continue;
            }

            ASSERT(src.Size() == dst.Size());

            size_t bufferCount = src.Number();
            size_t base = bufferCount / streamCount_;
            size_t remainder = bufferCount % streamCount_;
            size_t deviceId = AffinityDeviceId(src, dst);
            ASCEND_ASSERT(aclrtSetDevice(deviceId));

            size_t offset = 0;
            for (size_t s = 0; s < streamCount_; s++) {
                size_t count = base + (s < remainder ? 1 : 0);
                if (count == 0) continue;

                AscendStreamContext ctx;
                ctx.deviceId = deviceId;
                ctx.size = src.Size();
                ASCEND_ASSERT(aclrtCreateStream(&ctx.stream));
                if (syncMode_ == CopySyncMode::EVENT) {
                    ASCEND_ASSERT(aclrtCreateEventExWithFlag(&ctx.endEvent, ACL_EVENT_SYNC));
                }
                ctx.src.reserve(count);
                ctx.dst.reserve(count);
                for (size_t j = 0; j < count; j++) {
                    ctx.src.push_back(src[offset + j]);
                    ctx.dst.push_back(dst[offset + j]);
                }
                contexts_.push_back(std::move(ctx));
                offset += count;
            }
        }

        ASCEND_ASSERT(aclrtSetDevice(contexts_[0].deviceId));
        for (const auto& ctx : contexts_) { ASSERT(ctx.deviceId == contexts_[0].deviceId); }
        startGate_.Setup(contexts_[0].deviceId, contexts_.size(), StartTraceEnabled());
        ASCEND_ASSERT(aclrtCreateEventExWithFlag(&totalStart_, ACL_EVENT_TIME_LINE));
        ASCEND_ASSERT(aclrtCreateEventExWithFlag(&totalEnd_, ACL_EVENT_TIME_LINE));
    }

    void Cleanup() override
    {
        startGate_.Cleanup();
        for (auto& ctx : contexts_) {
            ASCEND_ASSERT(aclrtSetDevice(ctx.deviceId));
            if (ctx.endEvent != nullptr) { ASCEND_ASSERT(aclrtDestroyEvent(ctx.endEvent)); }
            ASCEND_ASSERT(aclrtDestroyStream(ctx.stream));
        }
        ASCEND_ASSERT(aclrtSetDevice(contexts_[0].deviceId));
        ASCEND_ASSERT(aclrtDestroyEvent(totalStart_));
        ASCEND_ASSERT(aclrtDestroyEvent(totalEnd_));
        contexts_.clear();
    }

    std::pair<size_t, size_t> DoCopyOnce() override
    {
        using namespace std::chrono;

        const bool traceStart = ShouldTraceStart();
        ASCEND_ASSERT(aclrtSetDevice(contexts_[0].deviceId));
        startGate_.Prepare(totalStart_);
        for (size_t i = 0; i < contexts_.size(); ++i) {
            startGate_.Arm(i, contexts_[i].stream, traceStart);
        }

        auto submitStart = steady_clock::now();
        for (auto& ctx : contexts_) {
            ASCEND_ASSERT(aclrtSetDevice(ctx.deviceId));
            for (size_t i = 0; i < ctx.src.size(); i++) {
                const auto size = ctx.sizes.empty() ? ctx.size : ctx.sizes[i];
                ASCEND_ASSERT(aclrtMemcpyAsync(ctx.dst[i], size, ctx.src[i], size,
                                               ACL_MEMCPY_HOST_TO_DEVICE, ctx.stream));
            }
        }
        auto submitCost = duration_cast<microseconds>(steady_clock::now() - submitStart).count();

        if (syncMode_ == CopySyncMode::EVENT) {
            for (size_t i = 0; i < contexts_.size(); i++) {
                ASCEND_ASSERT(aclrtSetDevice(contexts_[i].deviceId));
                ASCEND_ASSERT(aclrtRecordEvent(contexts_[i].endEvent, contexts_[i].stream));
            }
        }

        const uint64_t barrierEnterNs = traceStart ? CopyStartMonotonicNs() : 0;
        WaitForProcessReadyBarrier();
        const uint64_t barrierExitNs = traceStart ? CopyStartMonotonicNs() : 0;
        ASCEND_ASSERT(aclrtSetDevice(contexts_[0].deviceId));
        const uint64_t wallStartNs = CopyStartMonotonicNs();
        startGate_.Release(traceStart);
        const uint64_t releaseSubmitNs = CopyStartMonotonicNs();

        if (syncMode_ == CopySyncMode::EVENT) {
            for (const auto& ctx : contexts_) {
                ASCEND_ASSERT(aclrtStreamWaitEvent(startGate_.ControlStream(), ctx.endEvent));
            }
        } else {
            for (auto& ctx : contexts_) {
                ASCEND_ASSERT(aclrtSetDevice(ctx.deviceId));
                ASCEND_ASSERT(aclrtSynchronizeStream(ctx.stream));
            }
        }

        ASCEND_ASSERT(aclrtSetDevice(contexts_[0].deviceId));
        ASCEND_ASSERT(aclrtRecordEvent(totalEnd_, startGate_.ControlStream()));
        const uint64_t syncEnterNs = CopyStartMonotonicNs();
        ASCEND_ASSERT(aclrtSynchronizeStream(startGate_.ControlStream()));
        const uint64_t wallEndNs = CopyStartMonotonicNs();
        SetWallClockRange(wallStartNs, wallEndNs);

        float copyCostMs = 0.f;
        ASCEND_ASSERT(aclrtEventElapsedTime(&copyCostMs, totalStart_, totalEnd_));
        size_t copyCost = static_cast<size_t>(copyCostMs * 1000);

        if (traceStart) {
            for (const auto& ctx : contexts_) {
                ASCEND_ASSERT(aclrtSetDevice(ctx.deviceId));
                ASCEND_ASSERT(aclrtSynchronizeStream(ctx.stream));
            }
            ASCEND_ASSERT(aclrtSetDevice(contexts_[0].deviceId));
            EmitCopyStartTrace(Name(), contexts_[0].deviceId, CurrentIteration(), barrierEnterNs,
                               barrierExitNs, wallStartNs, releaseSubmitNs, syncEnterNs, wallEndNs,
                               copyCost, startGate_.StartTimestamps());
        }

        return {copyCost, submitCost};
    }

public:
    H2DCEMultiStreamCopyInstance(size_t iterations, bool affinitySrc, size_t streamCount,
                                 CopySyncMode syncMode = CopySyncMode::EVENT)
        : CopyInstance(iterations, affinitySrc), streamCount_(streamCount), syncMode_(syncMode)
    {
    }

    std::string Name() const override { return "CE-MS" + std::to_string(streamCount_); }
};

#endif  // COPY_INSTANCE_ASCEND_H
