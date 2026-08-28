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
#ifndef COPY_INSTANCE_FFTS_DIRECT_H2D_ASCEND_H
#define COPY_INSTANCE_FFTS_DIRECT_H2D_ASCEND_H

#include <algorithm>
#include <chrono>
#include <string>
#include <utility>
#include <vector>
#include "ascend/ascend_stream_start_gate.h"
#include "ascend/error_handle_ascend.h"
#include "copy_instance.h"
#include "copy_sync_mode.h"
#include "ffts_d2d_dispatcher_ascend.h"
#include "glm_io_buffer_ascend.h"
#include "mapped_host_buffer_ffts_direct_h2d_ascend.h"

class FftsDirectH2DCopyInstance : public CopyInstance {
protected:
    struct DirectContext {
        size_t deviceId = 0;
        aclrtStream stream = nullptr;
        aclrtEvent endEvent = nullptr;
        std::vector<std::vector<AscendFftsCopySpec>> tasks;
        FftsD2DDispatcher dispatcher;
    };

    std::vector<DirectContext> contexts_;
    AscendStreamStartGate startGate_;
    aclrtEvent totalStart_ = nullptr;
    aclrtEvent totalEnd_ = nullptr;
    size_t fragsPerTask_ = 0;
    size_t streamCount_ = 1;
    size_t laneCount_ = 0;
    CopySyncMode syncMode_ = CopySyncMode::EVENT;

    void Prepare(const std::vector<const CopyBuffer*>& srcBuffers,
                 const std::vector<const CopyBuffer*>& dstBuffers) override
    {
        ASSERT(!srcBuffers.empty());
        ASSERT(srcBuffers.size() == dstBuffers.size());
        ASSERT(streamCount_ > 0);

        contexts_.clear();
        contexts_.reserve(srcBuffers.size());
        for (size_t i = 0; i < srcBuffers.size(); ++i) {
            const auto* src = srcBuffers[i];
            const auto* dst = dstBuffers[i];
            ASSERT(src != nullptr);
            ASSERT(dst != nullptr);
            ASSERT(src->Number() == dst->Number());
            ASSERT(src->Size() == dst->Size());

            const auto* glmSrc = dynamic_cast<const GlmSharedHostCopyBuffer*>(src);
            const auto* glmDst = dynamic_cast<const GlmIoCopyBuffer*>(dst);
            if (glmSrc != nullptr || glmDst != nullptr) {
                ASSERT(glmSrc != nullptr);
                ASSERT(glmDst != nullptr);
                const auto taskCount = glmSrc->Number();
                ASSERT(taskCount > 0);
                const auto activeStreamCount = std::min(streamCount_, taskCount);
                const auto firstContext = contexts_.size();
                const auto deviceId = AffinityDeviceId(*src, *dst);

                ASCEND_ASSERT(aclrtSetDevice(deviceId));
                for (size_t stream = 0; stream < activeStreamCount; ++stream) {
                    DirectContext ctx;
                    ctx.deviceId = deviceId;
                    ctx.dispatcher.SetMaxReadyLanes(laneCount_);
                    ASCEND_ASSERT(aclrtCreateStream(&ctx.stream));
                    if (syncMode_ == CopySyncMode::EVENT) {
                        ASCEND_ASSERT(aclrtCreateEvent(&ctx.endEvent));
                    }
                    ctx.tasks.reserve((taskCount + activeStreamCount - 1 - stream) /
                                      activeStreamCount);
                    contexts_.push_back(std::move(ctx));
                }

                for (size_t task = 0; task < taskCount; ++task) {
                    std::vector<AscendFftsCopySpec> copies;
                    copies.reserve(GlmIoCopyBuffer::IoCount());
                    for (size_t io = 0; io < GlmIoCopyBuffer::IoCount(); ++io) {
                        ASSERT(glmSrc->IoSize(io) == glmDst->IoSize(io));
                        copies.push_back({glmDst->IoAt(task, io), glmSrc->MappedIoAt(task, io),
                                          glmSrc->IoSize(io)});
                    }
                    contexts_[firstContext + task % activeStreamCount].tasks.push_back(
                        std::move(copies));
                }
                continue;
            }

            const auto* mappedSrc = dynamic_cast<const FftsDirectMappedHostBuffer*>(src);
            ASSERT(mappedSrc != nullptr);

            const auto deviceId = AffinityDeviceId(*src, *dst);
            const auto size = src->Size();
            const auto number = src->Number();
            ASSERT(number > 0);

            const auto taskFrags = fragsPerTask_ == 0 ? number : fragsPerTask_;
            ASSERT(taskFrags > 0);
            ASSERT(number % taskFrags == 0);
            const auto taskCount = number / taskFrags;
            const auto activeStreamCount = std::min(streamCount_, taskCount);
            const auto firstContext = contexts_.size();

            ASCEND_ASSERT(aclrtSetDevice(deviceId));
            for (size_t stream = 0; stream < activeStreamCount; ++stream) {
                DirectContext ctx;
                ctx.deviceId = deviceId;
                ctx.dispatcher.SetMaxReadyLanes(laneCount_);
                ASCEND_ASSERT(aclrtCreateStream(&ctx.stream));
                if (syncMode_ == CopySyncMode::EVENT) {
                    ASCEND_ASSERT(aclrtCreateEvent(&ctx.endEvent));
                }
                ctx.tasks.reserve((taskCount + activeStreamCount - 1 - stream) /
                                  activeStreamCount);
                contexts_.push_back(std::move(ctx));
            }

            size_t taskIndex = 0;
            for (size_t first = 0; first < number; first += taskFrags) {
                std::vector<AscendFftsCopySpec> copies;
                copies.reserve(taskFrags);
                for (size_t fragment = first; fragment < first + taskFrags; ++fragment) {
                    copies.push_back({(*dst)[fragment], mappedSrc->MappedAt(fragment), size});
                }
                const auto stream = taskIndex % activeStreamCount;
                contexts_[firstContext + stream].tasks.push_back(std::move(copies));
                ++taskIndex;
            }
        }

        ASCEND_ASSERT(aclrtSetDevice(contexts_[0].deviceId));
        for (const auto& ctx : contexts_) { ASSERT(ctx.deviceId == contexts_[0].deviceId); }
        startGate_.Setup(contexts_[0].deviceId, contexts_.size());
        ASCEND_ASSERT(aclrtCreateEvent(&totalStart_));
        ASCEND_ASSERT(aclrtCreateEvent(&totalEnd_));
    }

    void Cleanup() override
    {
        startGate_.Cleanup();
        for (auto& ctx : contexts_) {
            ASCEND_ASSERT(aclrtSetDevice(ctx.deviceId));
            if (ctx.endEvent != nullptr) {
                ASCEND_ASSERT(aclrtDestroyEvent(ctx.endEvent));
                ctx.endEvent = nullptr;
            }
            if (ctx.stream != nullptr) {
                ASCEND_ASSERT(aclrtDestroyStream(ctx.stream));
                ctx.stream = nullptr;
            }
        }
        if (!contexts_.empty()) { ASCEND_ASSERT(aclrtSetDevice(contexts_[0].deviceId)); }
        if (totalStart_ != nullptr) {
            ASCEND_ASSERT(aclrtDestroyEvent(totalStart_));
            totalStart_ = nullptr;
        }
        if (totalEnd_ != nullptr) {
            ASCEND_ASSERT(aclrtDestroyEvent(totalEnd_));
            totalEnd_ = nullptr;
        }
        contexts_.clear();
    }

    std::pair<size_t, size_t> DoCopyOnce() override
    {
        using namespace std::chrono;

        ASCEND_ASSERT(aclrtSetDevice(contexts_[0].deviceId));
        for (size_t i = 0; i < contexts_.size(); ++i) {
            startGate_.Arm(i, contexts_[i].stream);
        }

        const auto submitStart = steady_clock::now();
        for (auto& ctx : contexts_) { SubmitContext(ctx); }
        const auto submitCost = static_cast<size_t>(
            duration_cast<microseconds>(steady_clock::now() - submitStart).count());

        WaitForProcessReadyBarrier();
        ASCEND_ASSERT(aclrtSetDevice(contexts_[0].deviceId));
        ASCEND_ASSERT(aclrtRecordEvent(totalStart_, startGate_.ControlStream()));
        startGate_.Release();

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
        ASCEND_ASSERT(aclrtSynchronizeStream(startGate_.ControlStream()));

        float copyCostMs = 0.0f;
        ASCEND_ASSERT(aclrtEventElapsedTime(&copyCostMs, totalStart_, totalEnd_));
        const auto copyCost = static_cast<size_t>(copyCostMs * 1000);
        return {copyCost, submitCost};
    }

    void SubmitContext(DirectContext& ctx) const
    {
        ASCEND_ASSERT(aclrtSetDevice(ctx.deviceId));
        for (const auto& copies : ctx.tasks) {
            const auto readyCount = ctx.dispatcher.BuildCopies(copies);
            ASSERT(readyCount > 0);
            ctx.dispatcher.Launch(ctx.stream, readyCount);
        }
        if (syncMode_ == CopySyncMode::EVENT) {
            ASCEND_ASSERT(aclrtRecordEvent(ctx.endEvent, ctx.stream));
        }
    }

public:
    FftsDirectH2DCopyInstance(size_t iterations, bool affinitySrc, size_t fragsPerTask = 0,
                              size_t streamCount = 1,
                              CopySyncMode syncMode = CopySyncMode::EVENT,
                              size_t laneCount = 0)
        : CopyInstance(iterations, affinitySrc),
          fragsPerTask_(fragsPerTask),
          streamCount_(streamCount),
          laneCount_(laneCount),
          syncMode_(syncMode)
    {
    }

    std::string Name() const override
    {
        auto name = "ffts-direct-h2d-" + std::to_string(streamCount_) + "s";
        if (laneCount_ > 0) { name += "-L" + std::to_string(laneCount_); }
        return name;
    }
};

#endif  // COPY_INSTANCE_FFTS_DIRECT_H2D_ASCEND_H
