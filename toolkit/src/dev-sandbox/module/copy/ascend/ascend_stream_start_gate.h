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
#ifndef ASCEND_STREAM_START_GATE_H
#define ASCEND_STREAM_START_GATE_H

#include <cstddef>
#include <cstdint>
#include <vector>
#include "error_handle_ascend.h"

class AscendStreamStartGate {
    size_t deviceId_ = 0;
    aclrtStream controlStream_ = nullptr;
    aclrtStream releaseStream_ = nullptr;
    aclrtNotify releaseNotify_ = nullptr;
    aclrtEvent startEvent_ = nullptr;
    size_t streamCount_ = 0;
    std::vector<aclrtEvent> startEvents_;

public:
    void Setup(size_t deviceId, size_t streamCount, bool traceStart)
    {
        ASSERT(streamCount > 0);
        deviceId_ = deviceId;
        streamCount_ = streamCount;
        ASCEND_ASSERT(aclrtSetDevice(deviceId_));
        ASCEND_ASSERT(aclrtCreateStream(&controlStream_));
        ASCEND_ASSERT(aclrtCreateStream(&releaseStream_));
        ASCEND_ASSERT(aclrtCreateNotify(&releaseNotify_, ACL_NOTIFY_DEFAULT));
        ASCEND_ASSERT(aclrtCreateEventExWithFlag(&startEvent_, ACL_EVENT_SYNC));
        if (traceStart) {
            startEvents_.resize(streamCount);
            for (auto& event : startEvents_) {
                ASCEND_ASSERT(aclrtCreateEventExWithFlag(&event, ACL_EVENT_TIME_LINE));
            }
        }
    }

    void Prepare(aclrtEvent totalStart)
    {
        ASSERT(totalStart != nullptr);
        ASCEND_ASSERT(aclrtSetDevice(deviceId_));
        // Record the reusable Event before any data Stream waits on it. The control Stream
        // cannot complete the Event until the one-to-one Notify is released after the process
        // barrier, then the same Event completion broadcasts to every waiting data Stream.
        ASCEND_ASSERT(aclrtWaitAndResetNotify(releaseNotify_, controlStream_, 0));
        ASCEND_ASSERT(aclrtRecordEvent(totalStart, controlStream_));
        ASCEND_ASSERT(aclrtRecordEvent(startEvent_, controlStream_));
    }

    void Arm(size_t index, aclrtStream stream, bool recordStart)
    {
        ASSERT(index < streamCount_);
        ASSERT(!recordStart || index < startEvents_.size());
        ASCEND_ASSERT(aclrtSetDevice(deviceId_));
        ASCEND_ASSERT(aclrtStreamWaitEvent(stream, startEvent_));
        if (recordStart) { ASCEND_ASSERT(aclrtRecordEvent(startEvents_[index], stream)); }
    }

    void Release(bool)
    {
        ASCEND_ASSERT(aclrtSetDevice(deviceId_));
        ASCEND_ASSERT(aclrtRecordNotify(releaseNotify_, releaseStream_));
    }

    aclrtStream ControlStream() const { return controlStream_; }

    std::vector<uint64_t> StartTimestamps() const
    {
        std::vector<uint64_t> timestamps;
        timestamps.reserve(startEvents_.size());
        ASCEND_ASSERT(aclrtSetDevice(deviceId_));
        for (auto event : startEvents_) {
            uint64_t timestamp = 0;
            ASCEND_ASSERT(aclrtEventGetTimestamp(event, &timestamp));
            timestamps.push_back(timestamp);
        }
        return timestamps;
    }

    void Cleanup()
    {
        ASCEND_ASSERT(aclrtSetDevice(deviceId_));
        for (auto event : startEvents_) { ASCEND_ASSERT(aclrtDestroyEvent(event)); }
        startEvents_.clear();
        if (startEvent_ != nullptr) {
            ASCEND_ASSERT(aclrtDestroyEvent(startEvent_));
            startEvent_ = nullptr;
        }
        if (releaseNotify_ != nullptr) {
            ASCEND_ASSERT(aclrtDestroyNotify(releaseNotify_));
            releaseNotify_ = nullptr;
        }
        if (releaseStream_ != nullptr) {
            ASCEND_ASSERT(aclrtDestroyStream(releaseStream_));
            releaseStream_ = nullptr;
        }
        if (controlStream_ != nullptr) {
            ASCEND_ASSERT(aclrtDestroyStream(controlStream_));
            controlStream_ = nullptr;
        }
        streamCount_ = 0;
    }
};

#endif  // ASCEND_STREAM_START_GATE_H
