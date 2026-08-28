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
    static constexpr uint64_t kDefaultNotifyFlag = 0;

    size_t deviceId_ = 0;
    aclrtStream controlStream_ = nullptr;
    std::vector<aclrtNotify> notifies_;
    std::vector<aclrtEvent> startEvents_;

public:
    void Setup(size_t deviceId, size_t streamCount, bool traceStart)
    {
        ASSERT(streamCount > 0);
        deviceId_ = deviceId;
        ASCEND_ASSERT(aclrtSetDevice(deviceId_));
        ASCEND_ASSERT(aclrtCreateStream(&controlStream_));
        notifies_.resize(streamCount);
        for (auto& notify : notifies_) {
            ASCEND_ASSERT(aclrtCreateNotify(&notify, kDefaultNotifyFlag));
        }
        if (traceStart) {
            startEvents_.resize(streamCount);
            for (auto& event : startEvents_) {
                ASCEND_ASSERT(aclrtCreateEventWithFlag(&event, ACL_EVENT_TIME_LINE));
            }
        }
    }

    void Arm(size_t index, aclrtStream stream, bool recordStart)
    {
        ASSERT(index < notifies_.size());
        ASSERT(!recordStart || index < startEvents_.size());
        ASCEND_ASSERT(aclrtSetDevice(deviceId_));
        ASCEND_ASSERT(aclrtWaitAndResetNotify(notifies_[index], stream, 0));
        if (recordStart) { ASCEND_ASSERT(aclrtRecordEvent(startEvents_[index], stream)); }
    }

    void Release()
    {
        ASCEND_ASSERT(aclrtSetDevice(deviceId_));
        for (auto notify : notifies_) {
            ASCEND_ASSERT(aclrtRecordNotify(notify, controlStream_));
        }
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
        for (auto notify : notifies_) { ASCEND_ASSERT(aclrtDestroyNotify(notify)); }
        notifies_.clear();
        if (controlStream_ != nullptr) {
            ASCEND_ASSERT(aclrtDestroyStream(controlStream_));
            controlStream_ = nullptr;
        }
    }
};

#endif  // ASCEND_STREAM_START_GATE_H
