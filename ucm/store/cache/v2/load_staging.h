/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
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
#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include "status/status.h"
#include "trans/device.h"

namespace UC::Cache2 {

/* Owned by the transfer thread. Reset only after all H2D streams have drained;
 * a source buffer must remain unchanged until its asynchronous copy completes. */
class LoadStaging {
    std::shared_ptr<void> data_;
    size_t blockBytes_{0};
    size_t blocks_{0};
    size_t used_{0};

public:
    Status Setup(int32_t deviceId, size_t blockBytes, size_t blocks)
    {
        if (blockBytes == 0 || blocks == 0 ||
            blocks > std::numeric_limits<size_t>::max() / blockBytes) {
            return Status::InvalidParam("invalid load staging geometry");
        }
        Trans::Device device;
        auto status = device.Setup(deviceId);
        if (status.Failure()) { return status; }
        auto allocator = device.MakeBuffer();
        if (!allocator) { return Status::Error("cannot create load staging allocator"); }
        data_ = allocator->MakeHostBuffer(blockBytes * blocks);
        if (!data_) { return Status::OutOfMemory(); }
        blockBytes_ = blockBytes;
        blocks_ = blocks;
        used_ = 0;
        return Status::OK();
    }

    bool Full() const { return used_ == blocks_; }
    void Reset() { used_ = 0; }
    void* Copy(const void* source)
    {
        assert(data_ && !Full());
        auto* destination = static_cast<std::byte*>(data_.get()) + used_ * blockBytes_;
        std::memcpy(destination, source, blockBytes_);
        ++used_;
        return destination;
    }
};

}  // namespace UC::Cache2
