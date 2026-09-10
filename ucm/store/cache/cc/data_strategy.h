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
 */
#pragma once

#include <cstddef>
#include <cstring>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include "cache_types.h"
#include "global_config.h"
#include "ipc/fd_socket.h"
#include "ipc/mem_fd.h"

namespace UC::CacheStore {

class DataStrategy {
    MemFd data_;
    FdSocket dataSock_;
    std::thread acceptThread_;
    size_t slotSize_{0};

public:
    ~DataStrategy()
    {
        dataSock_.Close();
        if (acceptThread_.joinable()) { acceptThread_.join(); }
    }

    Status Setup(int32_t rank, size_t slotSize, size_t size)
    {
        slotSize_ = slotSize;
        std::string name = "ucm_v2_data_" + std::to_string(rank);
        auto s = data_.Create(name, size, true);
        if (s.Failure()) { return s; }
        constexpr size_t kFirstTouchChunk = 256 * 1024 * 1024;
        auto* p = static_cast<std::byte*>(data_.Addr());
        for (size_t off = 0; off < size;) {
            size_t n = (size - off > kFirstTouchChunk) ? kFirstTouchChunk : (size - off);
            std::memset(p + off, 0, n);
            off += n;
        }
        s = dataSock_.Listen(name);
        if (s.Failure()) { return s; }
        acceptThread_ = std::thread([this] {
            for (;;) {
                if (dataSock_.AcceptAndSend(data_.Fd()).Failure()) { break; }
            }
        });
        return Status::OK();
    }

    void* LocalDataAddr(size_t localIdx)
    {
        return static_cast<std::byte*>(data_.Addr()) + localIdx * slotSize_;
    }
};

}  // namespace UC::CacheStore
