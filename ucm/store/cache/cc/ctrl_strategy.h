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

#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <thread>
#include "ctrl_layout.h"
#include "global_config.h"
#include "ipc/fd_socket.h"
#include "ipc/mem_fd.h"

namespace UC::CacheStore {

inline constexpr const char* kAbstractSockName = "ucm_v2_cache_ctrl";

class CtrlStrategy {
    MemFd ctrlMem_;
    FdSocket socket_;
    int32_t ctrlFd_{-1};
    std::thread acceptThread_;
    CtrlLayout layout_;

public:
    ~CtrlStrategy()
    {
        socket_.Close();
        if (acceptThread_.joinable()) { acceptThread_.join(); }
    }

    Status Setup(const Config& cfg)
    {
        auto s = socket_.Listen(kAbstractSockName);
        if (s.Success()) {
            auto r = SetupCreator(cfg);
            if (r.Failure()) { socket_.Close(); }
            return r;
        }
        if (s != Status::DuplicateKey()) { return s; }
        return SetupJoiner(cfg);
    }

    CtrlLayout& Layout() { return layout_; }

private:
    Status SetupCreator(const Config& cfg)
    {
        auto slotSize = AlignUp(cfg.shardSize, cfg.alignSize);
        if (slotSize == 0 || cfg.bufferCapacity < slotSize) {
            return Status::InvalidParam("ctrl creator requires valid shardSize and capacity");
        }
        auto m = cfg.bufferCapacity / slotSize;
        auto nBuckets = CalcBucketCount(m);
        auto totalSize = CtrlLayout::TotalSize(nBuckets, kMaxRanks * m);
        auto s = ctrlMem_.Create("ucm_v2_ctrl", totalSize, true);
        if (s.Failure()) { return s; }
        ctrlFd_ = ctrlMem_.Fd();
        layout_.Bind(ctrlMem_.Addr(), kMaxRanks, m, nBuckets);
        layout_.InitHeader(slotSize);
        acceptThread_ = std::thread([this] { AcceptLoop(); });
        layout_.SetMagic();
        return Status::OK();
    }

    Status SetupJoiner(const Config& cfg)
    {
        constexpr auto backoff = std::chrono::milliseconds(50);
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(cfg.timeoutMs);
        for (;;) {
            auto s = socket_.Connect(kAbstractSockName);
            if (s.Success()) { break; }
            if (cfg.timeoutMs > 0 && std::chrono::steady_clock::now() >= deadline) {
                return Status::Retry();
            }
            std::this_thread::sleep_for(backoff);
        }
        int32_t fd = -1;
        auto s = socket_.RecvFd(fd);
        socket_.Close();
        if (s.Failure()) { return s; }
        s = ctrlMem_.Adopt(fd, sizeof(Header));
        if (s.Failure()) { return s; }
        layout_.Bind(ctrlMem_.Addr(), kMaxRanks, 0, 0);
        if (!layout_.WaitReady(cfg.timeoutMs)) { return Status::Retry(); }
        auto m = layout_.Hdr()->nSlotsPerRank;
        auto nBuckets = layout_.Hdr()->nBuckets;
        if (m == 0 || nBuckets == 0 || (nBuckets & (nBuckets - 1)) != 0) {
            return Status::InvalidParam("ctrl header invalid");
        }
        s = ctrlMem_.Remap(CtrlLayout::TotalSize(nBuckets, kMaxRanks * m));
        if (s.Failure()) { return s; }
        layout_.Bind(ctrlMem_.Addr(), kMaxRanks, m, nBuckets);
        return Status::OK();
    }

    void AcceptLoop()
    {
        for (;;) {
            if (socket_.AcceptAndSend(ctrlFd_).Failure()) { break; }
        }
        socket_.Close();
    }
};

inline std::unique_ptr<CtrlStrategy> MakeCtrlStrategy() { return std::make_unique<CtrlStrategy>(); }

}  // namespace UC::CacheStore
