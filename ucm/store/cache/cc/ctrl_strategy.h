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

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include "cache_domain.h"
#include "ctrl_layout.h"
#include "global_config.h"
#include "ipc/fd_socket.h"
#include "ipc/mem_fd.h"

namespace UC::CacheStore {

class CtrlStrategy {
    MemFd ctrlMem_;
    FdSocket socket_;
    int32_t ctrlFd_{-1};
    std::thread acceptThread_;
    CtrlLayout layout_;
    std::string socketName_;

public:
    ~CtrlStrategy()
    {
        socket_.Close();
        if (acceptThread_.joinable()) { acceptThread_.join(); }
    }

    Status Setup(const Config& cfg)
    {
        socketName_ = CacheDomainName(cfg.uniqueId) + "_ctrl";
        if (!cfg.shareBufferEnable) { return SetupCreator(cfg); }
        // A scheduler/control-only participant normally has no shard layout yet. Let a
        // worker create the control region, then attach to the published header.
        if (cfg.deviceId < 0 && cfg.shardSize == 0) { return SetupJoiner(cfg); }
        auto s = socket_.Listen(socketName_);
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
        if (cfg.alignSize == 0 || (cfg.alignSize & (cfg.alignSize - 1)) != 0 ||
            cfg.shardSize > std::numeric_limits<size_t>::max() - (cfg.alignSize - 1)) {
            return Status::InvalidParam("invalid cache slot alignment");
        }
        auto slotSize = AlignUp(cfg.shardSize, cfg.alignSize);
        if (slotSize == 0 || cfg.bufferCapacity < slotSize) {
            return Status::InvalidParam("ctrl creator requires valid shardSize and capacity");
        }
        const auto maxRanks = cfg.shareBufferRankStriped ? cfg.localRankSize : kMaxRanks;
        if (maxRanks == 0 || maxRanks > kMaxRanks) {
            return Status::InvalidParam("invalid cache segment count({})", maxRanks);
        }
        const auto totalConfiguredSlots = cfg.bufferCapacity / slotSize;
        const auto m =
            cfg.shareBufferRankStriped ? totalConfiguredSlots / maxRanks : totalConfiguredSlots;
        if (m == 0 || m > std::numeric_limits<size_t>::max() / maxRanks) {
            return Status::InvalidParam("cache control layout too large");
        }
        auto nBuckets = CalcBucketCount(m);
        const auto totalSlots = maxRanks * m;
        const auto prefixSize = CtrlLayout::SlotMetaOffset(nBuckets);
        if (totalSlots > (std::numeric_limits<size_t>::max() - prefixSize) / sizeof(SlotMeta)) {
            return Status::InvalidParam("cache control layout too large");
        }
        const auto metaEnd = prefixSize + sizeof(SlotMeta) * totalSlots;
        if (maxRanks > (std::numeric_limits<size_t>::max() - metaEnd) / sizeof(PrefetchRing)) {
            return Status::InvalidParam("cache control layout too large");
        }
        auto totalSize = CtrlLayout::TotalSize(nBuckets, totalSlots, maxRanks);
        auto s = ctrlMem_.Create("ucm_v2_ctrl", totalSize, true);
        if (s.Failure()) { return s; }
        ctrlFd_ = ctrlMem_.Fd();
        layout_.Bind(ctrlMem_.Addr(), maxRanks, m, nBuckets);
        layout_.InitHeader(slotSize, cfg.shareBufferRankStriped, cfg.shareBufferNumaNodes);
        layout_.SetMagic();
        if (cfg.shareBufferEnable) {
            acceptThread_ = std::thread([this] { AcceptLoop(); });
        }
        return Status::OK();
    }

    Status SetupJoiner(const Config& cfg)
    {
        constexpr auto backoff = std::chrono::milliseconds(50);
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(cfg.timeoutMs);
        for (;;) {
            auto s = socket_.Connect(socketName_);
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
        auto* header = layout_.Hdr();
        const auto m = header->nSlotsPerRank;
        const auto nBuckets = header->nBuckets;
        const auto maxRanks = header->maxRanks;
        if (m == 0 || maxRanks == 0 || maxRanks > kMaxRanks || nBuckets == 0 ||
            nBuckets > kMaxBuckets || (nBuckets & (nBuckets - 1)) != 0 ||
            m > std::numeric_limits<size_t>::max() / maxRanks ||
            header->numaNodeCount > kMaxRanks) {
            return Status::InvalidParam("ctrl header invalid");
        }
        const auto rankStriped = header->rankStriped != 0;
        if (rankStriped != cfg.shareBufferRankStriped ||
            maxRanks != (rankStriped ? cfg.localRankSize : kMaxRanks)) {
            return Status::InvalidParam("cache participants disagree on rank layout");
        }
        if (rankStriped) {
            if (header->numaNodeCount != cfg.shareBufferNumaNodes.size() ||
                !std::equal(cfg.shareBufferNumaNodes.begin(), cfg.shareBufferNumaNodes.end(),
                            header->numaNodes)) {
                return Status::InvalidParam("cache participants disagree on NUMA node layout");
            }
        }
        if (cfg.shardSize != 0) {
            if (cfg.alignSize == 0 || (cfg.alignSize & (cfg.alignSize - 1)) != 0 ||
                cfg.shardSize > std::numeric_limits<size_t>::max() - (cfg.alignSize - 1) ||
                AlignUp(cfg.shardSize, cfg.alignSize) != header->slotSize) {
                return Status::InvalidParam("cache participants disagree on slot size");
            }
        }
        const auto totalSlots = maxRanks * m;
        const auto prefixSize = CtrlLayout::SlotMetaOffset(nBuckets);
        if (totalSlots > (std::numeric_limits<size_t>::max() - prefixSize) / sizeof(SlotMeta)) {
            return Status::InvalidParam("ctrl header invalid");
        }
        const auto metaEnd = prefixSize + sizeof(SlotMeta) * totalSlots;
        if (maxRanks > (std::numeric_limits<size_t>::max() - metaEnd) / sizeof(PrefetchRing)) {
            return Status::InvalidParam("ctrl header invalid");
        }
        s = ctrlMem_.Remap(CtrlLayout::TotalSize(nBuckets, totalSlots, maxRanks));
        if (s.Failure()) { return s; }
        layout_.Bind(ctrlMem_.Addr(), maxRanks, m, nBuckets);
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
