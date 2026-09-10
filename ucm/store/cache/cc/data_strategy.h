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
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include "cache_types.h"
#include "ctrl_layout.h"
#include "global_config.h"
#include "ipc/fd_socket.h"
#include "ipc/mem_fd.h"
#include "trans/buffer.h"

namespace UC::CacheStore {

class DataStrategy {
    struct RemoteEntry {
        std::atomic<void*> addr{nullptr};
        std::atomic<void*> deviceAddr{nullptr};
        int32_t fd{-1};
        size_t size{0};
    };
    MemFd data_;
    FdSocket dataSock_;
    std::thread acceptThread_;
    size_t slotSize_{0};
    void* devicePtr_{nullptr};
    /* Non-owning view of the shared control plane: readiness of a remote rank is checked
     * before mapping its data. Owned by Buffer; outlives this object. */
    CtrlLayout* ctrl_{nullptr};
    size_t myRank_{kInvalidIndex};
    size_t nSlotsPerRank_{0};
    /* Guards lazy init / teardown of remoteCache_ entries; remoteCache_ is process-local. */
    std::mutex remoteMtx_;
    RemoteEntry remoteCache_[kMaxRanks];

public:
    ~DataStrategy()
    {
        dataSock_.Close();
        if (acceptThread_.joinable()) { acceptThread_.join(); }
        /* Remote mappings: release the device registration before the host range is
         * unmapped. */
        std::lock_guard<std::mutex> guard(remoteMtx_);
        for (size_t r = 0; r < kMaxRanks; r++) {
            void* a = remoteCache_[r].addr.load(std::memory_order_acquire);
            if (a != nullptr) {
                if (remoteCache_[r].deviceAddr.load(std::memory_order_acquire) != nullptr) {
                    Trans::Buffer::UnregisterHostBuffer(a);
                }
                ::munmap(a, remoteCache_[r].size);
                ::close(remoteCache_[r].fd);
            }
        }
        /* Local buffer: release the device registration before data_ unmaps the host
         * range. */
        if (devicePtr_ != nullptr) { Trans::Buffer::UnregisterHostBuffer(data_.Addr()); }
    }

    Status Setup(CtrlLayout& ctrl, int32_t deviceId, size_t rank, size_t slotSize,
                 size_t nSlotsPerRank)
    {
        ctrl_ = &ctrl;
        myRank_ = rank;
        nSlotsPerRank_ = nSlotsPerRank;
        slotSize_ = slotSize;
        auto size = nSlotsPerRank * slotSize;
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
        s = Trans::Buffer::RegisterHostBuffer(data_.Addr(), size, &devicePtr_);
        if (s.Failure()) { return s; }
        s = dataSock_.Listen(name);
        if (s.Failure()) { return s; }
        acceptThread_ = std::thread([this] {
            for (;;) {
                if (dataSock_.AcceptAndSend(data_.Fd()).Failure()) { break; }
            }
        });
        return Status::OK();
    }

    void* DataAt(size_t slotIdx)
    {
        if (slotIdx == kInvalidIndex || nSlotsPerRank_ == 0) { return nullptr; }
        auto rank = slotIdx / nSlotsPerRank_;
        auto localIdx = slotIdx % nSlotsPerRank_;
        if (rank == myRank_) { return LocalDataAddr(localIdx); }
        if (rank >= kMaxRanks) { return nullptr; }
        auto& entry = remoteCache_[rank];
        void* cur = entry.addr.load(std::memory_order_acquire);
        if (cur != nullptr) { return static_cast<std::byte*>(cur) + localIdx * slotSize_; }
        return MapRemoteData(rank, localIdx);
    }

    void* DeviceDataAt(size_t slotIdx)
    {
        if (slotIdx == kInvalidIndex || nSlotsPerRank_ == 0) { return nullptr; }
        auto rank = slotIdx / nSlotsPerRank_;
        auto localIdx = slotIdx % nSlotsPerRank_;
        if (rank == myRank_) { return LocalDeviceDataAddr(localIdx); }
        if (rank >= kMaxRanks) { return nullptr; }
        auto& entry = remoteCache_[rank];
        void* cur = entry.deviceAddr.load(std::memory_order_acquire);
        if (cur != nullptr) { return static_cast<std::byte*>(cur) + localIdx * slotSize_; }
        /* First touch maps the rank and registers it with the device in one go (see
         * MapRemoteData). A non-null host mapping with a null deviceAddr means
         * registration failed once and is not retried. */
        if (MapRemoteData(rank, localIdx) == nullptr) { return nullptr; }
        cur = entry.deviceAddr.load(std::memory_order_acquire);
        return cur != nullptr ? static_cast<std::byte*>(cur) + localIdx * slotSize_ : nullptr;
    }

    void* LocalDataAddr(size_t localIdx)
    {
        return static_cast<std::byte*>(data_.Addr()) + localIdx * slotSize_;
    }

    void* LocalDeviceDataAddr(size_t localIdx)
    {
        return static_cast<std::byte*>(devicePtr_) + localIdx * slotSize_;
    }

private:
    void* MapRemoteData(size_t rank, size_t localIdx)
    {
        std::lock_guard<std::mutex> guard(remoteMtx_);
        auto& entry = remoteCache_[rank];
        void* cur = entry.addr.load(std::memory_order_acquire);
        if (cur != nullptr) { return static_cast<std::byte*>(cur) + localIdx * slotSize_; }
        auto desc = ctrl_->GetRankDesc(rank);
        if (!desc || desc.Value().ready.load(std::memory_order_relaxed) != 1) { return nullptr; }
        std::string name = "ucm_v2_data_" + std::to_string(rank);
        FdSocket s;
        if (s.Connect(name).Failure()) { return nullptr; }
        int32_t fd = -1;
        if (s.RecvFd(fd).Failure()) {
            s.Close();
            return nullptr;
        }
        s.Close();
        size_t size = nSlotsPerRank_ * slotSize_;
        void* a = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (a == MAP_FAILED) {
            ::close(fd);
            return nullptr;
        }
        /* Register the mapping with the local device for SDMA-direct access. On failure
         * the host mapping stays usable and deviceAddr stays null (not retried). */
        void* dev = nullptr;
        if (Trans::Buffer::RegisterHostBuffer(a, size, &dev).Success()) {
            entry.deviceAddr.store(dev, std::memory_order_release);
        }
        /* fd/size/deviceAddr are written before addr is published so any thread that
         * observes the mapping through addr (acquire) also observes consistent teardown
         * fields. */
        entry.fd = fd;
        entry.size = size;
        entry.addr.store(a, std::memory_order_release);
        return static_cast<std::byte*>(a) + localIdx * slotSize_;
    }
};

}  // namespace UC::CacheStore
