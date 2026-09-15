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
#include <thread>
#include <vector>
#include "cache_types.h"

namespace UC::CacheStore {

/* Shared control-plane layout:
 * [Header][nBuckets bucket heads][kLockStripes striped locks][slot metadata].
 * There are no slot locks: per-slot coordination is the lock-free pin protocol on
 * SlotMeta::reference (see Buffer).
 *
 * Slot metadata is initialized lazily per rank (InitSlotRange): a rank's slots are only
 * reachable from buckets after that rank links them in, which happens strictly after its
 * own Setup, so untouched zero pages are never traversed. */
class CtrlLayout {
    void* base_{nullptr};
    size_t maxRanks_{0};
    size_t nSlotsPerRank_{0};
    size_t nBuckets_{0};
    size_t totalSlots_{0};

public:
    static size_t BucketsOffset() { return AlignUp(sizeof(Header), 64); }
    static size_t LocksOffset(size_t nBuckets)
    {
        return AlignUp(BucketsOffset() + sizeof(size_t) * nBuckets, alignof(BucketLock));
    }
    static size_t SlotMetaOffset(size_t nBuckets)
    {
        return AlignUp(LocksOffset(nBuckets) + sizeof(BucketLock) * kLockStripes,
                       alignof(SlotMeta));
    }
    static size_t TotalSize(size_t nBuckets, size_t totalSlots)
    {
        return SlotMetaOffset(nBuckets) + sizeof(SlotMeta) * totalSlots;
    }

    void Bind(void* base, size_t maxRanks, size_t nSlotsPerRank, size_t nBuckets)
    {
        base_ = base;
        maxRanks_ = maxRanks;
        nSlotsPerRank_ = nSlotsPerRank;
        nBuckets_ = nBuckets;
        totalSlots_ = maxRanks * nSlotsPerRank;
    }

    Header* Hdr() const { return static_cast<Header*>(base_); }
    size_t TotalSlots() const { return totalSlots_; }
    size_t NSlotsPerRank() const { return nSlotsPerRank_; }
    size_t NBuckets() const { return nBuckets_; }

    std::atomic<size_t>* Buckets() const
    {
        return reinterpret_cast<std::atomic<size_t>*>(static_cast<std::byte*>(base_) +
                                                      BucketsOffset());
    }
    /* Striped lock: one lock array entry per kLockStripes buckets. Distinct buckets may
     * share a stripe. Alloc reuses the held lock when both buckets share a stripe. */
    BucketLock* LockOf(size_t iBucket) const
    {
        auto* stripes =
            reinterpret_cast<BucketLock*>(static_cast<std::byte*>(base_) + LocksOffset(nBuckets_));
        return &stripes[iBucket & (kLockStripes - 1)];
    }
    SlotMeta* SlotMetaArr() const
    {
        return reinterpret_cast<SlotMeta*>(static_cast<std::byte*>(base_) +
                                           SlotMetaOffset(nBuckets_));
    }
    /* Initializes everything except slot metadata (see InitSlotRange). */
    void InitHeader(size_t slotSize, const std::vector<size_t>& numaNodes)
    {
        auto h = Hdr();
        h->magic.store(0, std::memory_order_relaxed);
        h->maxRanks = maxRanks_;
        h->nSlotsPerRank = nSlotsPerRank_;
        h->slotSize = slotSize;
        h->nBuckets = nBuckets_;
        h->numaNodeCount = numaNodes.size();
        for (size_t i = 0; i < numaNodes.size(); ++i) { h->numaNodes[i] = numaNodes[i]; }
        for (size_t i = 0; i < maxRanks_; i++) {
            h->rankDescs[i].ready.store(0, std::memory_order_relaxed);
            h->clockHands[i].store(0, std::memory_order_relaxed);
        }
        auto* buckets = Buckets();
        for (size_t i = 0; i < nBuckets_; i++) {
            buckets[i].store(kInvalidIndex, std::memory_order_relaxed);
        }
        auto* stripes =
            reinterpret_cast<BucketLock*>(static_cast<std::byte*>(base_) + LocksOffset(nBuckets_));
        for (size_t i = 0; i < kLockStripes; i++) { stripes[i].Init(); }
    }

    /* Lazily initialize one rank's slot range. Only legal while that rank's
     * rankDescs[rank].ready == 0: an initialized rank's slots may be pinned by remote
     * processes, and resetting them would underflow those pins. */
    void InitSlotRange(size_t rank)
    {
        auto* slotMeta = SlotMetaArr();
        for (size_t i = rank * nSlotsPerRank_; i < (rank + 1) * nSlotsPerRank_; i++) {
            slotMeta[i].Init();
        }
    }

    Status SetRankDesc(size_t rank, const RankDataDesc& d)
    {
        if (rank >= maxRanks_) { return Status::InvalidParam("rank out of range"); }
        Hdr()->rankDescs[rank].ready.store(d.ready.load(std::memory_order_relaxed),
                                           std::memory_order_release);
        return Status::OK();
    }

    Expected<RankDataDesc> GetRankDesc(size_t rank) const
    {
        if (rank >= maxRanks_) { return Status::InvalidParam("rank out of range"); }
        if (Hdr()->rankDescs[rank].ready.load(std::memory_order_acquire) != 1) {
            return Status::NotFound();
        }
        RankDataDesc out;
        out.ready.store(1, std::memory_order_relaxed);
        return out;
    }

    void SetMagic() { Hdr()->magic.store(kMagic, std::memory_order_release); }

    bool WaitReady(size_t timeoutMs) const
    {
        constexpr auto interval = std::chrono::milliseconds(50);
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);
        while (Hdr()->magic.load(std::memory_order_acquire) != kMagic) {
            if (timeoutMs > 0 && std::chrono::steady_clock::now() >= deadline) { return false; }
            std::this_thread::sleep_for(interval);
        }
        return true;
    }
};

}  // namespace UC::CacheStore
