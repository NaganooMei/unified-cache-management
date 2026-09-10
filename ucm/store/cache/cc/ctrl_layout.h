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
#include "cache_types.h"

namespace UC::CacheStore {

/* Shared control-plane layout:
 * [Header][nBuckets bucket heads][kLockStripes striped locks][slot metadata]
 * [kMaxRanks prefetch rings]. There are no slot locks: per-slot coordination is the
 * lock-free pin protocol on SlotMeta::reference (see Buffer).
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
        return SlotMetaOffset(nBuckets) + sizeof(SlotMeta) * totalSlots +
               sizeof(PrefetchRing) * kMaxRanks;
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
     * share a stripe (prob. 1/kLockStripes), which only adds serialization; the Alloc
     * TryLock backoff handles the same-stripe case naturally. */
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
    PrefetchRing* RingOf(size_t rank) const
    {
        auto* rings = reinterpret_cast<PrefetchRing*>(static_cast<std::byte*>(base_) +
                                                      SlotMetaOffset(nBuckets_) +
                                                      sizeof(SlotMeta) * totalSlots_);
        return rings + rank;
    }

    /* Prefetch ring ops. Contract: exactly one producer thread per domain (the
     * scheduler's Prefetch caller) and one consumer thread per rank (the worker's
     * prefetch executor); neither end of a ring may be called concurrently. Overflow
     * keeps the front of the batch, drops the remainder and counts it. */
    void RingPush(size_t rank, const Detail::BlockId* blocks, size_t num)
    {
        if (rank >= maxRanks_ || num == 0) { return; }
        auto* ring = RingOf(rank);
        auto h = ring->head.load(std::memory_order_relaxed);
        auto t = ring->tail.load(std::memory_order_acquire);
        auto free = kPrefetchDepth - static_cast<size_t>(h - t);
        auto n = num < free ? num : free;
        for (size_t i = 0; i < n; i++) { ring->entries[(h + i) % kPrefetchDepth] = blocks[i]; }
        ring->head.store(h + n, std::memory_order_release);
        if (n < num) { ring->dropped.fetch_add(num - n, std::memory_order_relaxed); }
    }

    size_t RingDrain(size_t rank, Detail::BlockId* out, size_t max)
    {
        if (rank >= maxRanks_ || max == 0) { return 0; }
        auto* ring = RingOf(rank);
        auto t = ring->tail.load(std::memory_order_relaxed);
        auto h = ring->head.load(std::memory_order_acquire);
        auto avail = static_cast<size_t>(h - t);
        auto n = max < avail ? max : avail;
        for (size_t i = 0; i < n; i++) { out[i] = ring->entries[(t + i) % kPrefetchDepth]; }
        ring->tail.store(t + n, std::memory_order_release);
        return n;
    }

    uint64_t RingDropped(size_t rank) const
    {
        if (rank >= maxRanks_) { return 0; }
        return RingOf(rank)->dropped.load(std::memory_order_relaxed);
    }

    /* Initializes everything except slot metadata (see InitSlotRange). */
    void InitHeader(size_t slotSize)
    {
        auto h = Hdr();
        h->magic.store(0, std::memory_order_relaxed);
        h->maxRanks = maxRanks_;
        h->nSlotsPerRank = nSlotsPerRank_;
        h->slotSize = slotSize;
        h->nBuckets = nBuckets_;
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
        /* Ring counters only; entries stay untouched until first use. */
        for (size_t r = 0; r < maxRanks_; r++) {
            auto* ring = RingOf(r);
            ring->head.store(0, std::memory_order_relaxed);
            ring->tail.store(0, std::memory_order_relaxed);
            ring->dropped.store(0, std::memory_order_relaxed);
        }
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
