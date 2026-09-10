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

#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <sys/mman.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include "cache_types.h"
#include "ctrl_strategy.h"
#include "data_strategy.h"
#include "global_config.h"

namespace UC::CacheStore {

class Buffer {
public:
    class Handle {
        friend class Buffer;
        Buffer* buf_{nullptr};
        size_t slotIdx_{kInvalidIndex};
        bool owner_{false};

        Handle(Buffer* buf, size_t slotIdx, bool owner)
            : buf_(buf), slotIdx_(slotIdx), owner_(owner)
        {
        }

    public:
        Handle() = default;
        Handle(const Handle&) = delete;
        Handle& operator=(const Handle&) = delete;
        Handle(Handle&& o) noexcept : buf_(o.buf_), slotIdx_(o.slotIdx_), owner_(o.owner_)
        {
            o.buf_ = nullptr;
            o.slotIdx_ = kInvalidIndex;
            o.owner_ = false;
        }
        Handle& operator=(Handle&& o) noexcept
        {
            Handle tmp(std::move(o));
            Swap(tmp);
            return *this;
        }
        ~Handle()
        {
            if (Valid()) { buf_->Release(slotIdx_); }
        }
        explicit operator bool() const { return Valid(); }
        bool Owner() const { return owner_; }
        void* Data() { return Valid() ? buf_->DataAt(slotIdx_) : nullptr; }
        bool Ready() const { return Valid() && buf_->Ready(slotIdx_); }
        State GetState() const { return Valid() ? buf_->GetState(slotIdx_) : State::Failed; }
        void MarkReady()
        {
            if (Valid()) { buf_->MarkReady(slotIdx_); }
        }
        void MarkFailed()
        {
            if (Valid()) { buf_->MarkFailed(slotIdx_); }
        }

    private:
        bool Valid() const { return buf_ != nullptr && slotIdx_ != kInvalidIndex; }
        void Swap(Handle& o) noexcept
        {
            std::swap(buf_, o.buf_);
            std::swap(slotIdx_, o.slotIdx_);
            std::swap(owner_, o.owner_);
        }
    };

private:
    struct RemoteEntry {
        std::atomic<void*> addr{nullptr};
        int32_t fd{-1};
        size_t size{0};
    };
    std::unique_ptr<CtrlStrategy> ctrl_;
    std::unique_ptr<DataStrategy> data_;
    /* Guards lazy init / teardown of remoteCache_ entries; remoteCache_ is process-local. */
    std::mutex remoteMtx_;
    size_t myRank_{kInvalidIndex};
    size_t nSlotsPerRank_{0};
    size_t nBuckets_{0};
    size_t slotSize_{0};
    size_t reserved_{0};
    RemoteEntry remoteCache_[kMaxRanks];

    /* Optimistic pin attempts before falling back to the bucket-lock path. */
    static constexpr size_t kPinSpinFast = 64;
    /* With the bucket lock held a claim on a chained slot is always transient (the claimer
     * must TryLock this bucket, fail and roll back), so the slow path may wait it out. */
    static constexpr size_t kPinSpinSlow = std::numeric_limits<size_t>::max();

public:
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    ~Buffer()
    {
        std::lock_guard<std::mutex> guard(remoteMtx_);
        for (size_t r = 0; r < kMaxRanks; r++) {
            void* a = remoteCache_[r].addr.load(std::memory_order_acquire);
            if (a != nullptr) {
                ::munmap(a, remoteCache_[r].size);
                ::close(remoteCache_[r].fd);
            }
        }
    }

    Status Setup(const Config& cfg)
    {
        reserved_ = cfg.loadExclusiveBufferNumber;
        ctrl_ = MakeCtrlStrategy();
        if (auto s = ctrl_->Setup(cfg); s.Failure()) { return s; }
        slotSize_ = ctrl_->Layout().Hdr()->slotSize;
        nSlotsPerRank_ = ctrl_->Layout().Hdr()->nSlotsPerRank;
        nBuckets_ = ctrl_->Layout().Hdr()->nBuckets;
        if (nSlotsPerRank_ == 0 || nBuckets_ == 0) {
            return Status::InvalidParam("ctrl header has zero slots per rank or buckets");
        }
        if (cfg.deviceId >= 0) {
            if (cfg.physicalDeviceId < 0 ||
                cfg.physicalDeviceId >= static_cast<int32_t>(kMaxRanks)) {
                return Status::InvalidParam("physicalDeviceId({}) must be in [0, {})",
                                            cfg.physicalDeviceId, kMaxRanks);
            }
            /* reserved_ comes from the local config while nSlotsPerRank_ comes from the
             * creator's header; reject mismatched configurations instead of underflowing
             * in FetchNode. Only allocating ranks need this invariant. */
            if (reserved_ >= nSlotsPerRank_) {
                return Status::InvalidParam(
                    "loadExclusiveBufferNumber({}) must be less than slots per "
                    "rank({})",
                    reserved_, nSlotsPerRank_);
            }
            myRank_ = static_cast<size_t>(cfg.physicalDeviceId);
            /* Lazy per-rank slot initialization. The ready flag is the rejoin guard: once
             * set, this rank's slots may be pinned by remote processes and must not be
             * reset; while it is clear, no slot of this rank can be reachable from any
             * bucket (linking requires a completed Setup), so (re-)initialization is
             * safe. */
            if (ctrl_->Layout().Hdr()->rankDescs[myRank_].ready.load(std::memory_order_acquire) ==
                0) {
                ctrl_->Layout().InitSlotRange(myRank_);
            }
            data_ = std::make_unique<DataStrategy>();
            if (auto s = data_->Setup(cfg.physicalDeviceId, slotSize_, nSlotsPerRank_ * slotSize_);
                s.Failure()) {
                return s;
            }
            RankDataDesc desc;
            desc.ready.store(1, std::memory_order_relaxed);
            if (auto s = ctrl_->Layout().SetRankDesc(cfg.physicalDeviceId, desc); s.Failure()) {
                return s;
            }
        }
        /* else: control-plane-only participant; myRank_ stays kInvalidIndex and the
         * allocation APIs below are disabled for it. */
        return Status::OK();
    }

    /* Bucket count of the shared hash table (diagnostics / tests). */
    size_t NumBuckets() const { return nBuckets_; }
    /* This rank's index; kInvalidIndex for control-plane-only participants. */
    size_t MyRank() const { return myRank_; }
    size_t SlotSize() const { return slotSize_; }
    /* Online = the rank completed Setup (sticky: ranks do not leave in this deployment). */
    bool RankReady(size_t rank) const
    {
        if (rank >= kMaxRanks) { return false; }
        return ctrl_->Layout().Hdr()->rankDescs[rank].ready.load(std::memory_order_acquire) == 1;
    }

    /* Prefetch command rings. Contract: exactly one producer thread per domain (the
     * scheduler's Prefetch caller) and one consumer thread per rank (the worker's
     * prefetch executor); neither end of a ring may be called concurrently. Overflow
     * keeps the front of the batch, drops the remainder and counts it. */
    void EnqueuePrefetch(size_t rank, const Detail::BlockId* blocks, size_t num)
    {
        if (rank >= kMaxRanks || num == 0) { return; }
        auto* ring = ctrl_->Layout().RingOf(rank);
        auto h = ring->head.load(std::memory_order_relaxed);
        auto t = ring->tail.load(std::memory_order_acquire);
        auto free = kPrefetchDepth - static_cast<size_t>(h - t);
        auto n = num < free ? num : free;
        for (size_t i = 0; i < n; i++) { ring->entries[(h + i) % kPrefetchDepth] = blocks[i]; }
        ring->head.store(h + n, std::memory_order_release);
        if (n < num) { ring->dropped.fetch_add(num - n, std::memory_order_relaxed); }
    }

    size_t DrainPrefetch(size_t rank, Detail::BlockId* out, size_t max)
    {
        if (rank >= kMaxRanks || max == 0) { return 0; }
        auto* ring = ctrl_->Layout().RingOf(rank);
        auto t = ring->tail.load(std::memory_order_relaxed);
        auto h = ring->head.load(std::memory_order_acquire);
        auto avail = static_cast<size_t>(h - t);
        auto n = max < avail ? max : avail;
        for (size_t i = 0; i < n; i++) { out[i] = ring->entries[(t + i) % kPrefetchDepth]; }
        ring->tail.store(t + n, std::memory_order_release);
        return n;
    }

    uint64_t PrefetchDropped(size_t rank) const
    {
        if (rank >= kMaxRanks) { return 0; }
        return ctrl_->Layout().RingOf(rank)->dropped.load(std::memory_order_relaxed);
    }

    Handle Get(const Detail::BlockId& blockId, size_t offset, bool allowReserved = false)
    {
        if (myRank_ == kInvalidIndex) { return Handle{}; }
        auto iBucket = HashKey(blockId, nBuckets_);
        auto& layout = ctrl_->Layout();
        auto iNode = LookupOptimistic(layout, iBucket, blockId, offset);
        if (iNode != kInvalidIndex) {
            bool owner = false;
            if (PinHit(layout, iNode, iBucket, blockId, offset, kPinSpinFast, owner)) {
                return Handle{this, iNode, owner};
            }
        }
        layout.LockOf(iBucket)->Lock();
        iNode = Lookup(layout, iBucket, blockId, offset);
        if (iNode != kInvalidIndex) {
            bool owner = false;
            if (PinHit(layout, iNode, iBucket, blockId, offset, kPinSpinSlow, owner)) {
                layout.LockOf(iBucket)->Unlock();
                return Handle{this, iNode, owner};
            }
        }
        iNode = Alloc(layout, blockId, offset, iBucket, allowReserved);
        layout.LockOf(iBucket)->Unlock();
        return Handle(this, iNode, true);
    }

    void Prealloc(const Detail::BlockId& blockId, size_t offset, bool allowReserved = false)
    {
        if (myRank_ == kInvalidIndex) { return; }
        auto iBucket = HashKey(blockId, nBuckets_);
        auto& layout = ctrl_->Layout();
        layout.LockOf(iBucket)->Lock();
        auto iNode = Lookup(layout, iBucket, blockId, offset);
        if (iNode != kInvalidIndex) {
            layout.SlotMetaArr()[iNode].accessed.store(1, std::memory_order_relaxed);
        } else {
            auto n = Alloc(layout, blockId, offset, iBucket, allowReserved);
            Release(n);
        }
        layout.LockOf(iBucket)->Unlock();
    }

    bool Exist(const Detail::BlockId& blockId, size_t offset)
    {
        auto iBucket = HashKey(blockId, nBuckets_);
        auto& layout = ctrl_->Layout();
        auto iNode = LookupOptimistic(layout, iBucket, blockId, offset);
        if (iNode != kInvalidIndex) {
            /* Pin + re-validate so a hit is never reported for a slot that is being
             * reconfigured right now. */
            bool owner = false;
            if (PinHit(layout, iNode, iBucket, blockId, offset, kPinSpinFast, owner)) {
                Release(iNode);
                return true;
            }
        }
        layout.LockOf(iBucket)->Lock();
        iNode = Lookup(layout, iBucket, blockId, offset);
        bool found = (iNode != kInvalidIndex);
        if (found) { layout.SlotMetaArr()[iNode].accessed.store(1, std::memory_order_relaxed); }
        layout.LockOf(iBucket)->Unlock();
        return found;
    }

    /* Best-effort: mark every cached slot of the given blocks as recently used. Walks
     * chains without bucket locks; the hash filter keeps mid-reconfiguration nodes out.
     * Intentionally matches the whole block (all offsets), unlike Get. */
    void Touch(const Detail::BlockId* blocks, size_t num)
    {
        auto& layout = ctrl_->Layout();
        for (size_t i = 0; i < num; i++) {
            auto iBucket = HashKey(blocks[i], nBuckets_);
            auto iNode = layout.Buckets()[iBucket].load(std::memory_order_acquire);
            while (iNode != kInvalidIndex) {
                auto* meta = &layout.SlotMetaArr()[iNode];
                if (meta->hash.load(std::memory_order_acquire) == iBucket) {
                    /* Block-id words only: Touch intentionally matches every offset. */
                    size_t w[2];
                    BlockKeyWords(blocks[i], w);
                    if (meta->key[0].load(std::memory_order_acquire) == w[0] &&
                        meta->key[1].load(std::memory_order_acquire) == w[1]) {
                        meta->accessed.store(1, std::memory_order_relaxed);
                    }
                }
                iNode = meta->next.load(std::memory_order_acquire);
            }
        }
    }

    void* DataAt(size_t slotIdx)
    {
        if (!data_ || slotIdx == kInvalidIndex || nSlotsPerRank_ == 0) { return nullptr; }
        auto rank = slotIdx / nSlotsPerRank_;
        auto localIdx = slotIdx % nSlotsPerRank_;
        if (rank == myRank_) { return data_->LocalDataAddr(localIdx); }
        if (rank >= kMaxRanks) { return nullptr; }
        auto& entry = remoteCache_[rank];
        void* cur = entry.addr.load(std::memory_order_acquire);
        if (cur != nullptr) { return static_cast<std::byte*>(cur) + localIdx * slotSize_; }
        return MapRemoteData(rank, localIdx);
    }

private:
    void* MapRemoteData(size_t rank, size_t localIdx)
    {
        std::lock_guard<std::mutex> guard(remoteMtx_);
        auto& entry = remoteCache_[rank];
        void* cur = entry.addr.load(std::memory_order_acquire);
        if (cur != nullptr) { return static_cast<std::byte*>(cur) + localIdx * slotSize_; }
        auto desc = ctrl_->Layout().GetRankDesc(rank);
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
        /* fd/size are written before addr is published so any thread that observes the
         * mapping through addr (acquire) also observes consistent teardown fields. */
        entry.fd = fd;
        entry.size = size;
        entry.addr.store(a, std::memory_order_release);
        return static_cast<std::byte*>(a) + localIdx * slotSize_;
    }

    /* Lock-free reader pin: filter by key, CAS the pin, re-validate. Returns false when
     * the key does not match or the slot stays claimed beyond spinBudget (optimistic
     * callers then retry under the bucket lock). On success the caller holds a pin: the
     * slot cannot be reconfigured until Release. owner reports whether the caller must
     * load the block (first pin on a slot that is not Ready yet). */
    bool PinHit(CtrlLayout& layout, size_t iNode, size_t iBucket, const Detail::BlockId& blockId,
                size_t offset, size_t spinBudget, bool& owner)
    {
        auto* meta = &layout.SlotMetaArr()[iNode];
        for (size_t spin = 0; spin < spinBudget;) {
            if (!MatchKey(*meta, iBucket, blockId, offset)) { return false; }
            auto r = meta->reference.load(std::memory_order_acquire);
            if (r == kSlotClaimed) {
                std::this_thread::yield();
                ++spin;
                continue;
            }
            if (!meta->reference.compare_exchange_weak(r, r + 1, std::memory_order_acq_rel)) {
                std::this_thread::yield();
                ++spin;
                continue;
            }
            if (!MatchKey(*meta, iBucket, blockId, offset)) {
                /* The slot was reconfigured between the filter read and the pin. */
                meta->reference.fetch_sub(1, std::memory_order_release);
                return false;
            }
            auto st = meta->state.load(std::memory_order_acquire);
            owner = (r == 0 && st != State::Ready);
            if (owner && st == State::Failed) {
                meta->state.store(State::Loading, std::memory_order_release);
            }
            meta->accessed.store(1, std::memory_order_relaxed);
            return true;
        }
        return false;
    }

    template <std::memory_order Mo>
    size_t LookupT(CtrlLayout& layout, size_t iBucket, const Detail::BlockId& blockId,
                   size_t offset)
    {
        auto iNode = layout.Buckets()[iBucket].load(Mo);
        while (iNode != kInvalidIndex) {
            auto* meta = &layout.SlotMetaArr()[iNode];
            /* Hash filter + word comparison: nodes being reconfigured (or stale after a
             * crash) are unlinked and never match against this bucket. */
            if (MatchKey(*meta, iBucket, blockId, offset)) { return iNode; }
            iNode = meta->next.load(Mo);
        }
        return kInvalidIndex;
    }

    size_t Lookup(CtrlLayout& layout, size_t iBucket, const Detail::BlockId& blockId, size_t offset)
    {
        return LookupT<std::memory_order_relaxed>(layout, iBucket, blockId, offset);
    }

    size_t LookupOptimistic(CtrlLayout& layout, size_t iBucket, const Detail::BlockId& blockId,
                            size_t offset)
    {
        return LookupT<std::memory_order_acquire>(layout, iBucket, blockId, offset);
    }

    /* Reconfigure a victim slot for (blockId, offset). Called with the target bucket lock
     * held. Protocol: claim the slot exclusively via CAS(0 -> kSlotClaimed), unlink it
     * from its old bucket (TryLock, roll the claim back on failure), rewrite the key while
     * the slot is unreachable, then link it into the target bucket and publish with
     * reference.store(1, release).
     *
     * Design assumption: the number of slots per rank is far larger than the number of
     * concurrently pinned slots, so this loop is not expected to starve. If it ever does
     * (every slot pinned), the caller keeps spinning while holding the bucket lock. */
    size_t Alloc(CtrlLayout& layout, const Detail::BlockId& blockId, size_t offset, size_t iBucket,
                 bool allowReserved)
    {
        for (;;) {
            auto iNode = FetchNode(layout, allowReserved);
            auto* meta = &layout.SlotMetaArr()[iNode];
            size_t r = 0;
            if (!meta->reference.compare_exchange_strong(r, kSlotClaimed,
                                                         std::memory_order_acq_rel)) {
                /* In use: restore the CLOCK bit the scan cleared and try another slot. */
                meta->accessed.store(1, std::memory_order_relaxed);
                continue;
            }
            auto oldBucket = meta->hash.load(std::memory_order_relaxed);
            if (oldBucket != iBucket) {
                if (oldBucket != kInvalidIndex) {
                    /* Same-stripe TryLock (prob. 1/kLockStripes when B > L) fails against
                     * our own held stripe; the claim is rolled back and the next clock
                     * victim is tried, so this backoff loop always makes progress. */
                    if (!layout.LockOf(oldBucket)->TryLock()) {
                        meta->reference.store(0, std::memory_order_release);
                        continue;
                    }
                    Remove(layout, oldBucket, iNode);
                    layout.LockOf(oldBucket)->Unlock();
                }
                StoreKey(*meta, blockId, offset);
                meta->state.store(State::Loading, std::memory_order_relaxed);
                MoveTo(layout, iBucket, iNode);
            } else {
                /* Reusing a slot that already lives in the target bucket for another key. */
                StoreKey(*meta, blockId, offset);
                meta->state.store(State::Loading, std::memory_order_relaxed);
            }
            meta->accessed.store(1, std::memory_order_relaxed);
            /* Publish the new key to acquiring readers. */
            meta->reference.store(1, std::memory_order_release);
            return iNode;
        }
    }

    size_t FetchNode(CtrlLayout& layout, bool allowReserved)
    {
        auto total = nSlotsPerRank_ - (allowReserved ? 0 : reserved_);
        for (size_t i = 0; i < 2 * total; ++i) {
            auto cur =
                layout.Hdr()->clockHands[myRank_].fetch_add(1, std::memory_order_relaxed) % total +
                myRank_ * nSlotsPerRank_;
            uint8_t expected = 1;
            if (layout.SlotMetaArr()[cur].accessed.compare_exchange_strong(
                    expected, 0, std::memory_order_relaxed, std::memory_order_relaxed)) {
                continue;
            }
            return cur;
        }
        return layout.Hdr()->clockHands[myRank_].fetch_add(1, std::memory_order_relaxed) % total +
               myRank_ * nSlotsPerRank_;
    }

    /* Link iNode at the head of bucket iBucket. The slot must be claimed and the bucket
     * lock must be held. hash is stored with release as the publish point of the key and
     * must precede the head store. */
    void MoveTo(CtrlLayout& layout, size_t iBucket, size_t iNode)
    {
        auto* meta = &layout.SlotMetaArr()[iNode];
        auto& head = layout.Buckets()[iBucket];
        auto n = head.load(std::memory_order_relaxed);
        meta->next.store(n, std::memory_order_relaxed);
        meta->prev.store(kInvalidIndex, std::memory_order_relaxed);
        if (n != kInvalidIndex) {
            layout.SlotMetaArr()[n].prev.store(iNode, std::memory_order_relaxed);
        }
        meta->hash.store(iBucket, std::memory_order_release);
        head.store(iNode, std::memory_order_release);
    }

    /* Unlink iNode from bucket iBucket. The slot must be claimed and the bucket lock must
     * be held; afterwards the slot is unreachable and its key may be rewritten. */
    void Remove(CtrlLayout& layout, size_t iBucket, size_t iNode)
    {
        auto* meta = &layout.SlotMetaArr()[iNode];
        auto p = meta->prev.load(std::memory_order_relaxed);
        auto n = meta->next.load(std::memory_order_relaxed);
        if (p != kInvalidIndex) {
            layout.SlotMetaArr()[p].next.store(n, std::memory_order_relaxed);
        }
        if (n != kInvalidIndex) {
            layout.SlotMetaArr()[n].prev.store(p, std::memory_order_relaxed);
        }
        if (layout.Buckets()[iBucket].load(std::memory_order_relaxed) == iNode) {
            layout.Buckets()[iBucket].store(n, std::memory_order_release);
        }
        meta->prev.store(kInvalidIndex, std::memory_order_relaxed);
        meta->next.store(kInvalidIndex, std::memory_order_relaxed);
        meta->hash.store(kInvalidIndex, std::memory_order_relaxed);
    }

    void Release(size_t slotIdx)
    {
        if (slotIdx == kInvalidIndex) { return; }
        ctrl_->Layout().SlotMetaArr()[slotIdx].reference.fetch_sub(1, std::memory_order_release);
    }

    bool Ready(size_t slotIdx) { return GetState(slotIdx) == State::Ready; }

    State GetState(size_t slotIdx)
    {
        return ctrl_->Layout().SlotMetaArr()[slotIdx].state.load(std::memory_order_acquire);
    }

    void MarkReady(size_t slotIdx)
    {
        ctrl_->Layout().SlotMetaArr()[slotIdx].state.store(State::Ready, std::memory_order_release);
    }

    void MarkFailed(size_t slotIdx)
    {
        ctrl_->Layout().SlotMetaArr()[slotIdx].state.store(State::Failed,
                                                           std::memory_order_release);
    }
};

}  // namespace UC::CacheStore
