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
#include <chrono>
#include <cstddef>
#include <limits>
#include <memory>
#include <thread>
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
            if (Valid()) {
                if (owner_ && GetState() == State::Loading) { MarkFailed(); }
                buf_->Release(slotIdx_);
            }
        }
        explicit operator bool() const { return Valid(); }
        bool Owner() const { return owner_; }
        size_t GlobalSlot() const { return Valid() ? slotIdx_ : kInvalidIndex; }
        size_t Segment() const { return Valid() ? buf_->SegmentAt(slotIdx_) : kInvalidIndex; }
        size_t ReferenceCount() const { return Valid() ? buf_->ReferenceCount(slotIdx_) : 0; }
        void* Data() { return Valid() ? buf_->DataAt(slotIdx_) : nullptr; }
        void* DeviceData() { return Valid() ? buf_->DeviceDataAt(slotIdx_) : nullptr; }
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
    std::unique_ptr<CtrlStrategy> ctrl_;
    std::unique_ptr<DataStrategy> data_;
    size_t myRank_{kInvalidIndex};
    size_t nSlotsPerRank_{0};
    size_t nBuckets_{0};
    size_t slotSize_{0};
    size_t reserved_{0};
    size_t timeoutMs_{30000};
    size_t maxRanks_{0};
    bool shared_{false};
    bool ownsRankData_{false};

    /* Optimistic pin attempts before falling back to the bucket-lock path. */
    static constexpr size_t kPinSpinFast = 64;

public:
    Buffer() = default;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    ~Buffer()
    {
        if (data_ && ownsRankData_ && !shared_ && myRank_ != kInvalidIndex) {
            ctrl_->Layout().Hdr()->rankDescs[myRank_].ready.store(3, std::memory_order_release);
        }
    }

    Status Setup(const Config& cfg)
    {
        timeoutMs_ = cfg.timeoutMs;
        ctrl_ = MakeCtrlStrategy();
        if (auto s = ctrl_->Setup(cfg); s.Failure()) { return s; }
        auto* header = ctrl_->Layout().Hdr();
        slotSize_ = header->slotSize;
        nSlotsPerRank_ = header->nSlotsPerRank;
        nBuckets_ = header->nBuckets;
        maxRanks_ = header->maxRanks;
        shared_ = cfg.shareBufferEnable;
        if (shared_ && cfg.loadExclusiveBufferNumber % maxRanks_ != 0) {
            return Status::InvalidParam(
                "loadExclusiveBufferNumber({}) must be divisible by segment count({})",
                cfg.loadExclusiveBufferNumber, maxRanks_);
        }
        reserved_ =
            shared_ ? cfg.loadExclusiveBufferNumber / maxRanks_ : cfg.loadExclusiveBufferNumber;
        if (nSlotsPerRank_ == 0 || nBuckets_ == 0) {
            return Status::InvalidParam("ctrl header has zero slots per rank or buckets");
        }
        if (cfg.deviceId >= 0) {
            if (cfg.physicalDeviceId < 0) {
                return Status::InvalidParam("invalid physicalDeviceId({})", cfg.physicalDeviceId);
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
            myRank_ = shared_ ? cfg.EffectiveBufferRank() : 0;
            if (myRank_ >= maxRanks_) {
                return Status::InvalidParam("cache rank({}) must be in [0, {})", myRank_,
                                            maxRanks_);
            }
            // 0 = unused, 2 = initializing, 1 = ready, 3 = stopped/failed.
            // A shared domain permits multiple DP participants to attach to the same
            // logical rank segment; exactly one creates and publishes its data.
            auto& ready = ctrl_->Layout().Hdr()->rankDescs[myRank_].ready;
            uint8_t expected = 0;
            ownsRankData_ = ready.compare_exchange_strong(expected, 2, std::memory_order_acq_rel);
            if (!ownsRankData_ && !shared_) { return Status::DuplicateKey(); }
            if (!ownsRankData_ && expected == 3) {
                return Status::Error("cache rank segment is unavailable");
            }
            if (ownsRankData_) { ctrl_->Layout().InitSlotRange(myRank_); }
            data_ = std::make_unique<DataStrategy>();
            auto s = data_->Setup(ctrl_->Layout(), cfg.deviceId, myRank_, slotSize_, nSlotsPerRank_,
                                  ownsRankData_, cfg);
            if (s.Failure()) {
                if (ownsRankData_) { ready.store(3, std::memory_order_release); }
                return s;
            }
            if (ownsRankData_) {
                RankDataDesc desc;
                desc.ready.store(1, std::memory_order_relaxed);
                if (auto publish = ctrl_->Layout().SetRankDesc(myRank_, desc); publish.Failure()) {
                    ready.store(3, std::memory_order_release);
                    return publish;
                }
            }
            if (shared_) {
                s = data_->MapAllSegments(timeoutMs_);
                if (s.Failure()) { return s; }
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
    size_t NumRanks() const { return maxRanks_; }
    size_t NumSlotsPerRank() const { return nSlotsPerRank_; }
    /* Online = the rank completed Setup (sticky: ranks do not leave in this deployment). */
    bool RankReady(size_t rank) const
    {
        if (rank >= maxRanks_) { return false; }
        return ctrl_->Layout().Hdr()->rankDescs[rank].ready.load(std::memory_order_acquire) == 1;
    }

    Handle Get(const Detail::BlockId& blockId, size_t offset, bool allowReserved = false,
               size_t preferredSegment = kInvalidIndex)
    {
        if (myRank_ == kInvalidIndex) { return Handle{}; }
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs_);
        auto attempts = nSlotsPerRank_ > std::numeric_limits<size_t>::max() / 2
                            ? std::numeric_limits<size_t>::max()
                            : 2 * nSlotsPerRank_;
        if (shared_ && attempts <= std::numeric_limits<size_t>::max() / maxRanks_) {
            attempts *= maxRanks_;
        }
        do {
            auto h = TryGet(blockId, offset, allowReserved, attempts, preferredSegment);
            if (h) { return h; }
            std::this_thread::yield();
        } while (timeoutMs_ == 0 || std::chrono::steady_clock::now() < deadline);
        return Handle{};
    }

    // Speculative callers neither wait for a bucket lock nor scan without a bound.
    Handle TryGet(const Detail::BlockId& blockId, size_t offset, bool allowReserved = false,
                  size_t attempts = 128, size_t preferredSegment = kInvalidIndex)
    {
        if (myRank_ == kInvalidIndex || attempts == 0) { return Handle{}; }
        auto iBucket = HashKey(blockId, nBuckets_);
        auto& layout = ctrl_->Layout();
        auto iNode = LookupOptimistic(layout, iBucket, blockId, offset);
        if (iNode != kInvalidIndex) {
            bool owner = false;
            if (PinHit(layout, iNode, iBucket, blockId, offset, kPinSpinFast, owner)) {
                return Handle{this, iNode, owner};
            }
        }
        if (!layout.LockOf(iBucket)->TryLock()) { return Handle{}; }
        iNode = Lookup(layout, iBucket, blockId, offset);
        if (iNode != kInvalidIndex) {
            bool owner = false;
            if (PinHit(layout, iNode, iBucket, blockId, offset, kPinSpinFast, owner)) {
                layout.LockOf(iBucket)->Unlock();
                return Handle{this, iNode, owner};
            }
            layout.LockOf(iBucket)->Unlock();
            return Handle{};
        }
        iNode = Alloc(layout, blockId, offset, iBucket, allowReserved, attempts, preferredSegment);
        layout.LockOf(iBucket)->Unlock();
        return Handle(this, iNode, true);
    }

    void Prealloc(const Detail::BlockId& blockId, size_t offset, bool allowReserved = false,
                  size_t preferredSegment = kInvalidIndex)
    {
        TryPrealloc(blockId, offset, allowReserved, preferredSegment);
    }

    bool TryPrealloc(const Detail::BlockId& blockId, size_t offset, bool allowReserved = false,
                     size_t preferredSegment = kInvalidIndex)
    {
        if (myRank_ == kInvalidIndex) { return false; }
        auto iBucket = HashKey(blockId, nBuckets_);
        auto& layout = ctrl_->Layout();
        if (!layout.LockOf(iBucket)->TryLock()) { return false; }
        auto iNode = Lookup(layout, iBucket, blockId, offset);
        if (iNode != kInvalidIndex) {
            layout.SlotMetaArr()[iNode].accessed.store(1, std::memory_order_relaxed);
            layout.LockOf(iBucket)->Unlock();
            return true;
        }
        /* Publish an unpinned metadata placeholder. A demand Get either observes the
         * transient kSlotClaimed value and retries, or pins reference 0 and becomes the
         * unique data-loading owner. Prealloc itself never impersonates a producer. */
        iNode =
            Alloc(layout, blockId, offset, iBucket, allowReserved, 128, preferredSegment, 0);
        layout.LockOf(iBucket)->Unlock();
        return iNode != kInvalidIndex;
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
            if (PinHit(layout, iNode, iBucket, blockId, offset, kPinSpinFast, owner, false)) {
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

    void* DataAt(size_t slotIdx) { return data_ ? data_->DataAt(slotIdx) : nullptr; }

    void* DeviceDataAt(size_t slotIdx) { return data_ ? data_->DeviceDataAt(slotIdx) : nullptr; }

    size_t SegmentAt(size_t slotIdx) const
    {
        return slotIdx == kInvalidIndex || nSlotsPerRank_ == 0 ? kInvalidIndex
                                                               : slotIdx / nSlotsPerRank_;
    }

private:
    /* Lock-free reader pin: filter by key, CAS the pin, re-validate. Returns false when
     * the key does not match or the slot stays claimed beyond spinBudget (optimistic
     * callers then retry under the bucket lock). On success the caller holds a pin: the
     * slot cannot be reconfigured until Release. owner reports whether the caller must
     * load the block (first pin on a slot that is not Ready yet). */
    bool PinHit(CtrlLayout& layout, size_t iNode, size_t iBucket, const Detail::BlockId& blockId,
                size_t offset, size_t spinBudget, bool& owner, bool takeOwnership = true)
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
            /* An observer must not consume the first pin of an unfilled placeholder:
             * demand Get relies on 0 -> 1 to elect the data-loading owner. */
            if (!takeOwnership && r == 0 &&
                meta->state.load(std::memory_order_acquire) != State::Ready) {
                return false;
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
            owner = (takeOwnership && r == 0 && st != State::Ready);
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
     * the slot is unreachable, then link it into the target bucket and publish
     * initialReference with release. Demand allocation publishes one owner pin;
     * metadata-only preallocation publishes zero for later owner election.
     * Return after a bounded scan so Get can release the bucket lock before retrying. */
    size_t Alloc(CtrlLayout& layout, const Detail::BlockId& blockId, size_t offset, size_t iBucket,
                 bool allowReserved, size_t attempts, size_t preferredSegment,
                 size_t initialReference = 1)
    {
        for (size_t scan = 0; scan < attempts; ++scan) {
            auto iNode = FetchNode(layout, allowReserved, preferredSegment, scan);
            if (iNode == kInvalidIndex) { continue; }
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
                    auto* oldLock = layout.LockOf(oldBucket);
                    auto sameStripe = oldLock == layout.LockOf(iBucket);
                    if (!sameStripe && !oldLock->TryLock()) {
                        meta->reference.store(0, std::memory_order_release);
                        continue;
                    }
                    Remove(layout, oldBucket, iNode);
                    if (!sameStripe) { oldLock->Unlock(); }
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
            meta->reference.store(initialReference, std::memory_order_release);
            return iNode;
        }
        return kInvalidIndex;
    }

    size_t FetchNode(CtrlLayout& layout, bool allowReserved, size_t preferredSegment,
                     size_t attempt)
    {
        auto total = nSlotsPerRank_ - (allowReserved ? 0 : reserved_);
        if (total == 0) { return kInvalidIndex; }
        size_t segment = myRank_;
        if (shared_) {
            const auto first = preferredSegment < maxRanks_ ? preferredSegment : myRank_;
            const auto window = total > std::numeric_limits<size_t>::max() / 2 ? total : 2 * total;
            const auto fallback = window == 0 ? 0 : attempt / window;
            if (fallback >= maxRanks_) { return kInvalidIndex; }
            segment = (first + fallback) % maxRanks_;
            if (!RankReady(segment)) { return kInvalidIndex; }
        }
        auto cur =
            layout.Hdr()->clockHands[segment].fetch_add(1, std::memory_order_relaxed) % total +
            segment * nSlotsPerRank_;
        if (layout.SlotMetaArr()[cur].accessed.exchange(0, std::memory_order_relaxed)) {
            return kInvalidIndex;
        }
        return cur;
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

    size_t ReferenceCount(size_t slotIdx)
    {
        return ctrl_->Layout().SlotMetaArr()[slotIdx].reference.load(std::memory_order_acquire);
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
