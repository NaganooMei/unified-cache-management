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
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include "mutex/shared_mutex.h"
#include "status/status.h"
#include "type/types.h"

namespace UC::CacheStore {

using BucketLock = UC::SharedMutex;

inline constexpr size_t kMaxRanks = 128;
inline constexpr size_t kInvalidIndex = std::numeric_limits<size_t>::max();
/* Sentinel of the lock-free pin protocol on SlotMeta::reference: a writer that is
 * exclusively reconfiguring the slot. Mutually exclusive with reader pins by CAS. */
inline constexpr size_t kSlotClaimed = std::numeric_limits<size_t>::max();
/* Bucket locks are striped over a fixed lock array, independent of the bucket count:
 * contention depends on the number of threads, not on the number of cached slots. */
inline constexpr size_t kLockStripes = 1ULL << 16;
/* Buckets are sized for the common deployment (ranks < 16 in 99% of cases) with an
 * average chain length of at most 2 at that rank count; larger rank counts degrade
 * chain length gracefully (e.g. 128 ranks -> ~12). */
inline constexpr size_t kBucketSizingRanks = 16;
inline constexpr size_t kMinBuckets = 1ULL << 14;
inline constexpr size_t kMaxBuckets = 1ULL << 24;
/* Depth of each per-rank prefetch command ring (see PrefetchRing). */
inline constexpr size_t kPrefetchDepth = 4096;
/* Bumped on every incompatible SlotMeta / layout change; all ranks sharing one cache
 * domain must run the same binary. */
inline constexpr uint32_t kMagic =
    (static_cast<uint32_t>('U') << 16) | (static_cast<uint32_t>('C') << 8) | 5u;

enum class State : uint8_t { Loading, Ready, Failed };

struct RankDataDesc {
    std::atomic<uint8_t> ready{0};

    RankDataDesc() = default;
    RankDataDesc(const RankDataDesc& o) : ready(o.ready.load(std::memory_order_relaxed)) {}
    RankDataDesc& operator=(const RankDataDesc& o)
    {
        ready.store(o.ready.load(std::memory_order_relaxed), std::memory_order_relaxed);
        return *this;
    }
};

/* Per-slot metadata. All fields are atomic: readers walk hash buckets without holding the
 * bucket lock, so every read must be race-free.
 *
 * Synchronization model (see Buffer for the full protocol):
 *  - reference is the only coordination point. 0 = reclaimable, kSlotClaimed = a writer
 *    owns the slot exclusively while reconfiguring it, (0, n) = n readers hold pins.
 *  - A reader pins via CAS(r -> r + 1) and must re-validate the key after the CAS: once
 *    pinned, no writer can be concurrent, so the re-validation is a consistent snapshot.
 *  - A writer claims via CAS(0 -> kSlotClaimed), unlinks the slot (hash = kInvalidIndex,
 *    unreachable for bucket walkers), rewrites the key, then publishes by linking the slot
 *    into the target bucket (hash stored with release, before the bucket head store) and
 *    finally reference.store(1, release). Acquiring readers therefore observe the new key.
 *  - state written with release by MarkReady/MarkFailed is visible to later pinners: the
 *    previous owner's Release (fetch_sub, release) keeps the release sequence alive, so
 *    the next pin's acquire-CAS synchronizes with everything the owner did before release.
 */
struct SlotMeta {
    /* Hot line: CAS contention point of the pin protocol. */
    alignas(64) std::atomic<size_t> reference{0};
    /* Cache key snapshot: key[0]/key[1] = BlockId (16B), key[2] = offset. */
    alignas(8) std::atomic<size_t> key[3]{0, 0, 0};
    /* Owning bucket index; kInvalidIndex while unlinked / being reconfigured. */
    std::atomic<size_t> hash{kInvalidIndex};
    std::atomic<size_t> prev{kInvalidIndex};
    std::atomic<size_t> next{kInvalidIndex};
    alignas(64) std::atomic<State> state{State::Loading};
    /* Hot line: CLOCK second-chance bit. */
    alignas(64) std::atomic<uint8_t> accessed{0};

    void Init()
    {
        reference.store(0, std::memory_order_relaxed);
        key[0].store(0, std::memory_order_relaxed);
        key[1].store(0, std::memory_order_relaxed);
        key[2].store(0, std::memory_order_relaxed);
        hash.store(kInvalidIndex, std::memory_order_relaxed);
        prev.store(kInvalidIndex, std::memory_order_relaxed);
        next.store(kInvalidIndex, std::memory_order_relaxed);
        state.store(State::Loading, std::memory_order_relaxed);
        accessed.store(0, std::memory_order_relaxed);
    }
};

/* Per-rank SPSC command ring for prefetch requests: producer is the scheduler's
 * (single-threaded) Prefetch caller, consumer is the owning worker's prefetch executor
 * thread. Entries are dropped and counted when the ring is full — prefetch is a
 * fire-and-forget hint. Entry publication: relaxed write + head release-store; slot
 * reclamation: entry read + tail release-store. */
struct PrefetchRing {
    alignas(64) std::atomic<uint64_t> head{0};
    alignas(64) std::atomic<uint64_t> tail{0};
    alignas(64) std::atomic<uint64_t> dropped{0};
    Detail::BlockId entries[kPrefetchDepth];
};

struct Header {
    std::atomic<uint32_t> magic{0};
    size_t maxRanks{0};
    size_t nSlotsPerRank{0};
    size_t slotSize{0};
    /* Power-of-two bucket count; joiners derive all layout offsets from it. */
    size_t nBuckets{0};
    alignas(64) RankDataDesc rankDescs[kMaxRanks];
    alignas(64) std::atomic<size_t> clockHands[kMaxRanks];
};

static_assert(std::atomic<uint32_t>::is_always_lock_free, "magic must be lock-free");
static_assert(std::atomic<uint8_t>::is_always_lock_free, "accessed/ready must be lock-free");
static_assert(std::atomic<size_t>::is_always_lock_free,
              "size_t atomics (indices, reference, key words) must be lock-free");
static_assert(std::atomic<State>::is_always_lock_free, "state must be lock-free");
static_assert(std::atomic<uint64_t>::is_always_lock_free, "ring counters must be lock-free");

inline size_t AlignUp(size_t value, size_t align) { return (value + align - 1) & ~(align - 1); }

inline size_t NextPow2(size_t value)
{
    size_t p = 1;
    while (p < value) { p <<= 1; }
    return p;
}

/* Bucket count for a cache domain: chain length <= 2 at kBucketSizingRanks, clamped to a
 * sane range so tiny domains do not degenerate and huge ones do not waste head memory. */
inline size_t CalcBucketCount(size_t nSlotsPerRank)
{
    auto target = kBucketSizingRanks * nSlotsPerRank / 2;
    return std::min(std::max(NextPow2(target), kMinBuckets), kMaxBuckets);
}

/* Mask-based bucket index; nBuckets must be a power of two. BlockIdHasher is
 * murmur-grade, so the low bits used by the mask are well distributed. */
inline size_t HashKey(const Detail::BlockId& blockId, size_t nBuckets)
{
    static Detail::BlockIdHasher blockHasher;
    return blockHasher(blockId) & (nBuckets - 1);
}

/* View a BlockId as two size_t words. memcpy (not a cast) is mandatory: BlockId is only
 * 1-byte aligned, so a uint64_t* reinterpretation would be an alignment and
 * strict-aliasing violation; a constant-size memcpy compiles to the same two moves. */
inline void BlockKeyWords(const Detail::BlockId& blockId, size_t (&words)[2])
{
    std::memcpy(&words[0], blockId.data(), sizeof(Detail::BlockId));
}

static_assert(sizeof(Detail::BlockId) == 2 * sizeof(size_t), "BlockId must be two size_t words");

/* Rewrite the key. Only legal while the slot is claimed (unlinked). */
inline void StoreKey(SlotMeta& meta, const Detail::BlockId& blockId, size_t offset)
{
    size_t w[2];
    BlockKeyWords(blockId, w);
    meta.key[0].store(w[0], std::memory_order_relaxed);
    meta.key[1].store(w[1], std::memory_order_relaxed);
    meta.key[2].store(offset, std::memory_order_relaxed);
}

/* Match a slot against a lookup: bucket filter first, then a short-circuit word
 * comparison (key[0]/key[1] = block id, key[2] = offset). Byte-level equivalent to
 * comparing the materialized BlockId, but skips the copy and exits on the first
 * differing word. The slot must not be concurrently reconfigured for the word reads
 * to be a consistent snapshot (guaranteed post-pin, or filtered by re-validation). */
inline bool MatchKey(const SlotMeta& meta, size_t iBucket, const Detail::BlockId& blockId,
                     size_t offset)
{
    if (meta.hash.load(std::memory_order_acquire) != iBucket) { return false; }
    size_t w[2];
    BlockKeyWords(blockId, w);
    return meta.key[0].load(std::memory_order_acquire) == w[0] &&
           meta.key[1].load(std::memory_order_acquire) == w[1] &&
           meta.key[2].load(std::memory_order_acquire) == offset;
}

}  // namespace UC::CacheStore
