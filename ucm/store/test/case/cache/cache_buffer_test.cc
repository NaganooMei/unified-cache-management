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
#include "cache_buffer.h"
#include <atomic>
#include <chrono>
#include <cstring>
#include <gtest/gtest.h>
#include <memory>
#include <mutex>
#include <set>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {
UC::CacheStore::Config MakeConfig(int32_t deviceId, size_t m = 64)
{
    UC::CacheStore::Config cfg;
    cfg.deviceId = deviceId;
    cfg.physicalDeviceId = deviceId;
    cfg.shardSize = 4096;
    cfg.alignSize = 4096;
    cfg.bufferCapacity = 4096 * m;
    cfg.loadExclusiveBufferNumber = 0;
    cfg.timeoutMs = 5000;
    return cfg;
}

UC::Detail::BlockId MakeBlockId(char c)
{
    UC::Detail::BlockId b;
    b.fill(static_cast<std::byte>(c));
    return b;
}

UC::Detail::BlockId MakeBlockIdN(uint32_t v)
{
    UC::Detail::BlockId b;
    b.fill(static_cast<std::byte>(0));
    std::memcpy(b.data(), &v, sizeof(v));
    return b;
}

}  // namespace

TEST(UcmV2CacheBufferTest, SingleProcessAllocGetExist)
{
    auto cfg = MakeConfig(0);
    UC::CacheStore::Buffer buf;
    ASSERT_TRUE(buf.Setup(cfg).Success());

    auto blk = MakeBlockId('a');
    EXPECT_FALSE(buf.Exist(blk, 0));
    auto h1 = buf.Get(blk, 0);
    ASSERT_TRUE(h1);
    ASSERT_TRUE(h1.Owner());
    EXPECT_NE(h1.Data(), nullptr);
    EXPECT_FALSE(h1.Ready());
    h1.MarkReady();
    EXPECT_TRUE(h1.Ready());
    EXPECT_TRUE(buf.Exist(blk, 0));
    auto h2 = buf.Get(blk, 0);
    ASSERT_TRUE(h2);
    EXPECT_FALSE(h2.Owner());
    EXPECT_TRUE(h2.Ready());
    EXPECT_EQ(h1.Data(), h2.Data());
    buf.Touch(&blk, 1);
    EXPECT_TRUE(buf.Exist(blk, 0));
}

TEST(UcmV2CacheBufferTest, ClockEviction)
{
    constexpr size_t M = 8;
    auto cfg = MakeConfig(0, M);
    UC::CacheStore::Buffer buf;
    ASSERT_TRUE(buf.Setup(cfg).Success());

    for (size_t i = 0; i < M; i++) {
        auto blk = MakeBlockId(static_cast<char>('a' + i));
        auto h = buf.Get(blk, 0);
        ASSERT_TRUE(h);
        h.MarkReady();
    }
    for (size_t i = 0; i < M; i++) {
        auto blk = MakeBlockId(static_cast<char>('a' + i));
        EXPECT_TRUE(buf.Exist(blk, 0));
    }
    auto blkNew = MakeBlockId('z');
    auto h = buf.Get(blkNew, 0);
    ASSERT_TRUE(h);
    h.MarkReady();
    EXPECT_TRUE(buf.Exist(blkNew, 0));
}

TEST(UcmV2CacheBufferTest, ForkTwoProcessesCtrlShared)
{
    auto cfg0 = MakeConfig(0);
    UC::CacheStore::Buffer buf0;
    ASSERT_TRUE(buf0.Setup(cfg0).Success());

    auto blk = MakeBlockId('x');
    {
        auto h = buf0.Get(blk, 0);
        h.MarkReady();
    }
    ASSERT_TRUE(buf0.Exist(blk, 0));

    pid_t pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        auto cfg1 = MakeConfig(1);
        UC::CacheStore::Buffer buf1;
        auto s = buf1.Setup(cfg1);
        if (s.Failure()) { _exit(2); }
        bool exist = buf1.Exist(blk, 0);
        _exit(exist ? 0 : 1);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    ASSERT_TRUE(WIFEXITED(status)) << "child crashed";
    EXPECT_EQ(WEXITSTATUS(status), 0) << "child failed to see remote cached block";
}

TEST(UcmV2CacheBufferTest, CrossRankDataFetch)
{
    auto cfg0 = MakeConfig(0);
    UC::CacheStore::Buffer buf0;
    ASSERT_TRUE(buf0.Setup(cfg0).Success());

    auto blk = MakeBlockId('p');
    {
        auto h = buf0.Get(blk, 0);
        h.MarkReady();
    }

    pid_t pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        auto cfg1 = MakeConfig(1);
        UC::CacheStore::Buffer buf1;
        if (buf1.Setup(cfg1).Failure()) { _exit(2); }
        auto h = buf1.Get(blk, 0);
        if (!h) { _exit(3); }
        void* d = h.Data();
        _exit(d != nullptr ? 0 : 4);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    ASSERT_TRUE(WIFEXITED(status)) << "child crashed";
    EXPECT_EQ(WEXITSTATUS(status), 0) << "cross-rank data fetch via socket failed";
}

TEST(UcmV2CacheBufferTest, CrossRankDataFetchNonMultipleCapacity)
{
    /* capacity is not a multiple of slotSize: the remote mapping must use the exact
     * per-rank data window (m * slotSize), never the raw capacity. */
    constexpr size_t M = 8;
    auto cfg0 = MakeConfig(0, M);
    cfg0.bufferCapacity = 4096 * M + 2048;
    UC::CacheStore::Buffer buf0;
    ASSERT_TRUE(buf0.Setup(cfg0).Success());

    auto blk = MakeBlockId('n');
    {
        auto h = buf0.Get(blk, 0);
        h.MarkReady();
    }

    pid_t pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        auto cfg1 = MakeConfig(1, M);
        cfg1.bufferCapacity = 4096 * M + 2048;
        UC::CacheStore::Buffer buf1;
        if (buf1.Setup(cfg1).Failure()) { _exit(2); }
        auto h = buf1.Get(blk, 0);
        if (!h) { _exit(3); }
        void* d = h.Data();
        if (d == nullptr) { _exit(4); }
        /* Touch the last slot of the window to prove the mapping covers the whole
         * per-rank data block. */
        auto* p = static_cast<std::byte*>(d);
        volatile std::byte last = p[4095];
        (void)last;
        _exit(0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    ASSERT_TRUE(WIFEXITED(status)) << "child crashed";
    EXPECT_EQ(WEXITSTATUS(status), 0) << "cross-rank fetch with non-multiple capacity failed";
}

TEST(UcmV2CacheBufferTest, ConcurrentGetSameKey)
{
    auto cfg = MakeConfig(0);
    UC::CacheStore::Buffer buf;
    ASSERT_TRUE(buf.Setup(cfg).Success());
    auto blk = MakeBlockId('c');
    {
        auto h = buf.Get(blk, 0);
        h.MarkReady();
    }
    constexpr int kThreads = 8;
    constexpr int kIters = 1000;
    std::atomic<int> hits{0};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int i = 0; i < kThreads; i++) {
        threads.emplace_back([&] {
            for (int j = 0; j < kIters; j++) {
                auto h = buf.Get(blk, 0);
                if (h && h.Ready()) { hits++; }
            }
        });
    }
    for (auto& t : threads) { t.join(); }
    EXPECT_EQ(hits.load(), kThreads * kIters);
}

TEST(UcmV2CacheBufferTest, ConcurrentGetDistinctKeys)
{
    constexpr size_t M = 64;
    auto cfg = MakeConfig(0, M);
    UC::CacheStore::Buffer buf;
    ASSERT_TRUE(buf.Setup(cfg).Success());
    std::vector<UC::Detail::BlockId> keys;
    for (size_t i = 0; i < M; i++) { keys.push_back(MakeBlockId(static_cast<char>('a' + i))); }
    constexpr int kThreads = 8;
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int i = 0; i < kThreads; i++) {
        threads.emplace_back([&] {
            for (size_t j = 0; j < keys.size(); j++) {
                auto h = buf.Get(keys[j], 0);
                if (h && h.Owner()) { h.MarkReady(); }
            }
        });
    }
    for (auto& t : threads) { t.join(); }
    for (auto& blk : keys) { EXPECT_TRUE(buf.Exist(blk, 0)); }
}

TEST(UcmV2CacheBufferTest, SameBlockDifferentOffsetsInSameBucket)
{
    auto cfg = MakeConfig(0);
    UC::CacheStore::Buffer buf;
    ASSERT_TRUE(buf.Setup(cfg).Success());
    auto blk = MakeBlockId('a');
    {
        auto h = buf.Get(blk, 0);
        h.MarkReady();
    }
    {
        auto h = buf.Get(blk, 16);
        h.MarkReady();
    }
    EXPECT_TRUE(buf.Exist(blk, 0));
    EXPECT_TRUE(buf.Exist(blk, 16));
    buf.Touch(&blk, 1);
    SUCCEED();
}

TEST(UcmV2CacheBufferTest, HashCollisionDifferentKeyNotMatched)
{
    auto cfg = MakeConfig(0);
    UC::CacheStore::Buffer buf;
    ASSERT_TRUE(buf.Setup(cfg).Success());
    auto nb = buf.NumBuckets();
    ASSERT_GT(nb, 1u);
    auto blk = MakeBlockId('a');
    /* Brute-force a different key that lands in the same bucket as blk (~16k tries). */
    auto target = UC::CacheStore::HashKey(blk, nb);
    UC::Detail::BlockId other{};
    bool collided = false;
    for (uint32_t v = 1; v < 0x1000000u; v++) {
        UC::Detail::BlockId b;
        b.fill(static_cast<std::byte>(0));
        std::memcpy(b.data(), &v, sizeof(v));
        if (UC::CacheStore::HashKey(b, nb) == target && b != blk) {
            other = b;
            collided = true;
            break;
        }
    }
    ASSERT_TRUE(collided) << "no colliding key found";
    {
        auto h = buf.Get(blk, 0);
        ASSERT_TRUE(h);
        ASSERT_TRUE(h.Owner());
        h.MarkReady();
    }
    /* The colliding key must not be confused with the cached entry. */
    EXPECT_FALSE(buf.Exist(other, 0));
    {
        auto h = buf.Get(other, 0);
        ASSERT_TRUE(h);
        EXPECT_TRUE(h.Owner());
        h.MarkReady();
    }
    EXPECT_TRUE(buf.Exist(blk, 0));
    EXPECT_TRUE(buf.Exist(other, 0));
    /* Touch matches whole blocks; both entries stay distinct through a batch walk. */
    UC::Detail::BlockId arr[2] = {blk, other};
    buf.Touch(arr, 2);
    EXPECT_TRUE(buf.Exist(blk, 0));
    EXPECT_TRUE(buf.Exist(other, 0));
}

TEST(UcmV2CacheBufferTest, BucketCountScales)
{
    namespace C = UC::CacheStore;
    /* Floor for tiny domains (m=2..64). */
    EXPECT_EQ(C::CalcBucketCount(2), C::kMinBuckets);
    EXPECT_EQ(C::CalcBucketCount(64), C::kMinBuckets);
    /* 32 GiB / 176 KB -> m = 190650 -> 2^21; 128 GiB -> m = 762600 -> 2^23. */
    EXPECT_EQ(C::CalcBucketCount(190650), 1ULL << 21);
    EXPECT_EQ(C::CalcBucketCount(762600), 1ULL << 23);
    /* Cap for very large domains. */
    EXPECT_EQ(C::CalcBucketCount(3ULL << 20), C::kMaxBuckets);
    /* End-to-end: the creator wires CalcBucketCount into the shared header. */
    auto cfg = MakeConfig(0);
    UC::CacheStore::Buffer buf;
    ASSERT_TRUE(buf.Setup(cfg).Success());
    EXPECT_EQ(buf.NumBuckets(), C::kMinBuckets);
}

TEST(UcmV2CacheBufferTest, ReadySlotNeverOwner)
{
    auto cfg = MakeConfig(0);
    UC::CacheStore::Buffer buf;
    ASSERT_TRUE(buf.Setup(cfg).Success());
    auto blk = MakeBlockId('r');
    {
        auto h = buf.Get(blk, 0);
        ASSERT_TRUE(h.Owner());
        h.MarkReady();
    }
    auto h2 = buf.Get(blk, 0);
    ASSERT_TRUE(h2);
    EXPECT_FALSE(h2.Owner());
    EXPECT_TRUE(h2.Ready());
}

TEST(UcmV2CacheBufferTest, PreallocThenGetOwnerLoading)
{
    auto cfg = MakeConfig(0);
    UC::CacheStore::Buffer buf;
    ASSERT_TRUE(buf.Setup(cfg).Success());
    auto blk = MakeBlockId('w');
    buf.Prealloc(blk, 0);
    EXPECT_TRUE(buf.Exist(blk, 0));
    auto h = buf.Get(blk, 0);
    ASSERT_TRUE(h);
    EXPECT_TRUE(h.Owner());
    EXPECT_FALSE(h.Ready());
    h.MarkReady();
    EXPECT_TRUE(h.Ready());
    buf.Prealloc(blk, 0);
    EXPECT_TRUE(buf.Exist(blk, 0));
}

TEST(UcmV2CacheBufferTest, FailedThenOwnerRetry)
{
    auto cfg = MakeConfig(0);
    UC::CacheStore::Buffer buf;
    ASSERT_TRUE(buf.Setup(cfg).Success());
    auto blk = MakeBlockId('f');
    {
        auto h = buf.Get(blk, 0);
        ASSERT_TRUE(h);
        ASSERT_TRUE(h.Owner());
        h.MarkFailed();
        EXPECT_FALSE(h.Ready());
        EXPECT_EQ(h.GetState(), UC::CacheStore::State::Failed);
    }
    {
        /* A failed slot with no references must be handed to a new owner and reset. */
        auto h = buf.Get(blk, 0);
        ASSERT_TRUE(h);
        EXPECT_TRUE(h.Owner());
        EXPECT_FALSE(h.Ready());
        h.MarkReady();
        EXPECT_TRUE(h.Ready());
    }
    {
        auto h = buf.Get(blk, 0);
        ASSERT_TRUE(h);
        EXPECT_FALSE(h.Owner());
        EXPECT_TRUE(h.Ready());
    }
}

TEST(UcmV2CacheBufferTest, ConcurrentTouchWhileAlloc)
{
    constexpr size_t M = 32;
    auto cfg = MakeConfig(0, M);
    UC::CacheStore::Buffer buf;
    ASSERT_TRUE(buf.Setup(cfg).Success());
    std::vector<UC::Detail::BlockId> keys;
    for (size_t i = 0; i < M; i++) { keys.push_back(MakeBlockId(static_cast<char>('a' + i))); }
    for (auto& k : keys) {
        auto h = buf.Get(k, 0);
        ASSERT_TRUE(h);
        h.MarkReady();
    }
    std::atomic<bool> stop{false};
    std::vector<std::thread> threads;
    threads.emplace_back([&] {
        while (!stop.load(std::memory_order_relaxed)) { buf.Touch(keys.data(), keys.size()); }
    });
    for (int t = 0; t < 2; t++) {
        threads.emplace_back([&, t] {
            size_t gen = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                for (size_t i = 0; i < M; i++) {
                    UC::Detail::BlockId b = keys[i];
                    b[14] = static_cast<std::byte>(gen & 0xFF);
                    b[15] = static_cast<std::byte>(static_cast<char>('0' + t));
                    auto h = buf.Get(b, 0);
                    if (h && h.Owner()) { h.MarkReady(); }
                }
                gen++;
            }
        });
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    stop.store(true, std::memory_order_relaxed);
    for (auto& th : threads) { th.join(); }
    SUCCEED();
}

TEST(UcmV2CacheBufferTest, PinStormReallocStress)
{
    /* A pinned slot must never be reconfigured: while the handle is held, concurrent
     * allocations may recycle every other slot, but the pinned slot's key, state and
     * data must stay intact. */
    constexpr size_t M = 16;
    constexpr size_t kSlotSize = 4096;
    auto cfg = MakeConfig(0, M);
    UC::CacheStore::Buffer buf;
    ASSERT_TRUE(buf.Setup(cfg).Success());
    auto blk = MakeBlockId('P');
    auto h = buf.Get(blk, 0);
    ASSERT_TRUE(h);
    ASSERT_TRUE(h.Owner());
    auto* p = static_cast<unsigned char*>(h.Data());
    ASSERT_NE(p, nullptr);
    for (size_t i = 0; i < kSlotSize; i++) { p[i] = static_cast<unsigned char>(i & 0xFF); }
    h.MarkReady();

    constexpr int kThreads = 4;
    std::atomic<bool> stop{false};
    std::atomic<bool> corrupted{false};
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; t++) {
        threads.emplace_back([&, t] {
            size_t gen = 0;
            while (!stop.load(std::memory_order_relaxed)) {
                for (size_t i = 0; i < M; i++) {
                    UC::Detail::BlockId b;
                    b.fill(static_cast<std::byte>(static_cast<char>('A' + t)));
                    b[0] = static_cast<std::byte>(gen & 0xFF);
                    b[1] = static_cast<std::byte>(i);
                    auto hh = buf.Get(b, 0);
                    if (!hh) { corrupted.store(true); }
                    if (hh && hh.Owner()) { hh.MarkReady(); }
                }
                gen++;
            }
        });
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    for (size_t i = 0; i < kSlotSize; i++) {
        EXPECT_EQ(p[i], static_cast<unsigned char>(i & 0xFF))
            << "pinned slot data clobbered at " << i;
        if (p[i] != static_cast<unsigned char>(i & 0xFF)) { break; }
    }
    EXPECT_TRUE(h.Ready());
    stop.store(true, std::memory_order_relaxed);
    for (auto& th : threads) { th.join(); }
    EXPECT_FALSE(corrupted.load());
}

TEST(UcmV2CacheBufferTest, PrefetchRingFifoOverflowDropReuse)
{
    namespace C = UC::CacheStore;
    auto cfg = MakeConfig(-1); /* control-plane-only creator: no data plane */
    C::Buffer buf;
    ASSERT_TRUE(buf.Setup(cfg).Success());

    /* FIFO order. */
    std::vector<UC::Detail::BlockId> in;
    for (uint32_t i = 0; i < 10; i++) { in.push_back(MakeBlockIdN(i + 1)); }
    buf.EnqueuePrefetch(3, in.data(), in.size());
    std::vector<UC::Detail::BlockId> out(C::kPrefetchDepth);
    EXPECT_EQ(buf.DrainPrefetch(3, out.data(), out.size()), 10u);
    for (size_t i = 0; i < 10; i++) { EXPECT_EQ(out[i], in[i]); }
    EXPECT_EQ(buf.DrainPrefetch(3, out.data(), out.size()), 0u);

    /* Overflow keeps the front of the batch and drops (counts) the remainder. */
    std::vector<UC::Detail::BlockId> big;
    for (uint32_t i = 0; i < C::kPrefetchDepth + 7; i++) { big.push_back(MakeBlockIdN(10000 + i)); }
    buf.EnqueuePrefetch(3, big.data(), big.size());
    EXPECT_EQ(buf.PrefetchDropped(3), 7u);
    size_t got = 0;
    size_t n;
    while ((n = buf.DrainPrefetch(3, out.data(), 1000)) > 0) {
        for (size_t i = 0; i < n; i++) { EXPECT_EQ(out[i], big[got + i]); }
        got += n;
    }
    EXPECT_EQ(got, C::kPrefetchDepth);
    EXPECT_EQ(buf.PrefetchDropped(3), 7u);

    /* Reuse after wraparound. */
    buf.EnqueuePrefetch(3, in.data(), in.size());
    EXPECT_EQ(buf.DrainPrefetch(3, out.data(), out.size()), 10u);
    for (size_t i = 0; i < 10; i++) { EXPECT_EQ(out[i], in[i]); }
    EXPECT_EQ(buf.PrefetchDropped(3), 7u);
}
