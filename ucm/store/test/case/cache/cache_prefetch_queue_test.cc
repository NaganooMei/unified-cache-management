// MIT License
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#include <gtest/gtest.h>
#include "cache/cc/buffer_manager.h"
#include "cache/cc/prefetch_queue.h"
#include "detail/mock_store.h"

namespace {
using namespace UC;
using namespace UC::CacheStore;
using testing::_;
using testing::Invoke;
using testing::Return;

Detail::BlockId Key(unsigned char value)
{
    Detail::BlockId key{};
    key[0] = static_cast<std::byte>(value);
    return key;
}

class CachePrefetchTest : public testing::Test {
protected:
    testing::StrictMock<Test::Detail::MockStore> backend;
    Config config;
    void SetUp() override
    {
        config.storeBackend = &backend;
        config.uniqueId = "prefetch-unit-test";
        config.deviceId = config.physicalDeviceId = 0;
        config.shardSize = 4096;
        config.blockSize = 8192;
        config.bufferCapacity = 4096 * 8;
        config.loadExclusiveBufferNumber = 1;
        config.cachePrefetchEnable = true;
        config.cachePrefetchBatchSize = 2;
    }
};

TEST_F(CachePrefetchTest, PrefixHintsOnlyContainBackendHits)
{
    BufferManager manager;
    ASSERT_TRUE(manager.Setup(config).Success());
    auto* buffer = manager.GetTransBuffer();
    Detail::BlockId keys[]{Key(0), Key(1), Key(2)};
    buffer->Get(keys[0], 0).MarkReady();
    EXPECT_CALL(backend, LookupOnPrefix(_, 2)).WillOnce(Return(0));
    auto result = manager.LookupOnPrefix(keys, 3);
    ASSERT_TRUE(result);
    EXPECT_EQ(result.Value(), 1);
    Detail::BlockId hint[3];
    ASSERT_EQ(buffer->DrainPrefetch(0, hint, 3), 1u);
    EXPECT_EQ(hint[0], keys[1]);
}

TEST_F(CachePrefetchTest, DisabledPrefetchKeepsLookupResult)
{
    config.cachePrefetchEnable = false;
    BufferManager manager;
    ASSERT_TRUE(manager.Setup(config).Success());
    auto key = Key(1);
    EXPECT_CALL(backend, LookupOnPrefix(_, 1)).WillOnce(Return(0));
    EXPECT_EQ(manager.LookupOnPrefix(&key, 1).Value(), 0);
    Detail::BlockId hint;
    EXPECT_EQ(manager.GetTransBuffer()->DrainPrefetch(0, &hint, 1), 0u);
}

TEST_F(CachePrefetchTest, InvalidPrefixResultDoesNotReadPastInput)
{
    BufferManager manager;
    ASSERT_TRUE(manager.Setup(config).Success());
    auto key = Key(1);
    EXPECT_CALL(backend, LookupOnPrefix(_, 1)).WillOnce(Return(1));
    EXPECT_FALSE(manager.LookupOnPrefix(&key, 1));
    EXPECT_CALL(backend, LookupOnPrefix(_, 1)).WillOnce(Return(-2));
    EXPECT_FALSE(manager.LookupOnPrefix(&key, 1));
}

TEST_F(CachePrefetchTest, PrefetchLoadsOnlyFirstShardAndDeduplicates)
{
    Buffer buffer;
    ASSERT_TRUE(buffer.Setup(config).Success());
    PrefetchQueue queue;
    ASSERT_TRUE(queue.Setup(config, &buffer, false).Success());
    Detail::BlockId keys[]{Key(1), Key(1)};
    buffer.EnqueuePrefetch(0, keys, 2);
    EXPECT_CALL(backend, Load(_)).WillOnce(Invoke([&](Detail::TaskDesc task)
                                                   -> Expected<Detail::TaskHandle> {
        EXPECT_EQ(task.size(), 1u);
        EXPECT_EQ(task[0].owner, keys[0]);
        EXPECT_EQ(task[0].index, 0u);
        *static_cast<unsigned char*>(task[0].addrs[0]) = 42;
        return Detail::TaskHandle{1};
    }));
    EXPECT_CALL(backend, Wait(1)).WillOnce(Return(Status::OK()));
    EXPECT_EQ(queue.PollOnce(), 2u);
    auto hit = buffer.Get(keys[0], 0);
    EXPECT_TRUE(hit.Ready());
    EXPECT_FALSE(hit.Owner());
    EXPECT_EQ(*static_cast<unsigned char*>(hit.Data()), 42);
    EXPECT_FALSE(buffer.Exist(keys[0], 1));
    buffer.EnqueuePrefetch(0, keys, 1);
    EXPECT_EQ(queue.PollOnce(), 1u);  // Already Ready: no second backend Load.
}

TEST_F(CachePrefetchTest, DemandDefersPrefetchAndFailureCanRetry)
{
    Buffer buffer;
    ASSERT_TRUE(buffer.Setup(config).Success());
    PrefetchQueue queue;
    ASSERT_TRUE(queue.Setup(config, &buffer, false).Success());
    auto key = Key(2);
    auto demand = buffer.AcquireDemand();
    buffer.EnqueuePrefetch(0, &key, 1);
    EXPECT_EQ(queue.PollOnce(), 0u);
    demand.reset();
    EXPECT_CALL(backend, Load(_)).WillOnce(Return(Detail::TaskHandle{2}));
    EXPECT_CALL(backend, Wait(2)).WillOnce(Return(Status::Error()));
    EXPECT_EQ(queue.PollOnce(), 1u);
    EXPECT_FALSE(buffer.Exist(key, 0));
    auto retry = buffer.Get(key, 0);
    EXPECT_TRUE(retry.Owner());
    EXPECT_EQ(retry.GetState(), State::Loading);
    retry.MarkReady();
    EXPECT_TRUE(buffer.Exist(key, 0));
}

TEST_F(CachePrefetchTest, PinnedCacheDropsPrefetchWithoutUsingDemandReserve)
{
    Buffer buffer;
    ASSERT_TRUE(buffer.Setup(config).Success());
    std::vector<Buffer::Handle> pinned;
    for (size_t i = 0; i < 7; ++i) { pinned.push_back(buffer.Get(Key(i), 0)); }
    PrefetchQueue queue;
    ASSERT_TRUE(queue.Setup(config, &buffer, false).Success());
    auto key = Key(10);
    buffer.EnqueuePrefetch(0, &key, 1);
    EXPECT_EQ(queue.PollOnce(), 1u);  // No backend calls are permitted by StrictMock.
    EXPECT_FALSE(buffer.Exist(key, 0));
    auto demand = buffer.TryGet(key, 0, true);
    EXPECT_TRUE(demand);
}
}  // namespace
