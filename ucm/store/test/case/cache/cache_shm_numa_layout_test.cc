/**
 * MIT License
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 */
#include <gtest/gtest.h>
#include <numeric>
#include "cache/cc/shm_numa_layout.h"

namespace {
namespace Numa = UC::CacheStore::ShmNuma;

TEST(UCCacheShmNumaLayoutTest, ParsesAndValidatesNodeLists)
{
    EXPECT_EQ(Numa::ParseNodes("0-3,7"), (std::vector<size_t>{0, 1, 2, 3, 7}));
    EXPECT_THROW(Numa::ParseNodes("3-1"), std::invalid_argument);
    EXPECT_THROW(Numa::ParseNodes("1,1"), std::invalid_argument);
    EXPECT_THROW(Numa::ValidateNodes({0, 0}), std::invalid_argument);
    EXPECT_EQ(Numa::SelectMemoryNodes({0, 2, 5}, {2, 3, 5}), (std::vector<size_t>{2, 5}));
}

TEST(UCCacheShmNumaLayoutTest, DividesPagesWithoutLosingCapacity)
{
    constexpr size_t page = 4096;
    const auto ranges = Numa::Plan(17 * page - 1, page, {0, 2, 4});
    ASSERT_EQ(ranges.size(), 3u);
    size_t end = 0;
    size_t smallest = std::numeric_limits<size_t>::max();
    size_t largest = 0;
    for (const auto& range : ranges) {
        EXPECT_EQ(range.offset, end);
        EXPECT_EQ(range.bytes % page, 0u);
        end += range.bytes;
        smallest = std::min(smallest, range.bytes);
        largest = std::max(largest, range.bytes);
    }
    EXPECT_GE(end, 17 * page - 1);
    EXPECT_LT(end - (17 * page - 1), page);
    EXPECT_LE(largest - smallest, page);
}

TEST(UCCacheShmNumaLayoutTest, MapsSegmentsEvenlyAcrossNodes)
{
    const std::vector<size_t> nodes{0, 1, 2, 3, 4, 5, 6, 7};
    std::vector<size_t> counts(nodes.size());
    for (size_t segment = 0; segment < 16; ++segment) {
        const auto selected = Numa::SegmentNodes(nodes, 16, segment);
        ASSERT_EQ(selected.size(), 1u);
        ++counts[selected.front()];
    }
    EXPECT_EQ(counts, std::vector<size_t>(8, 2));
    EXPECT_EQ(Numa::SegmentNodes(nodes, 4, 0), (std::vector<size_t>{0, 1}));
    EXPECT_EQ(Numa::SegmentNodes(nodes, 4, 3), (std::vector<size_t>{6, 7}));
    EXPECT_EQ(Numa::SegmentNodes({2, 4, 6}, 2, 1), (std::vector<size_t>{2, 4, 6}));
}

TEST(UCCacheShmNumaLayoutTest, PrefersDetectedDeviceNodeForSharedAndPrivateData)
{
    EXPECT_EQ(Numa::DataNodes(3, {0, 1}, 8, 5, true), (std::vector<size_t>{3}));
    EXPECT_EQ(Numa::DataNodes(3, {}, 1, 0, false), (std::vector<size_t>{3}));
    EXPECT_EQ(Numa::DataNodes(std::nullopt, {0, 1}, 8, 5, true),
              (std::vector<size_t>{1}));
    EXPECT_EQ(Numa::DataNodes(std::nullopt, {0, 1}, 1, 0, true),
              (std::vector<size_t>{0}));
    EXPECT_TRUE(Numa::DataNodes(std::nullopt, {}, 1, 0, false).empty());
}

TEST(UCCacheShmNumaLayoutTest, SpreadsFallbackRanksAcrossAvailableNodes)
{
    const std::vector<size_t> nodes{0, 2, 4, 6};
    EXPECT_EQ(Numa::RankNode(nodes, 0), (std::vector<size_t>{0}));
    EXPECT_EQ(Numa::RankNode(nodes, 3), (std::vector<size_t>{6}));
    EXPECT_EQ(Numa::RankNode(nodes, 4), (std::vector<size_t>{0}));
    EXPECT_TRUE(Numa::RankNode({}, 7).empty());
}

TEST(UCCacheShmNumaLayoutTest, BuildsKernelNodeMask)
{
    constexpr size_t bits = sizeof(unsigned long) * 8;
    for (size_t node : {0, 1, 63, 64, 127, 128}) {
        const auto mask = Numa::SingleNodeMask(node);
        const auto consumedBits = mask.maxNode - 1;
        ASSERT_GT(consumedBits, node);
        EXPECT_NE(mask.words[node / bits] & (1UL << (node % bits)), 0UL);
    }
}

}  // namespace
