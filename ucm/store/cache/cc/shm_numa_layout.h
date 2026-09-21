/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace UC::CacheStore::ShmNuma {

// Physical node IDs, in the user's order. Ranges such as 0-7 are accepted.
inline std::vector<size_t> ParseNodes(std::string_view text)
{
    std::vector<size_t> nodes;
    auto parse = [](std::string_view token) {
        size_t value = 0;
        const auto [end, error] = std::from_chars(token.data(), token.data() + token.size(), value);
        if (token.empty() || error != std::errc() || end != token.data() + token.size() ||
            value > 65535) {
            throw std::invalid_argument("invalid NUMA node ID: " + std::string(token));
        }
        return value;
    };
    do {
        const auto comma = text.find(',');
        const auto token = text.substr(0, comma);
        const auto dash = token.find('-');
        const auto first = parse(token.substr(0, dash));
        const auto last = dash == std::string_view::npos ? first : parse(token.substr(dash + 1));
        if (last < first) { throw std::invalid_argument("descending NUMA node range"); }
        for (size_t node = first; node <= last; ++node) {
            if (std::find(nodes.begin(), nodes.end(), node) != nodes.end()) {
                throw std::invalid_argument("duplicate NUMA node: " + std::to_string(node));
            }
            nodes.push_back(node);
        }
        if (comma == std::string_view::npos) { break; }
        text.remove_prefix(comma + 1);
    } while (true);
    return nodes;
}

struct Range {
    size_t offset;
    size_t bytes;
    size_t node;
};

struct NodeMask {
    std::vector<unsigned long> words;
    unsigned long maxNode;
};

inline NodeMask SingleNodeMask(size_t node)
{
    if (node > 65535) { throw std::invalid_argument("NUMA node ID exceeds 65535"); }
    constexpr size_t bitsPerWord = sizeof(unsigned long) * 8;
    std::vector<unsigned long> words(node / bitsPerWord + 1, 0);
    words[node / bitsPerWord] = 1UL << (node % bitsPerWord);
    // Linux get_nodes() decrements maxnode before copying/masking the bitmap.
    const auto maxNode = static_cast<unsigned long>(words.size() * bitsPerWord + 1);
    return {std::move(words), maxNode};
}

inline void ValidateNodes(const std::vector<size_t>& nodes)
{
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (nodes[i] > 65535 ||
            std::find(nodes.begin(), nodes.begin() + i, nodes[i]) != nodes.begin() + i) {
            throw std::invalid_argument("invalid or duplicate share_buffer_numa_nodes");
        }
    }
}

inline std::vector<size_t> SelectMemoryNodes(const std::vector<size_t>& memory,
                                             const std::vector<size_t>& allowed)
{
    ValidateNodes(memory);
    std::vector<size_t> nodes;
    for (const auto node : memory) {
        if (std::find(allowed.begin(), allowed.end(), node) != allowed.end()) {
            nodes.push_back(node);
        }
    }
    if (nodes.empty()) { throw std::invalid_argument("no allowed NUMA memory nodes"); }
    return nodes;
}

// Distribute complete base pages with at most one page of imbalance.
inline std::vector<Range> Plan(size_t bytes, size_t pageSize, const std::vector<size_t>& nodes)
{
    ValidateNodes(nodes);
    if (nodes.empty()) { return {}; }
    if (bytes == 0 || pageSize == 0 ||
        bytes > std::numeric_limits<size_t>::max() - (pageSize - 1)) {
        throw std::invalid_argument("invalid SHM NUMA size or page size");
    }
    const auto pages = (bytes + pageSize - 1) / pageSize;
    if (pages < nodes.size()) {
        throw std::invalid_argument("SHM has fewer pages than NUMA nodes");
    }
    std::vector<Range> ranges;
    size_t offset = 0;
    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto rangeBytes = (pages / nodes.size() + (i < pages % nodes.size())) * pageSize;
        ranges.push_back({offset, rangeBytes, nodes[i]});
        offset += rangeBytes;
    }
    return ranges;
}

inline std::vector<size_t> SegmentNodes(const std::vector<size_t>& nodes, size_t segments,
                                        size_t segment)
{
    if (nodes.empty()) { return {}; }
    ValidateNodes(nodes);
    if (segments == 0 || segment >= segments) {
        throw std::invalid_argument("invalid shared buffer segment count or index");
    }
    const auto groups = std::gcd(segments, nodes.size());
    const auto perGroup = nodes.size() / groups;
    const auto first = (segment % groups) * perGroup;
    return {nodes.begin() + first, nodes.begin() + first + perGroup};
}

inline std::vector<size_t> DataNodes(const std::optional<size_t>& detectedNode,
                                     const std::vector<size_t>& sharedNodes, size_t segments,
                                     size_t segment)
{
    if (detectedNode.has_value()) { return {*detectedNode}; }
    if (sharedNodes.empty()) { return {}; }
    // A single shared segment (for example MLA DP8 TP1) is one physical SHM
    // allocation shared by every DP participant. Keep it on one deterministic
    // NUMA node instead of striping the one segment or relying on first-touch.
    if (segments == 1) { return {sharedNodes.front()}; }
    return SegmentNodes(sharedNodes, segments, segment);
}

inline std::vector<size_t> RankNode(const std::vector<size_t>& nodes, size_t rank)
{
    if (nodes.empty()) { return {}; }
    ValidateNodes(nodes);
    return {nodes[rank % nodes.size()]};
}

}  // namespace UC::CacheStore::ShmNuma
