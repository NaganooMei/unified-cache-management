/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once

#include <cerrno>
#include <cstring>
#include <fstream>
#include <linux/mempolicy.h>
#include <map>
#include <sys/syscall.h>
#include <unistd.h>
#include "logger/logger.h"
#include "shm_numa_layout.h"
#include "status/status.h"

namespace UC::CacheStore::ShmNuma {

inline void CheckSystemCall(long result, const std::string& operation)
{
    if (result < 0) {
        const auto error = errno;
        throw std::runtime_error(operation + ": " + std::strerror(error) +
                                 " (errno=" + std::to_string(error) + ")");
    }
}

inline size_t PageSize()
{
    const auto size = sysconf(_SC_PAGESIZE);
    if (size <= 0) { throw std::runtime_error("cannot determine the system page size"); }
    return static_cast<size_t>(size);
}

inline std::vector<size_t> AllowedNodes()
{
    std::ifstream status("/proc/self/status");
    std::string line;
    const std::string prefix = "Mems_allowed_list:";
    while (std::getline(status, line)) {
        if (line.compare(0, prefix.size(), prefix) != 0) { continue; }
        const auto start = line.find_first_not_of(" \t", prefix.size());
        if (start == std::string::npos) { break; }
        return ParseNodes(line.substr(start));
    }
    throw std::runtime_error("cannot read Mems_allowed_list from /proc/self/status");
}

inline std::vector<size_t> DefaultNodes()
{
    std::ifstream memoryNodes("/sys/devices/system/node/has_memory");
    std::string text;
    if (!(memoryNodes >> text)) {
        throw std::runtime_error(
            "cannot read /sys/devices/system/node/has_memory; "
            "set share_buffer_numa_nodes explicitly");
    }
    return SelectMemoryNodes(ParseNodes(text), AllowedNodes());
}

inline void ValidateAllowedNodes(const std::vector<size_t>& nodes)
{
    const auto allowed = AllowedNodes();
    for (const auto node : nodes) {
        if (std::find(allowed.begin(), allowed.end(), node) == allowed.end()) {
            throw std::runtime_error("NUMA node " + std::to_string(node) +
                                     " is outside Mems_allowed_list");
        }
    }
}

// Bind before first touch. MAP_POPULATE must not be used for the mapping.
inline void BindBeforeTouch(void* base, const std::vector<Range>& ranges, const std::string& name)
{
    const auto pageSize = PageSize();
    if (reinterpret_cast<size_t>(base) % pageSize != 0) {
        throw std::invalid_argument("SHM address is not page-aligned");
    }
    for (const auto& range : ranges) {
        ValidateAllowedNodes({range.node});
        const auto mask = SingleNodeMask(range.node);
        auto* address = static_cast<char*>(base) + range.offset;
        CheckSystemCall(syscall(SYS_mbind, address, range.bytes, MPOL_BIND | MPOL_F_STATIC_NODES,
                                mask.words.data(), mask.maxNode, 0UL),
                        "mbind node=" + std::to_string(range.node) + " shm=" + name);
        UC_INFO_UNLIMITED("SHM NUMA bind: file={} offset={} bytes={} node={}.", name, range.offset,
                          range.bytes, range.node);
    }
}

inline void Verify(void* base, const std::vector<Range>& ranges, const std::string& name)
{
    const auto pageSize = PageSize();
    constexpr size_t batchSize = 4096;
    std::vector<void*> addresses(batchSize);
    std::vector<int> status(batchSize);
    for (const auto& range : ranges) {
        std::map<int, size_t> counts;
        std::map<int, size_t> queryErrors;
        size_t mismatches = 0;
        const auto pageCount = range.bytes / pageSize;
        for (size_t first = 0; first < pageCount;) {
            const auto count = std::min(batchSize, pageCount - first);
            for (size_t i = 0; i < count; ++i) {
                addresses[i] = static_cast<char*>(base) + range.offset + (first + i) * pageSize;
            }
            std::fill(status.begin(), status.end(), -EIO);
            const auto result =
                syscall(SYS_move_pages, 0, count, addresses.data(), nullptr, status.data(), 0);
            if (result < 0) {
                const auto error = errno;
                UC_WARN_UNLIMITED(
                    "SHM NUMA verification is unavailable: file={} offset={} expectedNode={} "
                    "error={} (errno={}); continuing because NUMA verification is best-effort.",
                    name, range.offset, range.node, std::strerror(error), error);
                return;
            }
            for (size_t i = 0; i < count; ++i) {
                if (status[i] < 0) {
                    ++queryErrors[-status[i]];
                    continue;
                }
                ++counts[status[i]];
                if (static_cast<size_t>(status[i]) != range.node) { ++mismatches; }
            }
            first += count;
        }
        for (const auto& [node, pages] : counts) {
            UC_INFO_UNLIMITED(
                "SHM NUMA verify: file={} offset={} expectedNode={} actualNode={} pages={} "
                "bytes={} mismatches={}.",
                name, range.offset, range.node, node, pages, pages * pageSize, mismatches);
        }
        for (const auto& [error, pages] : queryErrors) {
            UC_WARN_UNLIMITED(
                "SHM NUMA page verification was incomplete: file={} offset={} expectedNode={} "
                "error={} (errno={}) pages={}; continuing because NUMA verification is "
                "best-effort.",
                name, range.offset, range.node, std::strerror(error), error, pages);
        }
        if (mismatches != 0) {
            UC_WARN_UNLIMITED(
                "SHM NUMA placement differs from the requested node: file={} offset={} "
                "expectedNode={} mismatches={}; continuing with potentially reduced "
                "performance.",
                name, range.offset, range.node, mismatches);
        }
    }
}

// Only the segment creator calls this, before publishing rank readiness.
inline Status Initialize(void* data, size_t bytes, const std::vector<size_t>& nodes,
                         const std::string& name)
{
    try {
        const auto ranges = Plan(bytes, PageSize(), nodes);
        if (ranges.empty()) { return Status::InvalidParam("empty SHM NUMA nodes"); }
        ValidateAllowedNodes(nodes);
        BindBeforeTouch(data, ranges, name);
        std::memset(data, 0, bytes);
        Verify(data, ranges, name);
        return Status::OK();
    } catch (const std::exception& error) {
        return Status::Error(error.what());
    }
}

}  // namespace UC::CacheStore::ShmNuma
