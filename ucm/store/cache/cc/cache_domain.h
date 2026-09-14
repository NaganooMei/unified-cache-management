// MIT License
// Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#pragma once

#include <cstdint>
#include <string>

namespace UC::CacheStore {

// Stable across processes, with a bounded length for abstract UNIX socket names.
inline std::string CacheDomainName(const std::string& uniqueId)
{
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char c : uniqueId) {
        hash ^= c;
        hash *= 1099511628211ULL;
    }
    return "ucm_cache_" + std::to_string(hash);
}

}  // namespace UC::CacheStore
