/**
 * MIT License
 *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
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
 * */
#ifndef UNIFIEDCACHE_CACHE_STORE_CC_TRANS_TASK_H
#define UNIFIEDCACHE_CACHE_STORE_CC_TRANS_TASK_H

#include <atomic>
#include <cstdint>
#include <limits>
#include <string>
#include "status/status.h"
#include "type/types.h"

namespace UC::CacheStore {

enum class LoadDispatchPhase : uint8_t {
    Queued,
    BufferGet,
    BackendSubmit,
    RunningQueue,
    Prealloc,
    Done,
    Failed
};

enum class LoadTransferPhase : uint8_t {
    Idle,
    OwnerBackendWait,
    PeerReadyWait,
    H2dSubmit,
    H2dSync,
    Done,
    Failed
};

inline const char* LoadDispatchPhaseName(LoadDispatchPhase phase)
{
    switch (phase) {
        case LoadDispatchPhase::Queued:
            return "queued";
        case LoadDispatchPhase::BufferGet:
            return "buffer_get";
        case LoadDispatchPhase::BackendSubmit:
            return "backend_submit";
        case LoadDispatchPhase::RunningQueue:
            return "running_queue";
        case LoadDispatchPhase::Prealloc:
            return "prealloc";
        case LoadDispatchPhase::Done:
            return "done";
        case LoadDispatchPhase::Failed:
            return "failed";
    }
    return "unknown";
}

inline const char* LoadTransferPhaseName(LoadTransferPhase phase)
{
    switch (phase) {
        case LoadTransferPhase::Idle:
            return "idle";
        case LoadTransferPhase::OwnerBackendWait:
            return "owner_backend_wait";
        case LoadTransferPhase::PeerReadyWait:
            return "peer_ready_wait";
        case LoadTransferPhase::H2dSubmit:
            return "h2d_submit";
        case LoadTransferPhase::H2dSync:
            return "h2d_sync";
        case LoadTransferPhase::Done:
            return "done";
        case LoadTransferPhase::Failed:
            return "failed";
    }
    return "unknown";
}

class TransTask {
public:
    enum class Type : uint8_t { LOAD, DUMP };
    static constexpr size_t kDiagnosticInvalidIndex = std::numeric_limits<size_t>::max();

public:
    Detail::TaskHandle id{0};
    Type type{Type::DUMP};
    Detail::TaskDesc desc;
    std::atomic<int32_t> failureStatus{Status::OK().Underlying()};
    std::atomic<LoadDispatchPhase> loadDispatchPhase{LoadDispatchPhase::Queued};
    std::atomic<LoadTransferPhase> loadTransferPhase{LoadTransferPhase::Idle};
    std::atomic<size_t> dispatchedShards{0};
    std::atomic<size_t> transferredShards{0};
    std::atomic<size_t> pendingOwnerShards{0};
    std::atomic<size_t> dispatchOriginalIndex{kDiagnosticInvalidIndex};
    std::atomic<size_t> dispatchShardIndex{kDiagnosticInvalidIndex};
    std::atomic<size_t> dispatchPreferredSegment{kDiagnosticInvalidIndex};
    std::atomic<size_t> dispatchBlockHash{0};
    std::atomic<size_t> transferOriginalIndex{kDiagnosticInvalidIndex};
    std::atomic<size_t> transferShardIndex{kDiagnosticInvalidIndex};
    std::atomic<size_t> transferGlobalSlot{kDiagnosticInvalidIndex};
    std::atomic<size_t> transferSegment{kDiagnosticInvalidIndex};
    std::atomic<size_t> transferBlockHash{0};
    std::atomic<Detail::TaskHandle> transferBackendTask{0};
    std::atomic<int32_t> transferSlotState{-1};

public:
    TransTask(Type type, Detail::TaskDesc desc) : id{NextId()}, type{type}, desc{std::move(desc)} {}
    TransTask(TransTask&& other) noexcept
        : id{other.id},
          type{other.type},
          desc{std::move(other.desc)},
          failureStatus{other.failureStatus.load()},
          loadDispatchPhase{other.loadDispatchPhase.load()},
          loadTransferPhase{other.loadTransferPhase.load()},
          dispatchedShards{other.dispatchedShards.load()},
          transferredShards{other.transferredShards.load()},
          pendingOwnerShards{other.pendingOwnerShards.load()},
          dispatchOriginalIndex{other.dispatchOriginalIndex.load()},
          dispatchShardIndex{other.dispatchShardIndex.load()},
          dispatchPreferredSegment{other.dispatchPreferredSegment.load()},
          dispatchBlockHash{other.dispatchBlockHash.load()},
          transferOriginalIndex{other.transferOriginalIndex.load()},
          transferShardIndex{other.transferShardIndex.load()},
          transferGlobalSlot{other.transferGlobalSlot.load()},
          transferSegment{other.transferSegment.load()},
          transferBlockHash{other.transferBlockHash.load()},
          transferBackendTask{other.transferBackendTask.load()},
          transferSlotState{other.transferSlotState.load()}
    {
    }
    void Fail(const Status& status)
    {
        auto expected = Status::OK().Underlying();
        failureStatus.compare_exchange_strong(expected, status.Underlying());
    }
    Status FailureStatus() const
    {
        const auto status = failureStatus.load();
        return status == Status::OK().Underlying() ? Status::Error() : Status{status, {}};
    }

private:
    static size_t NextId() noexcept
    {
        static std::atomic<size_t> id{1};
        return id.fetch_add(1, std::memory_order_relaxed);
    };
};

}  // namespace UC::CacheStore

#endif
