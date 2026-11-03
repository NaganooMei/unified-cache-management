/**
 * MIT License
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
#include "metrics_api.h"
#include "time/now_time.h"

namespace UC::Cache2 {

// Caller-local counters only: never add shared metadata or clocks to the retry loop.
struct BufferGetStats {
    uint64_t retries{0};
    uint64_t peerKeyMiss{0};
    uint64_t peerUnclaimed{0};
    uint64_t peerAllocatedPins{0};
    uint64_t pinRetries{0};
    uint64_t claimedRetries{0};
    uint64_t lockRetries{0};
    uint64_t allocScans{0};
    uint64_t allocFailures{0};
    uint64_t loadingClaims{0};
    uint64_t preallocCreated{0};
    uint64_t preallocExisting{0};
    uint64_t preallocLockFailures{0};
    uint64_t preallocAllocFailures{0};
};

// One dispatcher task owns this object. Publish after dispatch/preallocation,
// including early returns. Keep publication out of individual Get measurements.
class LoadDiagnostics {
public:
    enum class Kind { TargetGet, PeerGet, BackendCall, RunningPush };

    explicit LoadDiagnostics(bool enabled, size_t shards) : enabled_{enabled}
    {
        if (enabled_) { observations_.reserve(3 * shards); }
    }
    LoadDiagnostics(const LoadDiagnostics&) = delete;
    LoadDiagnostics& operator=(const LoadDiagnostics&) = delete;
    ~LoadDiagnostics()
    {
        if (!enabled_) { return; }
        for (const auto& [kind, ms] : observations_) {
            switch (kind) {
                case Kind::TargetGet:
                    Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_get_target_duration_ms"),
                                         ms);
                    break;
                case Kind::PeerGet:
                    Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_get_peer_duration_ms"), ms);
                    break;
                case Kind::BackendCall:
                    Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_backend_call_duration_ms"),
                                         ms);
                    break;
                case Kind::RunningPush:
                    Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_running_push_duration_ms"),
                                         ms);
                    break;
            }
        }
        Publish("cache_load_get_retries_total", stats_.retries);
        Publish("cache_load_get_peer_key_miss_total", stats_.peerKeyMiss);
        Publish("cache_load_get_peer_unclaimed_total", stats_.peerUnclaimed);
        Publish("cache_load_get_peer_allocated_pins_total", stats_.peerAllocatedPins);
        Publish("cache_load_get_pin_retries_total", stats_.pinRetries);
        Publish("cache_load_get_claimed_retries_total", stats_.claimedRetries);
        Publish("cache_load_get_lock_retries_total", stats_.lockRetries);
        Publish("cache_load_get_alloc_scans_total", stats_.allocScans);
        Publish("cache_load_get_alloc_failures_total", stats_.allocFailures);
        Publish("cache_load_get_loading_claims_total", stats_.loadingClaims);
        Publish("cache_load_prealloc_created_total", stats_.preallocCreated);
        Publish("cache_load_prealloc_existing_total", stats_.preallocExisting);
        Publish("cache_load_prealloc_lock_failures_total", stats_.preallocLockFailures);
        Publish("cache_load_prealloc_alloc_failures_total", stats_.preallocAllocFailures);
        // The transfer waiter can finish before the dispatcher reaches Prealloc.
        // Consumers must wait for this marker as well as load completion.
        Metrics::UpdateStats(NAME_TO_METRIC_ID("cache_load_diagnostics_tasks_total"), 1.0);
    }

    BufferGetStats* Stats() { return enabled_ ? &stats_ : nullptr; }
    double Start() const { return enabled_ ? NowTime::Now() : 0.0; }
    void Observe(Kind kind, double start)
    {
        if (enabled_) { observations_.emplace_back(kind, (NowTime::Now() - start) * 1e3); }
    }

private:
    static void Publish(const char* name, uint64_t count)
    {
        if (count != 0) { Metrics::UpdateStats(name, static_cast<double>(count)); }
    }

    bool enabled_;
    BufferGetStats stats_{};
    std::vector<std::pair<Kind, double>> observations_;
};

}  // namespace UC::Cache2
