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
 * */
#pragma once

#include <array>
#include <vector>
#include "cache_buffer.h"

namespace UC::CacheStore {

class PrefetchQueue {
    std::atomic<bool> stop_{false};
    Buffer* buffer_{nullptr};
    StoreV1* backend_{nullptr};
    std::thread prefetchThread_;
    size_t batchSize_{32};
    bool enabled_{false};

public:
    ~PrefetchQueue()
    {
        stop_.store(true, std::memory_order_relaxed);
        if (prefetchThread_.joinable()) { prefetchThread_.join(); }
    };
    // startWorker=false allows a single external consumer to drive PollOnce in tests.
    Status Setup(const Config& config, Buffer* buffer, bool startWorker = true)
    {
        buffer_ = buffer;
        backend_ = config.storeBackend;
        enabled_ = config.cachePrefetchEnable && config.shareBufferEnable &&
                   !config.cacheLoadBackendOnly && buffer && buffer->MyRank() != kInvalidIndex;
        if (!enabled_) { return Status::OK(); }
        if (!backend_ || config.cachePrefetchBatchSize == 0 || config.cachePrefetchBatchSize > 64) {
            return Status::InvalidParam("invalid cache prefetch backend or batch size");
        }
        batchSize_ = config.cachePrefetchBatchSize;
        if (startWorker) {
            prefetchThread_ = std::thread([this] { PrefetchLoop(); });
        }
        return Status::OK();
    }

    size_t PollOnce()
    {
        if (!enabled_ || stop_.load(std::memory_order_relaxed) || buffer_->HasDemand()) {
            return 0;
        }
        std::array<Detail::BlockId, 64> batch;
        auto n = buffer_->DrainPrefetch(buffer_->MyRank(), batch.data(), batchSize_);
        if (n != 0) { PrefetchBatch(batch.data(), n); }
        return n;
    }

private:
    /* Prefetch executor: drains this rank's command ring and loads the first shard of
     * each requested block from the backend into the cache. Sequential on purpose —
     * prefetch is a background hint; integration point for the future Dispatch engine. */
    void PrefetchLoop()
    {
        constexpr auto idle = std::chrono::milliseconds(1);
        while (!stop_.load(std::memory_order_relaxed)) {
            if (PollOnce() == 0) { std::this_thread::sleep_for(idle); }
        }
    }

    /* One backend Load per drained batch. Failure is batch-granular (Wait reports a
     * single status): on failure the whole batch is marked Failed and healed by the
     * next Get/prefetch that takes ownership. */
    void PrefetchBatch(const Detail::BlockId* blocks, size_t num)
    {
        Detail::TaskDesc task;
        std::vector<Buffer::Handle> handles;
        handles.reserve(num);
        for (size_t i = 0; i < num; i++) {
            if (stop_.load(std::memory_order_relaxed) || buffer_->HasDemand()) { break; }
            /* First shard only; allowReserved=false keeps the load-exclusive region
             * for real loads. A non-owner handle means the block is cached or being
             * loaded by any rank — ensure-cached semantics, nothing to do. */
            auto h = buffer_->TryGet(blocks[i], 0, false);
            if (!h || !h.Owner()) { continue; }
            if (h.Data() == nullptr) {
                h.MarkFailed();
                continue;
            }
            Detail::Shard shard;
            shard.owner = blocks[i];
            shard.index = 0;
            shard.addrs.push_back(h.Data());
            task.push_back(std::move(shard));
            handles.emplace_back(std::move(h));
        }
        // Admission is best-effort. A batch already submitted below must be drained,
        // even if demand arrives or shutdown begins, before its Handles can be released.
        if (task.empty() || buffer_->HasDemand() || stop_.load(std::memory_order_relaxed)) {
            return;
        }
        auto r = backend_->Load(std::move(task));
        auto ok = r.HasValue() && backend_->Wait(r.Value()).Success();
        for (auto& h : handles) {
            if (ok) {
                h.MarkReady();
            } else {
                h.MarkFailed();
            }
        }
    }
};

}  // namespace UC::CacheStore
