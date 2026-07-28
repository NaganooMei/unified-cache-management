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
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sys/syscall.h>
#include <unistd.h>
#include "trans_queue.h"
#include "logger/logger.h"
#include "metrics_api.h"
#include "posix_file.h"

namespace {

constexpr const char* S2H_TRACE_ENABLE_ENV = "UCM_CACHE_POSIX_S2H_TRACE";
constexpr const char* S2H_TRACE_EPOCH_ENV = "UCM_CACHE_POSIX_S2H_TRACE_EPOCH";
constexpr const char* S2H_TRACE_WORKER_ENV = "UCM_CACHE_POSIX_S2H_TRACE_WORKER";

bool S2hTraceEnabled()
{
    const auto* enabled = std::getenv(S2H_TRACE_ENABLE_ENV);
    return enabled != nullptr && std::strcmp(enabled, "1") == 0;
}

long S2hTraceValue(const char* name)
{
    const auto* value = std::getenv(name);
    return value == nullptr ? -1 : std::strtol(value, nullptr, 10);
}

std::uint64_t MonotonicNs()
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

}  // namespace

namespace UC::PosixStore {

Status TransQueue::Setup(const Config& config, TaskIdSet* failureSet, const SpaceLayout* layout)
{
    failureSet_ = failureSet;
    layout_ = layout;
    ioSize_ = config.tensorSize;
    shardSize_ = config.shardSize;
    nShardPerBlock_ = config.blockSize / config.shardSize;
    ioDirect_ = config.ioDirect;
    timeoutMs_ = config.timeoutMs;
    auto success =
        loadPool_.SetNWorker(config.dataTransConcurrency)
            .SetWorkerFn([this](auto& ios, auto&) { LoadWorker(ios); })
            .SetWorkerTimeoutFn([this](IoUnit& ios, ssize_t tid) { OnIoUnitTimeout(ios); },
                                config.timeoutMs)
            .SetCpuAffinity(config.cpuAffinityCores)
            .Run();
    if (!success) [[unlikely]] {
        return Status::Error(fmt::format("workers({}) start failed", config.dataTransConcurrency));
    }
    success = dumpPool_.SetNWorker(config.dataTransConcurrency)
                  .SetWorkerFn([this](auto& ios, auto&) { DumpWorker(ios); })
                  .SetWorkerTimeoutFn([this](IoUnit& ios, ssize_t tid) { OnIoUnitTimeout(ios); },
                                      config.timeoutMs)
                  .SetCpuAffinity(config.cpuAffinityCores)
                  .Run();
    if (!success) [[unlikely]] {
        return Status::Error(fmt::format("workers({}) start failed", config.dataTransConcurrency));
    }
    return Status::OK();
}

void TransQueue::OnIoUnitTimeout(IoUnit& ios)
{
    UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("posix_io_timeout_total"), 1.0);
    ios.task->Fail(Status::Timeout());
    if (!failureSet_->Contains(ios.task->id)) { failureSet_->Insert(ios.task->id); }
    ios.waiter->Done();
}

void TransQueue::Push(TaskPtr task, WaiterPtr waiter)
{
    waiter->Set(task->desc.size());
    std::list<IoUnit> ios;
    for (auto&& shard : task->desc) { ios.emplace_back<IoUnit>({task, std::move(shard), waiter}); }
    ios.front().firstIo = true;
    if (task->type == TransTask::Type::DUMP) {
        dumpPool_.Push(ios);
    } else {
        loadPool_.Push(ios);
    }
}

void TransQueue::Cancel(TaskPtr task)
{
    auto& pool = task->type == TransTask::Type::DUMP ? dumpPool_ : loadPool_;
    const auto tid = task->id;
    pool.TraverseWaitQueue(
        [this, tid](IoUnit& ios) {
            return ios.task->id == tid || ios.waiter->IsTimeout(timeoutMs_);
        },
        [this](IoUnit& ios) { OnIoUnitTimeout(ios); },
        [this, tid](IoUnit& ios) {
            return ios.task->id > tid && !ios.waiter->IsTimeout(timeoutMs_);
        });
}

void TransQueue::LoadWorker(IoUnit& ios)
{
    if (ios.firstIo) {
        auto wait = NowTime::Now() - ios.waiter->startTp;
        UC_DEBUG("Posix load task({}) start running, wait {:.3f}ms.", ios.task->id, wait * 1e3);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("posix_load_queue_wait_duration_ms"),
                                 wait * 1e3);
    }
    if (failureSet_->Contains(ios.task->id)) {
        ios.waiter->Done();
        return;
    }
    auto s = S2H(ios);
    if (s.Failure()) [[unlikely]] {
        ios.task->Fail(s);
        failureSet_->Insert(ios.task->id);
    }
    ios.waiter->Done();
}

void TransQueue::DumpWorker(IoUnit& ios)
{
    if (ios.firstIo) {
        auto wait = NowTime::Now() - ios.waiter->startTp;
        UC_DEBUG("Posix dump task({}) start running, wait {:.3f}ms.", ios.task->id, wait * 1e3);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("posix_dump_queue_wait_duration_ms"),
                                 wait * 1e3);
    }
    if (failureSet_->Contains(ios.task->id)) {
        ios.waiter->Done();
        return;
    }
    auto s = H2S(ios);
    if (ios.shard.index + 1 == nShardPerBlock_) {
        layout_->CommitFile(ios.shard.owner, s.Success());
    }
    if (s.Failure()) [[unlikely]] {
        ios.task->Fail(s);
        failureSet_->Insert(ios.task->id);
    }
    ios.waiter->Done();
}

Status TransQueue::H2S(IoUnit& ios)
{
    const auto& path = layout_->DataFilePath(ios.shard.owner, true);
    PosixFile file{path};
    auto flags = PosixFile::OpenFlag::CREATE | PosixFile::OpenFlag::WRITE_ONLY;
    if (ioDirect_) { flags |= PosixFile::OpenFlag::DIRECT; }
    auto s = file.Open(flags);
    if (s.Failure()) [[unlikely]] {
        UC_ERROR("Failed({}) to open file({}) with flags({}).", s, path, flags);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("posix_open_errors_total"), 1.0);
        return s;
    }
    auto offset = shardSize_ * ios.shard.index;
    for (const auto& addr : ios.shard.addrs) {
        s = file.Write(addr, ioSize_, offset);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to write file({}:{}).", s, path, offset);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("posix_io_errors_total"), 1.0);
            return s;
        }
        offset += ioSize_;
    }
    return Status::OK();
}

Status TransQueue::S2H(IoUnit& ios)
{
    const auto& path = layout_->DataFilePath(ios.shard.owner, false);
    PosixFile file{path};
    auto flags = PosixFile::OpenFlag::READ_ONLY;
    if (ioDirect_) { flags |= PosixFile::OpenFlag::DIRECT; }
    auto s = file.Open(flags);
    if (s.Failure()) [[unlikely]] {
        UC_ERROR("Failed({}) to open file({}) with flags({}).", s, path, flags);
        UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("posix_open_errors_total"), 1.0);
        return s;
    }
    auto offset = shardSize_ * ios.shard.index;
    size_t ioIndex = 0;
    for (const auto& addr : ios.shard.addrs) {
        const auto traceS2h = S2hTraceEnabled();
        size_t activeAtStart = 0;
        std::uint64_t startNs = 0;
        if (traceS2h) {
            activeAtStart = activeS2h_.fetch_add(1, std::memory_order_relaxed) + 1;
            startNs = MonotonicNs();
        }
        s = file.Read(addr, ioSize_, offset);
        if (traceS2h) {
            const auto endNs = MonotonicNs();
            const auto activeAfter = activeS2h_.fetch_sub(1, std::memory_order_relaxed) - 1;
            UC_INFO_UNLIMITED(
                "[S2H_TRACE] event=pread epoch={} worker={} pid={} tid={} posix_task={} "
                "shard={} io_index={} offset={} size={} start_ns={} end_ns={} duration_ns={} "
                "active_at_start={} active_after={} status={}",
                S2hTraceValue(S2H_TRACE_EPOCH_ENV), S2hTraceValue(S2H_TRACE_WORKER_ENV),
                getpid(), syscall(SYS_gettid), ios.task->id, ios.shard.index, ioIndex, offset,
                ioSize_, startNs, endNs, endNs - startNs, activeAtStart, activeAfter,
                s.Success() ? "ok" : "error");
        }
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to read file({}:{}).", s, path, offset);
            UC::Metrics::UpdateStats(NAME_TO_METRIC_ID("posix_io_errors_total"), 1.0);
            return s;
        }
        offset += ioSize_;
        ioIndex++;
    }
    return Status::OK();
}

}  // namespace UC::PosixStore
