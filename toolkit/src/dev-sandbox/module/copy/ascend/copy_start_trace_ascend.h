/**
 * MIT License
 *
 * Copyright (c) 2026 Mag1c.H
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
#ifndef COPY_START_TRACE_ASCEND_H
#define COPY_START_TRACE_ASCEND_H

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>
#include <time.h>
#include <unistd.h>
#include "error_handle_ascend.h"

inline uint64_t CopyStartMonotonicNs()
{
    timespec time{};
#ifdef CLOCK_MONOTONIC_RAW
    const auto clockId = CLOCK_MONOTONIC_RAW;
#else
    const auto clockId = CLOCK_MONOTONIC;
#endif
    ASSERT(clock_gettime(clockId, &time) == 0);
    return static_cast<uint64_t>(time.tv_sec) * 1000000000ULL +
           static_cast<uint64_t>(time.tv_nsec);
}

inline void EmitCopyStartTrace(const std::string& method, size_t deviceId, size_t iteration,
                               uint64_t barrierEnterNs, uint64_t barrierExitNs,
                               uint64_t wallStartNs, uint64_t releaseSubmitNs,
                               uint64_t syncEnterNs, uint64_t wallEndNs, size_t deviceGateUs,
                               size_t deviceCopyUs,
                               const std::vector<uint64_t>& streamTimestampsUs)
{
    ASSERT(!streamTimestampsUs.empty());
    const auto [minimum, maximum] =
        std::minmax_element(streamTimestampsUs.begin(), streamTimestampsUs.end());

    std::ostringstream output;
    output << "COPY_START_TRACE"
           << " start_gate=event_broadcast"
           << " pid=" << getpid() << " method=" << method << " device=" << deviceId
           << " iteration=" << iteration << " streams=" << streamTimestampsUs.size()
           << " barrier_enter_ns=" << barrierEnterNs << " barrier_exit_ns=" << barrierExitNs
           << " notify_submit_ns=" << releaseSubmitNs
           << " barrier_wait_us=" << (barrierExitNs - barrierEnterNs) / 1000
           << " notify_submit_delay_us=" << (releaseSubmitNs - barrierExitNs) / 1000
           << " wall_start_ns=" << wallStartNs << " release_submit_ns=" << releaseSubmitNs
           << " sync_enter_ns=" << syncEnterNs << " wall_end_ns=" << wallEndNs
           << " device_gate_us=" << deviceGateUs << " device_copy_us=" << deviceCopyUs
           << " stream_start_min_us=" << *minimum << " stream_start_max_us=" << *maximum
           << " stream_start_skew_us=" << (*maximum - *minimum)
           << " stream_timestamps_us=";
    for (size_t i = 0; i < streamTimestampsUs.size(); ++i) {
        if (i > 0) { output << ','; }
        output << streamTimestampsUs[i];
    }
    output << '\n';

    const auto line = output.str();
    const auto written = write(STDERR_FILENO, line.data(), line.size());
    (void)written;
}

#endif  // COPY_START_TRACE_ASCEND_H
