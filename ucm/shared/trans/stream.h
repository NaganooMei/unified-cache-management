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
#ifndef UNIFIEDCACHE_TRANS_STREAM_H
#define UNIFIEDCACHE_TRANS_STREAM_H

#include <cstdint>
#include <functional>
#include <vector>
#include "status/status.h"

#ifndef UCM_RUNTIME_ASCEND_IO_AGGREGATION
#define UCM_RUNTIME_ASCEND_IO_AGGREGATION 0
#endif
#ifndef UCM_RUNTIME_ASCEND_SDMA_DIRECT
#define UCM_RUNTIME_ASCEND_SDMA_DIRECT 0
#endif

namespace UC::Trans {

struct StreamOptions {
    int32_t deviceId{-1};
    size_t streamNumber{1};
    std::vector<size_t> tensorSizes{};
    bool cacheIOAggregation{false};
    size_t ioAggregationPipelineDepth{2};
    size_t ioAggregationMaxReadyLanes{8};
    bool cacheSdmaDirect{false};
    size_t sdmaDirectMaxReadyLanes{8};
};

class Stream {
public:
    virtual ~Stream() = default;
    virtual Status Setup() = 0;
    virtual Status Setup(const StreamOptions& options)
    {
        (void)options;
        return Setup();
    }

    virtual Status DeviceToHost(void* device, void* host, size_t size) = 0;
    virtual Status DeviceToHost(void* device[], void* host[], size_t size, size_t number) = 0;
    virtual Status DeviceToHost(void* device[], void* host, size_t size, size_t number) = 0;
    virtual Status DeviceToHostAsync(void* device, void* host, size_t size) = 0;
    virtual Status DeviceToHostAsync(void* device[], void* host[], size_t size, size_t number) = 0;
    virtual Status DeviceToHostAsync(void* device[], void* host, size_t size, size_t number) = 0;

    virtual Status HostToDevice(void* host, void* device, size_t size) = 0;
    virtual Status HostToDevice(void* host[], void* device[], size_t size, size_t number) = 0;
    virtual Status HostToDevice(void* host, void* device[], size_t size, size_t number) = 0;
    virtual Status HostToDeviceAsync(void* host, void* device, size_t size) = 0;
    virtual Status HostToDeviceAsync(void* host[], void* device[], size_t size, size_t number) = 0;
    virtual Status HostToDeviceAsync(void* host, void* device[], size_t size, size_t number) = 0;
    virtual Status HostToDeviceAsync(void* host, void* device[], const std::vector<size_t>& sizes,
                                     void* mappedHost = nullptr)
    {
        (void)mappedHost;
        size_t offset = 0;
        for (size_t i = 0; i < sizes.size(); ++i) {
            auto* pHost = static_cast<void*>(static_cast<int8_t*>(host) + offset);
            auto s = HostToDeviceAsync(pHost, device[i], sizes[i]);
            if (s.Failure()) [[unlikely]] { return s; }
            offset += sizes[i];
        }
        return Status::OK();
    }
    virtual Status HostToDeviceAsync(const std::vector<void*>& hosts,
                                     const std::vector<void*>& mappedHosts,
                                     const std::vector<void**>& devices,
                                     const std::vector<size_t>& sizes)
    {
        if (hosts.size() != devices.size() ||
            (!mappedHosts.empty() && mappedHosts.size() != hosts.size())) {
            return Status::InvalidParam("invalid H2D task copy inputs");
        }
        for (size_t i = 0; i < hosts.size(); ++i) {
            void* mappedHost = mappedHosts.empty() ? nullptr : mappedHosts[i];
            auto s = HostToDeviceAsync(hosts[i], devices[i], sizes, mappedHost);
            if (s.Failure()) [[unlikely]] { return s; }
        }
        return Status::OK();
    }
    virtual Status DeviceToHostAsync(void* device[], void* host, const std::vector<size_t>& sizes,
                                     void* mappedHost = nullptr)
    {
        (void)mappedHost;
        size_t offset = 0;
        for (size_t i = 0; i < sizes.size(); ++i) {
            auto* pHost = static_cast<void*>(static_cast<int8_t*>(host) + offset);
            auto s = DeviceToHostAsync(device[i], pHost, sizes[i]);
            if (s.Failure()) [[unlikely]] { return s; }
            offset += sizes[i];
        }
        return Status::OK();
    }
    virtual Status DeviceToHostAsync(const std::vector<void**>& devices,
                                     const std::vector<void*>& hosts,
                                     const std::vector<void*>& mappedHosts,
                                     const std::vector<size_t>& sizes)
    {
        if (hosts.size() != devices.size() ||
            (!mappedHosts.empty() && mappedHosts.size() != hosts.size())) {
            return Status::InvalidParam("invalid D2H task copy inputs");
        }
        for (size_t i = 0; i < hosts.size(); ++i) {
            void* mappedHost = mappedHosts.empty() ? nullptr : mappedHosts[i];
            auto s = DeviceToHostAsync(devices[i], hosts[i], sizes, mappedHost);
            if (s.Failure()) [[unlikely]] { return s; }
        }
        return Status::OK();
    }

    virtual Status AppendCallback(std::function<void(bool)> cb) = 0;
    virtual Status Synchronized() = 0;
    virtual Status WaitEvent(void* event) = 0;
};

inline Status ValidateStreamOptions(const StreamOptions& options)
{
    if (options.streamNumber == 0) { return Status::InvalidParam("invalid stream number"); }
    if (options.cacheIOAggregation && options.cacheSdmaDirect) {
        return Status::InvalidParam("conflicting Ascend copy options");
    }
    if (options.cacheIOAggregation && !UCM_RUNTIME_ASCEND_IO_AGGREGATION) {
        return Status::InvalidParam("Cache IO aggregation is not compiled");
    }
    if (options.cacheSdmaDirect && !UCM_RUNTIME_ASCEND_SDMA_DIRECT) {
        return Status::InvalidParam("Cache SDMA Direct is not compiled");
    }
    return Status::OK();
}

inline Status ResolveCopyStreamPoolSize(const StreamOptions& options, size_t& streamPoolSize)
{
    auto s = ValidateStreamOptions(options);
    if (s.Failure()) [[unlikely]] { return s; }
    streamPoolSize =
        (options.cacheIOAggregation || options.cacheSdmaDirect) ? 1 : options.streamNumber;
    return Status::OK();
}

}  // namespace UC::Trans

#endif
