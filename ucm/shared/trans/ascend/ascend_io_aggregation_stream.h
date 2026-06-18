/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef UNIFIEDCACHE_TRANS_ASCEND_IO_AGGREGATION_STREAM_H
#define UNIFIEDCACHE_TRANS_ASCEND_IO_AGGREGATION_STREAM_H

#include <memory>
#include "trans/stream.h"

namespace UC::Trans {

class AscendShardIOAggregator;

class AscendIoAggregationStream : public Stream {
public:
    AscendIoAggregationStream();
    ~AscendIoAggregationStream() override;
    Status Setup() override;
    Status Setup(const StreamOptions& options) override;

    Status DeviceToHost(void* device, void* host, size_t size) override;
    Status DeviceToHost(void* device[], void* host[], size_t size, size_t number) override;
    Status DeviceToHost(void* device[], void* host, size_t size, size_t number) override;
    Status DeviceToHostAsync(void* device, void* host, size_t size) override;
    Status DeviceToHostAsync(void* device[], void* host[], size_t size, size_t number) override;
    Status DeviceToHostAsync(void* device[], void* host, size_t size, size_t number) override;

    Status HostToDevice(void* host, void* device, size_t size) override;
    Status HostToDevice(void* host[], void* device[], size_t size, size_t number) override;
    Status HostToDevice(void* host, void* device[], size_t size, size_t number) override;
    Status HostToDeviceAsync(void* host, void* device, size_t size) override;
    Status HostToDeviceAsync(void* host[], void* device[], size_t size, size_t number) override;
    Status HostToDeviceAsync(void* host, void* device[], size_t size, size_t number) override;
    Status HostToDeviceAsync(void* host, void* device[], const std::vector<size_t>& sizes,
                             void* mappedHost = nullptr) override;
    Status DeviceToHostAsync(void* device[], void* host, const std::vector<size_t>& sizes,
                             void* mappedHost = nullptr) override;

    Status AppendCallback(std::function<void(bool)> cb) override;
    Status Synchronized() override;
    Status WaitEvent(void* event) override;

private:
    std::unique_ptr<AscendShardIOAggregator> aggregator_{nullptr};
};

}  // namespace UC::Trans

#endif
