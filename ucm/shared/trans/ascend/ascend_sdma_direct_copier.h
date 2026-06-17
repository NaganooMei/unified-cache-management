/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef UNIFIEDCACHE_TRANS_ASCEND_SDMA_DIRECT_COPIER_H
#define UNIFIEDCACHE_TRANS_ASCEND_SDMA_DIRECT_COPIER_H

#include <acl/acl.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "ffts_sdma_dispatcher.h"
#include "status/status.h"

namespace UC::Trans {

struct AscendSdmaDirectCopyConfig {
    int32_t deviceId{-1};
    std::string launchMode{"shard"};
    uint16_t maxReadyLanes{8};
};

class AscendSdmaDirectCopier {
    enum class LaunchMode { SHARD, TASK };
    struct InFlightObject {
        std::vector<AscendFftsCopySpec> specs;
        FftsSdmaDispatcher dispatcher;
    };

public:
    AscendSdmaDirectCopier() = default;
    ~AscendSdmaDirectCopier();
    AscendSdmaDirectCopier(const AscendSdmaDirectCopier&) = delete;
    AscendSdmaDirectCopier& operator=(const AscendSdmaDirectCopier&) = delete;

    Status Setup(const AscendSdmaDirectCopyConfig& config);
    Status WaitEvent(void* event);
    Status SubmitLoadObject(const void* hostDevicePtr, void** devices,
                            const std::vector<size_t>& sizes);
    Status SubmitDumpObject(void** devices, void* hostDevicePtr, const std::vector<size_t>& sizes);
    Status Synchronize();

private:
    void Cleanup() noexcept;
    Status BuildHostToDeviceSpecs(const void* hostDevicePtr, void** devices,
                                  const std::vector<size_t>& sizes,
                                  std::vector<AscendFftsCopySpec>& specs) const;
    Status BuildDeviceToHostSpecs(void** devices, void* hostDevicePtr,
                                  const std::vector<size_t>& sizes,
                                  std::vector<AscendFftsCopySpec>& specs) const;
    Status LaunchSpecs(std::vector<AscendFftsCopySpec>&& specs);
    Status LaunchPendingTask();
    static Status AclStatus(aclError ret, const char* expr);

    int32_t deviceId_{-1};
    aclrtStream fftsStream_{nullptr};
    uint16_t maxReadyLanes_{8};
    LaunchMode launchMode_{LaunchMode::SHARD};
    std::vector<AscendFftsCopySpec> pendingSpecs_{};
    std::vector<std::unique_ptr<InFlightObject>> inFlight_{};
    bool setup_{false};
};

}  // namespace UC::Trans

#endif
