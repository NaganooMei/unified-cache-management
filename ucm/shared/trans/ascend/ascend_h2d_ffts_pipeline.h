/**
 * MIT License
 *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef UNIFIEDCACHE_TRANS_ASCEND_H2D_FFTS_PIPELINE_H
#define UNIFIEDCACHE_TRANS_ASCEND_H2D_FFTS_PIPELINE_H

#include <acl/acl.h>
#include <cstdint>
#include <memory>
#include <vector>
#include "ffts_d2d_dispatcher.h"
#include "status/status.h"

namespace UC::Trans {

struct AscendH2DFftsPipelineConfig {
    int32_t deviceId{-1};
    size_t pipelineDepth{2};
    uint16_t maxReadyLanes{8};
    size_t objectBytes{0};
    size_t maxFragments{0};
};

class AscendH2DFftsPipeline {
    struct InFlightObject {
        std::vector<AscendFftsCopySpec> specs;
        FftsD2DDispatcher dispatcher;
    };

public:
    AscendH2DFftsPipeline() = default;
    ~AscendH2DFftsPipeline();
    AscendH2DFftsPipeline(const AscendH2DFftsPipeline&) = delete;
    AscendH2DFftsPipeline& operator=(const AscendH2DFftsPipeline&) = delete;

    Status Setup(const AscendH2DFftsPipelineConfig& config);
    Status SubmitObject(void* host, void** devices, const std::vector<size_t>& sizes);
    Status Synchronize();

private:
    void Cleanup() noexcept;
    Status BuildCopySpecs(InFlightObject& object, void* staging, void** devices,
                          const std::vector<size_t>& sizes) const;

    bool setup_{false};
    int32_t deviceId_{-1};
    size_t pipelineDepth_{0};
    uint16_t maxReadyLanes_{0};
    size_t objectBytes_{0};
    size_t maxFragments_{0};
    size_t nextObjectIndex_{0};
    aclrtStream h2dStream_{nullptr};
    aclrtStream fftsStream_{nullptr};
    std::vector<void*> stagingBuffers_{};
    std::vector<aclrtEvent> slotReady_{};
    std::vector<aclrtEvent> slotFree_{};
    std::vector<std::unique_ptr<InFlightObject>> inFlight_{};
};

}  // namespace UC::Trans

#endif
