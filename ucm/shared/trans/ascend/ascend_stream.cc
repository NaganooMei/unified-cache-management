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
#include "ascend_stream.h"
#include <limits>

#if UCM_RUNTIME_ASCEND_SDMA_DIRECT
#include "ascend_sdma_direct_copier.h"
#endif

namespace UC::Trans {

AscendStream::~AscendStream()
{
    if (cbThread_.joinable()) {
        auto tid = cbThread_.native_handle();
        (void)aclrtUnSubscribeReport(tid, stream_);
        stop_ = true;
        cbThread_.join();
    }
    if (stream_) {
        (void)aclrtDestroyStream(stream_);
        stream_ = nullptr;
    }
}

Status AscendStream::Setup()
{
    auto ret =
        aclrtCreateStreamWithConfig(&stream_, 0, ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC);
    if (ret != ACL_SUCCESS) [[unlikely]] { return Status{ret, std::to_string(ret)}; }
    cbThread_ = std::thread([this] {
        while (!this->stop_) { (void)aclrtProcessReport(10); }
    });
    auto tid = cbThread_.native_handle();
    ret = aclrtSubscribeReport(tid, stream_);
    if (ret != ACL_SUCCESS) [[unlikely]] { return Status{ret, std::to_string(ret)}; }
    return Status::OK();
}

Status AscendStream::Setup(const StreamOptions& options)
{
    ResolvedStreamOptions resolved;
    auto s = ResolveStreamOptions(options, resolved);
    if (s.Failure()) [[unlikely]] { return s; }
#if UCM_RUNTIME_ASCEND_SDMA_DIRECT
    if (resolved.streamOptions.cacheSdmaDirect) {
        if (resolved.streamOptions.tensorSizes.empty()) {
            return Status::InvalidParam("invalid tensor sizes for Cache SDMA Direct");
        }
        if (resolved.streamOptions.sdmaDirectLaunchMode != "shard" &&
            resolved.streamOptions.sdmaDirectLaunchMode != "task") {
            return Status::InvalidParam("invalid Cache SDMA Direct launch mode({})",
                                        resolved.streamOptions.sdmaDirectLaunchMode);
        }
        if (resolved.streamOptions.sdmaDirectMaxReadyLanes == 0 ||
            resolved.streamOptions.sdmaDirectMaxReadyLanes >
                static_cast<size_t>(std::numeric_limits<uint16_t>::max())) {
            return Status::InvalidParam("invalid Cache SDMA Direct max ready lanes({})",
                                        resolved.streamOptions.sdmaDirectMaxReadyLanes);
        }
        AscendSdmaDirectCopyConfig config;
        config.deviceId = resolved.streamOptions.deviceId;
        config.launchMode = resolved.streamOptions.sdmaDirectLaunchMode;
        config.maxReadyLanes =
            static_cast<uint16_t>(resolved.streamOptions.sdmaDirectMaxReadyLanes);
        sdmaDirectCopier_ = std::make_unique<AscendSdmaDirectCopier>();
        s = sdmaDirectCopier_->Setup(config);
        if (s.Failure()) [[unlikely]] {
            sdmaDirectCopier_.reset();
            sdmaDirect_ = false;
            return s;
        }
        sdmaDirect_ = true;
        return Status::OK();
    }
#endif
    return Setup();
}

Status AscendStream::DeviceToHost(void* device, void* host, size_t size)
{
    auto ret = aclrtMemcpy(host, size, device, size, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret == ACL_SUCCESS) { return Status::OK(); }
    return Status{ret, std::to_string(ret)};
}

Status AscendStream::DeviceToHost(void* device[], void* host[], size_t size, size_t number)
{
    auto s = DeviceToHostAsync(device, host, size, number);
    if (s.Failure()) [[unlikely]] { return s; }
    return Synchronized();
}

Status AscendStream::DeviceToHost(void* device[], void* host, size_t size, size_t number)
{
    auto s = DeviceToHostAsync(device, host, size, number);
    if (s.Failure()) [[unlikely]] { return s; }
    return Synchronized();
}

Status AscendStream::DeviceToHostAsync(void* device, void* host, size_t size)
{
    auto ret = aclrtMemcpyAsync(host, size, device, size, ACL_MEMCPY_DEVICE_TO_HOST, stream_);
    if (ret == ACL_SUCCESS) { return Status::OK(); }
    return Status{ret, std::to_string(ret)};
}

Status AscendStream::DeviceToHostAsync(void* device[], void* host[], size_t size, size_t number)
{
    for (size_t i = 0; i < number; i++) {
        auto s = DeviceToHostAsync(device[i], host[i], size);
        if (s.Failure()) [[unlikely]] { return s; }
    }
    return Status::OK();
}

Status AscendStream::DeviceToHostAsync(void* device[], void* host, size_t size, size_t number)
{
    for (size_t i = 0; i < number; i++) {
        auto pHost = (void*)(((int8_t*)host) + size * i);
        auto s = DeviceToHostAsync(device[i], pHost, size);
        if (s.Failure()) [[unlikely]] { return s; }
    }
    return Status::OK();
}

Status AscendStream::HostToDevice(void* host, void* device, size_t size)
{
    auto ret = aclrtMemcpy(device, size, host, size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret == ACL_SUCCESS) { return Status::OK(); }
    return Status{ret, std::to_string(ret)};
}

Status AscendStream::HostToDevice(void* host[], void* device[], size_t size, size_t number)
{
    auto s = HostToDeviceAsync(host, device, size, number);
    if (s.Failure()) [[unlikely]] { return s; }
    return Synchronized();
}

Status AscendStream::HostToDevice(void* host, void* device[], size_t size, size_t number)
{
    auto s = HostToDeviceAsync(host, device, size, number);
    if (s.Failure()) [[unlikely]] { return s; }
    return Synchronized();
}

Status AscendStream::HostToDeviceAsync(void* host, void* device, size_t size)
{
    auto ret = aclrtMemcpyAsync(device, size, host, size, ACL_MEMCPY_HOST_TO_DEVICE, stream_);
    if (ret == ACL_SUCCESS) { return Status::OK(); }
    return Status{ret, std::to_string(ret)};
}

Status AscendStream::HostToDeviceAsync(void* host[], void* device[], size_t size, size_t number)
{
    for (size_t i = 0; i < number; i++) {
        auto s = HostToDeviceAsync(host[i], device[i], size);
        if (s.Failure()) [[unlikely]] { return s; }
    }
    return Status::OK();
}

Status AscendStream::HostToDeviceAsync(void* host, void* device[], size_t size, size_t number)
{
    for (size_t i = 0; i < number; i++) {
        auto pHost = (void*)(((int8_t*)host) + size * i);
        auto s = HostToDeviceAsync(pHost, device[i], size);
        if (s.Failure()) [[unlikely]] { return s; }
    }
    return Status::OK();
}

Status AscendStream::HostToDeviceScatterAsync(void* host, void* hostDevicePtr, void** device,
                                              const std::vector<size_t>& sizes)
{
    (void)host;
#if UCM_RUNTIME_ASCEND_SDMA_DIRECT
    if (sdmaDirect_) { return sdmaDirectCopier_->SubmitLoadObject(hostDevicePtr, device, sizes); }
#endif
    return Stream::HostToDeviceScatterAsync(host, hostDevicePtr, device, sizes);
}

Status AscendStream::DeviceToHostGatherAsync(void** device, void* host, void* hostDevicePtr,
                                             const std::vector<size_t>& sizes)
{
    (void)host;
#if UCM_RUNTIME_ASCEND_SDMA_DIRECT
    if (sdmaDirect_) { return sdmaDirectCopier_->SubmitDumpObject(device, hostDevicePtr, sizes); }
#endif
    return Stream::DeviceToHostGatherAsync(device, host, hostDevicePtr, sizes);
}

using Closure = std::function<void(bool)>;

static void Trampoline(void* data)
{
    auto c = static_cast<Closure*>(data);
    (*c)(true);
    delete c;
}

Status Trans::AscendStream::AppendCallback(std::function<void(bool)> cb)
{
#if UCM_RUNTIME_ASCEND_SDMA_DIRECT
    if (sdmaDirect_) { return Status::InvalidParam("Cache SDMA Direct callback unsupported"); }
#endif
    auto c = new (std::nothrow) Closure{std::move(cb)};
    if (!c) [[unlikely]] { return Status::Error("out of memory for appending callback"); }
    auto ret = aclrtLaunchCallback(Trampoline, (void*)c, ACL_CALLBACK_NO_BLOCK, stream_);
    if (ret != ACL_SUCCESS) [[unlikely]] {
        delete c;
        return Status{ret, std::to_string(ret)};
    }
    return Status::OK();
}

Status AscendStream::Synchronized()
{
#if UCM_RUNTIME_ASCEND_SDMA_DIRECT
    if (sdmaDirect_) { return sdmaDirectCopier_->Synchronize(); }
#endif
    auto ret = aclrtSynchronizeStream(stream_);
    if (ret == ACL_SUCCESS) { return Status::OK(); }
    return Status{ret, std::to_string(ret)};
}

Status AscendStream::WaitEvent(void* event)
{
#if UCM_RUNTIME_ASCEND_SDMA_DIRECT
    if (sdmaDirect_) { return sdmaDirectCopier_->WaitEvent(event); }
#endif
    if (event == nullptr) { return Status::OK(); }
    auto ret = aclrtStreamWaitEvent(stream_, static_cast<aclrtEvent>(event));
    if (ret != ACL_SUCCESS) [[unlikely]] { return Status{ret, std::to_string(ret)}; }
    return Status::OK();
}

}  // namespace UC::Trans
