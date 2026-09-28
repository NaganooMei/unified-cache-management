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
#include "data_backend_memfd.h"

namespace UC::Cache2 {

namespace {
constexpr const char* kNotImplemented = "cache2 memfd backend: not implemented";
}  // namespace

MemfdDataBackend::MemfdDataBackend(const std::string& domainId) : domainId_(domainId) {}

MemfdDataBackend::~MemfdDataBackend() { Reset(); }

Status MemfdDataBackend::Setup(const BackendOptions& options)
{
    (void)options;
    return Status::Error(kNotImplemented);
}

Status MemfdDataBackend::BindLocal(size_t rank)
{
    (void)rank;
    return Status::Error(kNotImplemented);
}

Status MemfdDataBackend::ExportLocal(uint64_t* handle)
{
    (void)handle;
    return Status::Error(kNotImplemented);
}

Status MemfdDataBackend::ImportPeer(size_t rank, uint64_t handle)
{
    (void)rank;
    (void)handle;
    return Status::Error(kNotImplemented);
}

void MemfdDataBackend::FinalizeSetup() {}

void* MemfdDataBackend::HostAddrOf(size_t rank) const
{
    (void)rank;
    return nullptr;
}

void* MemfdDataBackend::DeviceAddrOf(size_t rank) const
{
    (void)rank;
    return nullptr;
}

void* MemfdDataBackend::HostMappedDeviceAddrOf(size_t rank) const
{
    (void)rank;
    return nullptr;
}

void MemfdDataBackend::Reset() { StopFdServer(); }

Status MemfdDataBackend::CreateLocalSegment(size_t rank)
{
    (void)rank;
    return Status::Error(kNotImplemented);
}

Status MemfdDataBackend::PlaceAndRegister(size_t rank)
{
    (void)rank;
    return Status::Error(kNotImplemented);
}

Status MemfdDataBackend::StartFdServer() { return Status::Error(kNotImplemented); }

void MemfdDataBackend::StopFdServer() {}

std::string MemfdDataBackend::SocketName(size_t rank) const
{
    (void)rank;
    return {};
}

}  // namespace UC::Cache2
