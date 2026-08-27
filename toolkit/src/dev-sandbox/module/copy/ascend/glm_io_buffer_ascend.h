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
#ifndef GLM_IO_BUFFER_ASCEND_H
#define GLM_IO_BUFFER_ASCEND_H

#include <acl/acl.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include <utility>
#include "copy_buffer.h"
#include "error_handle.h"
#include "error_handle_ascend.h"

class GlmIoCopyBuffer : public CopyBuffer {
public:
    static constexpr std::array<size_t, 3> kIoSizes{128ull * 1024ull, 16ull * 1024ull,
                                                    32ull * 1024ull};
    static constexpr std::array<size_t, 3> kIoOffsets{0, kIoSizes[0],
                                                      kIoSizes[0] + kIoSizes[1]};
    static constexpr size_t kTaskBytes = kIoSizes[0] + kIoSizes[1] + kIoSizes[2];

    GlmIoCopyBuffer(size_t device, size_t taskCount)
        : CopyBuffer{device, kTaskBytes, taskCount}
    {
        ASSERT(taskCount > 0);
        ASSERT(taskCount <= std::numeric_limits<size_t>::max() / kTaskBytes);
    }

    static constexpr size_t IoCount() { return kIoSizes.size(); }
    static constexpr size_t IoSize(size_t ioIndex) { return kIoSizes[ioIndex]; }
    size_t TotalBytes() const { return kTaskBytes * number_; }

    void* IoAt(size_t taskIndex, size_t ioIndex) const
    {
        ASSERT(taskIndex < number_);
        ASSERT(ioIndex < IoCount());
        return static_cast<void*>(static_cast<char*>(addr_) + taskIndex * kTaskBytes +
                                  kIoOffsets[ioIndex]);
    }
};

class GlmSharedHostRegion : public GlmIoCopyBuffer {
public:
    GlmSharedHostRegion(std::string tag, size_t taskCount)
        : GlmIoCopyBuffer{0, taskCount}
    {
        shmName_ = "/copy_ascend_glm_" + std::to_string(getpid()) + "_" + tag + "_" +
                   std::to_string(reinterpret_cast<std::uintptr_t>(this));
        const auto total = TotalBytes();
        const auto fd = shm_open(shmName_.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        ASSERT(fd != -1);
        ASSERT(ftruncate(fd, total) == 0);
        constexpr auto prot = PROT_READ | PROT_WRITE;
        constexpr auto flags = MAP_SHARED | MAP_POPULATE;
        addr_ = mmap(nullptr, total, prot, flags, fd, 0);
        const auto closeStatus = close(fd);
        ASSERT(closeStatus == 0);
        ASSERT(addr_ != MAP_FAILED);
        std::memset(addr_, 'g', total);
    }

    ~GlmSharedHostRegion() override
    {
        if (addr_ != nullptr && addr_ != MAP_FAILED) {
            munmap(addr_, TotalBytes());
            addr_ = nullptr;
        }
        if (!shmName_.empty()) { shm_unlink(shmName_.c_str()); }
    }

    const std::string& ShmName() const { return shmName_; }
    std::string Name() const override { return "acl::shm::glm"; }

private:
    std::string shmName_;
};

class GlmSharedHostCopyBuffer : public GlmIoCopyBuffer {
public:
    GlmSharedHostCopyBuffer(std::string shmName, size_t device, size_t taskCount)
        : GlmIoCopyBuffer{device, taskCount}, shmName_{std::move(shmName)}
    {
        const auto total = TotalBytes();
        const auto fd = shm_open(shmName_.c_str(), O_RDWR, 0600);
        ASSERT(fd != -1);
        constexpr auto prot = PROT_READ | PROT_WRITE;
        constexpr auto flags = MAP_SHARED | MAP_POPULATE;
        addr_ = mmap(nullptr, total, prot, flags, fd, 0);
        const auto closeStatus = close(fd);
        ASSERT(closeStatus == 0);
        ASSERT(addr_ != MAP_FAILED);

        ASCEND_ASSERT(aclrtSetDevice(device_));
        ASCEND_ASSERT(
            aclrtHostRegisterV2(addr_, total, ACL_HOST_REG_MAPPED | ACL_HOST_REG_PINNED));
        registered_ = true;
        ASCEND_ASSERT(aclrtHostGetDevicePointer(addr_, &mappedAddr_, 0));
    }

    ~GlmSharedHostCopyBuffer() override
    {
        if (addr_ != nullptr && addr_ != MAP_FAILED) {
            ASCEND_ASSERT(aclrtSetDevice(device_));
            if (registered_) { ASCEND_ASSERT(aclrtHostUnregister(addr_)); }
            munmap(addr_, TotalBytes());
            addr_ = nullptr;
        }
    }

    void* MappedIoAt(size_t taskIndex, size_t ioIndex) const
    {
        ASSERT(taskIndex < number_);
        ASSERT(ioIndex < IoCount());
        return static_cast<void*>(static_cast<char*>(mappedAddr_) + taskIndex * kTaskBytes +
                                  kIoOffsets[ioIndex]);
    }

    std::string Name() const override { return "acl::shm::glm"; }

private:
    std::string shmName_;
    void* mappedAddr_ = nullptr;
    bool registered_ = false;
};

class GlmDeviceCopyBuffer : public GlmIoCopyBuffer {
public:
    GlmDeviceCopyBuffer(size_t device, size_t taskCount) : GlmIoCopyBuffer{device, taskCount}
    {
        const auto total = TotalBytes();
        ASCEND_ASSERT(aclrtSetDevice(device_));
        ASCEND_ASSERT(aclrtMalloc(&addr_, total, ACL_MEM_MALLOC_HUGE_FIRST));
        ASCEND_ASSERT(aclrtMemset(addr_, total, 0, total));
    }

    ~GlmDeviceCopyBuffer() override
    {
        if (addr_ != nullptr) {
            ASCEND_ASSERT(aclrtSetDevice(device_));
            ASCEND_ASSERT(aclrtFree(addr_));
            addr_ = nullptr;
        }
    }

    std::string Name() const override { return "acl::device::" + std::to_string(device_); }
};

#endif  // GLM_IO_BUFFER_ASCEND_H
