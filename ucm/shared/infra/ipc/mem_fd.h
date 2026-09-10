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
 */
#pragma once

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <cstddef>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <unistd.h>
#include "status/status.h"

namespace UC {

class MemFd {
    int32_t fd_{-1};
    void* addr_{nullptr};
    size_t size_{0};

public:
    MemFd() = default;
    ~MemFd() { Unmap(); }
    MemFd(const MemFd&) = delete;
    MemFd& operator=(const MemFd&) = delete;

    Status Create(const std::string& name, size_t size, bool seal)
    {
        fd_ = ::memfd_create(name.c_str(), MFD_ALLOW_SEALING);
        if (fd_ < 0) { return Status::OsApiError("memfd_create failed"); }
        if (::ftruncate(fd_, static_cast<off_t>(size)) != 0) {
            return Status::OsApiError("ftruncate failed");
        }
        if (seal) {
            if (::fcntl(fd_, F_ADD_SEALS, F_SEAL_SEAL | F_SEAL_SHRINK) != 0) {
                return Status::OsApiError("F_ADD_SEALS failed");
            }
        }
        addr_ = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (addr_ == MAP_FAILED) {
            addr_ = nullptr;
            return Status::OsApiError("mmap failed");
        }
        size_ = size;
        return Status::OK();
    }

    void* Addr() const { return addr_; }
    int32_t Fd() const { return fd_; }
    size_t Size() const { return size_; }

    Status Adopt(int32_t fd, size_t size)
    {
        addr_ = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (addr_ == MAP_FAILED) {
            addr_ = nullptr;
            return Status::OsApiError("mmap adopt failed");
        }
        fd_ = fd;
        size_ = size;
        return Status::OK();
    }

    Status Remap(size_t size)
    {
        if (addr_ != nullptr) {
            ::munmap(addr_, size_);
            addr_ = nullptr;
            size_ = 0;
        }
        addr_ = ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
        if (addr_ == MAP_FAILED) {
            addr_ = nullptr;
            return Status::OsApiError("mmap remap failed");
        }
        size_ = size;
        return Status::OK();
    }

    void Unmap()
    {
        if (addr_ != nullptr) {
            ::munmap(addr_, size_);
            addr_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        size_ = 0;
    }
};

}  // namespace UC
