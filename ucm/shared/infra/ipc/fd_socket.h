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

#include <cstddef>
#include <cstring>
#include <errno.h>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include "status/status.h"

namespace UC {

class FdSocket {
    int32_t sock_{-1};

    static void FillAbstractAddr(sockaddr_un& addr, const std::string& name)
    {
        std::memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        addr.sun_path[0] = '\0';
        auto cap = sizeof(addr.sun_path) - 1;
        auto len = std::min<std::size_t>(name.size(), cap - 1);
        std::memcpy(addr.sun_path + 1, name.data(), len);
    }

public:
    FdSocket() = default;
    ~FdSocket() { Close(); }
    FdSocket(const FdSocket&) = delete;
    FdSocket& operator=(const FdSocket&) = delete;

    Status Listen(const std::string& abstractName)
    {
        sock_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
        if (sock_ < 0) { return Status::OsApiError("socket failed"); }
        sockaddr_un addr{};
        FillAbstractAddr(addr, abstractName);
        socklen_t addrlen = static_cast<socklen_t>(sizeof(sa_family_t) + 1 + abstractName.size());
        if (::bind(sock_, reinterpret_cast<sockaddr*>(&addr), addrlen) != 0) {
            if (errno == EADDRINUSE) { return Status::DuplicateKey(); }
            return Status::OsApiError("bind failed");
        }
        if (::listen(sock_, kMaxRanksBacklog) != 0) { return Status::OsApiError("listen failed"); }
        return Status::OK();
    }

    Status AcceptAndSend(int32_t fdToSend)
    {
        int32_t conn = static_cast<int32_t>(::accept(sock_, nullptr, nullptr));
        if (conn < 0) { return Status::OsApiError("accept failed"); }
        msghdr msg{};
        char buf = 'x';
        iovec iov{&buf, 1};
        msg.msg_iov = &iov;
        msg.msg_iovlen = 1;
        char cmsgbuf[CMSG_SPACE(sizeof(int32_t))];
        std::memset(cmsgbuf, 0, sizeof(cmsgbuf));
        msg.msg_control = cmsgbuf;
        msg.msg_controllen = sizeof(cmsgbuf);
        auto cmsg = CMSG_FIRSTHDR(&msg);
        cmsg->cmsg_level = SOL_SOCKET;
        cmsg->cmsg_type = SCM_RIGHTS;
        cmsg->cmsg_len = CMSG_LEN(sizeof(int32_t));
        std::memcpy(CMSG_DATA(cmsg), &fdToSend, sizeof(int32_t));
        auto sent = ::sendmsg(conn, &msg, 0);
        ::close(conn);
        if (sent < 0) { return Status::OsApiError("sendmsg failed"); }
        return Status::OK();
    }

    Status Connect(const std::string& abstractName)
    {
        sock_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
        if (sock_ < 0) { return Status::OsApiError("socket failed"); }
        sockaddr_un addr{};
        FillAbstractAddr(addr, abstractName);
        socklen_t addrlen = static_cast<socklen_t>(sizeof(sa_family_t) + 1 + abstractName.size());
        if (::connect(sock_, reinterpret_cast<sockaddr*>(&addr), addrlen) != 0) {
            return Status::OsApiError("connect failed");
        }
        return Status::OK();
    }

    Status RecvFd(int32_t& fdOut)
    {
        msghdr msg{};
        char buf = 0;
        iovec iov{&buf, 1};
        msg.msg_iov = &iov;
        msg.msg_iovlen = 1;
        char cmsgbuf[CMSG_SPACE(sizeof(int32_t))];
        std::memset(cmsgbuf, 0, sizeof(cmsgbuf));
        msg.msg_control = cmsgbuf;
        msg.msg_controllen = sizeof(cmsgbuf);
        auto recvd = ::recvmsg(sock_, &msg, 0);
        if (recvd <= 0) { return Status::OsApiError("recvmsg failed"); }
        int32_t fd = -1;
        auto cmsg = CMSG_FIRSTHDR(&msg);
        if (cmsg != nullptr && cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
            std::memcpy(&fd, CMSG_DATA(cmsg), sizeof(int32_t));
        }
        fdOut = fd;
        return Status::OK();
    }

    int32_t SockFd() const { return sock_; }
    void Close()
    {
        if (sock_ >= 0) {
            ::shutdown(sock_, SHUT_RDWR);
            ::close(sock_);
            sock_ = -1;
        }
    }

private:
    static constexpr int32_t kMaxRanksBacklog = 256;
};

}  // namespace UC
