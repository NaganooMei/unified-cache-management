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
#ifndef COPY_START_MODE_ASCEND_H
#define COPY_START_MODE_ASCEND_H

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

enum class CopyStartMode {
    LEGACY,
    PROCESS_BARRIER,
    DEVICE_GATE,
};

inline CopyStartMode ResolveCopyStartMode()
{
    const char* value = std::getenv("COPY_START_MODE");
    if (value == nullptr || value[0] == '\0' || std::string_view{value} == "device_gate") {
        return CopyStartMode::DEVICE_GATE;
    }
    if (std::string_view{value} == "legacy") { return CopyStartMode::LEGACY; }
    if (std::string_view{value} == "process_barrier") {
        return CopyStartMode::PROCESS_BARRIER;
    }
    throw std::invalid_argument(
        "COPY_START_MODE must be legacy, process_barrier, or device_gate");
}

inline std::string CopyStartModeMethodSuffix(CopyStartMode mode)
{
    switch (mode) {
        case CopyStartMode::LEGACY:
            return "-LEGACY";
        case CopyStartMode::PROCESS_BARRIER:
            return "-PB";
        case CopyStartMode::DEVICE_GATE:
            return {};
    }
    throw std::invalid_argument("unknown copy start mode");
}

#endif  // COPY_START_MODE_ASCEND_H
