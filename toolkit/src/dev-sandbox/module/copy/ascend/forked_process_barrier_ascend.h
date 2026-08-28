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
#ifndef FORKED_PROCESS_BARRIER_ASCEND_H
#define FORKED_PROCESS_BARRIER_ASCEND_H

#include <cerrno>
#include <cstddef>
#include <ctime>
#include <new>
#include <pthread.h>
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>

namespace ascend_copy {

class ForkedProcessBarrier {
    static constexpr time_t kWaitTimeoutSeconds = 60;

    struct SharedState {
        pthread_mutex_t mutex;
        pthread_cond_t condition;
        size_t participants = 0;
        size_t arrived = 0;
        size_t generation = 0;
        bool aborted = false;
    };

    SharedState* state_ = nullptr;
    pid_t ownerPid_ = -1;

    static void Check(int status, const char* operation)
    {
        if (status != 0) { throw std::runtime_error(operation); }
    }

public:
    explicit ForkedProcessBarrier(size_t participants) : ownerPid_(getpid())
    {
        if (participants == 0) { throw std::invalid_argument("barrier participants is zero"); }

        void* mapping = mmap(nullptr, sizeof(SharedState), PROT_READ | PROT_WRITE,
                             MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        if (mapping == MAP_FAILED) { throw std::runtime_error("mmap process barrier failed"); }
        state_ = new (mapping) SharedState{};
        state_->participants = participants;

        pthread_mutexattr_t mutexAttr;
        pthread_condattr_t conditionAttr;
        try {
            Check(pthread_mutexattr_init(&mutexAttr), "pthread_mutexattr_init failed");
            Check(pthread_mutexattr_setpshared(&mutexAttr, PTHREAD_PROCESS_SHARED),
                  "pthread_mutexattr_setpshared failed");
            Check(pthread_mutex_init(&state_->mutex, &mutexAttr), "pthread_mutex_init failed");
            Check(pthread_mutexattr_destroy(&mutexAttr), "pthread_mutexattr_destroy failed");

            Check(pthread_condattr_init(&conditionAttr), "pthread_condattr_init failed");
            Check(pthread_condattr_setpshared(&conditionAttr, PTHREAD_PROCESS_SHARED),
                  "pthread_condattr_setpshared failed");
            Check(pthread_condattr_setclock(&conditionAttr, CLOCK_MONOTONIC),
                  "pthread_condattr_setclock failed");
            Check(pthread_cond_init(&state_->condition, &conditionAttr),
                  "pthread_cond_init failed");
            Check(pthread_condattr_destroy(&conditionAttr), "pthread_condattr_destroy failed");
        } catch (...) {
            munmap(state_, sizeof(SharedState));
            state_ = nullptr;
            throw;
        }
    }

    ForkedProcessBarrier(const ForkedProcessBarrier&) = delete;
    ForkedProcessBarrier& operator=(const ForkedProcessBarrier&) = delete;

    ~ForkedProcessBarrier()
    {
        if (state_ == nullptr || getpid() != ownerPid_) { return; }
        pthread_cond_destroy(&state_->condition);
        pthread_mutex_destroy(&state_->mutex);
        munmap(state_, sizeof(SharedState));
    }

    bool Wait()
    {
        Check(pthread_mutex_lock(&state_->mutex), "pthread_mutex_lock failed");
        if (state_->aborted) {
            pthread_mutex_unlock(&state_->mutex);
            return false;
        }

        const auto generation = state_->generation;
        if (++state_->arrived == state_->participants) {
            state_->arrived = 0;
            ++state_->generation;
            Check(pthread_cond_broadcast(&state_->condition), "pthread_cond_broadcast failed");
        } else {
            while (!state_->aborted && state_->generation == generation) {
                timespec deadline{};
                Check(clock_gettime(CLOCK_MONOTONIC, &deadline), "clock_gettime failed");
                deadline.tv_sec += kWaitTimeoutSeconds;
                const auto status =
                    pthread_cond_timedwait(&state_->condition, &state_->mutex, &deadline);
                if (status == ETIMEDOUT) {
                    state_->aborted = true;
                    pthread_cond_broadcast(&state_->condition);
                    break;
                }
                if (status != 0) {
                    state_->aborted = true;
                    pthread_cond_broadcast(&state_->condition);
                    pthread_mutex_unlock(&state_->mutex);
                    throw std::runtime_error("pthread_cond_timedwait failed");
                }
            }
        }

        const bool success = !state_->aborted;
        Check(pthread_mutex_unlock(&state_->mutex), "pthread_mutex_unlock failed");
        return success;
    }

    void Abort() noexcept
    {
        if (state_ == nullptr || pthread_mutex_lock(&state_->mutex) != 0) { return; }
        state_->aborted = true;
        pthread_cond_broadcast(&state_->condition);
        pthread_mutex_unlock(&state_->mutex);
    }
};

}  // namespace ascend_copy

#endif  // FORKED_PROCESS_BARRIER_ASCEND_H
