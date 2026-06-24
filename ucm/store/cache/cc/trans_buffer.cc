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
#include "trans_buffer.h"
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include "logger/logger.h"
#include "posix_shm.h"
#include "trans/buffer.h"
#include "trans/device.h"

namespace UC::CacheStore {

static constexpr size_t nHashTableBucket = 16411;
static constexpr auto invalidIndex = std::numeric_limits<size_t>::max();

static inline size_t Hash(const Detail::BlockId& blockId, size_t shard)
{
    static UC::Detail::BlockIdHasher blockIdHasher;
    static std::hash<size_t> shardHasher;
    constexpr auto goldenSection = 0x9e3779b97f4a7c15ULL;
    size_t h1 = blockIdHasher(blockId);
    size_t h2 = shardHasher(shard);
    return (h1 ^ (h2 + goldenSection + (h1 << 6) + (h1 >> 2))) % nHashTableBucket;
}

struct BufferMetaNode {
    Detail::BlockId block;
    size_t shard;
    size_t reference;
    size_t hash;
    size_t prev;
    size_t next;
    bool ready;
    void Init()
    {
        reference = 0;
        hash = invalidIndex;
        prev = invalidIndex;
        next = invalidIndex;
        ready = false;
    }
};

class BufferStrategy {
protected:
    struct BaseConfig {
        int32_t deviceId{-1};
        size_t nodeSize{0};
        size_t totalSize{0};
        size_t reservedNumber{0};
    };
    BaseConfig base_;

public:
    BufferStrategy(int32_t deviceId, size_t nodeSize, size_t totalSize, size_t reservedNumber)
        : base_({deviceId, nodeSize, totalSize, reservedNumber})
    {
    }
    virtual ~BufferStrategy() = default;
    virtual Status Setup() = 0;
    virtual void BucketLock(size_t iBucket) = 0;
    virtual bool BucketTryLock(size_t iBucket) = 0;
    virtual void BucketUnlock(size_t iBucket) = 0;
    virtual void NodeLock(size_t iNode) = 0;
    virtual void NodeUnlock(size_t iNode) = 0;
    virtual size_t& FirstAt(size_t iBucket) = 0;
    virtual size_t FetchNode(bool allowReserved) = 0;
    virtual void* DataAt(size_t iNode) = 0;
    virtual BufferMetaNode* MetaAt(size_t iNode) = 0;
};

class LocalBufferStrategy : public BufferStrategy {
    struct BufferHeader {
        size_t buckets[nHashTableBucket];
        size_t freeHead;
        size_t nodeSize;
        size_t nNode;
    };
    struct LocalMutex {
        pthread_mutex_t mutex;
        ~LocalMutex() { pthread_mutex_destroy(&mutex); }
        void Init()
        {
            pthread_mutexattr_t attr;
            pthread_mutexattr_init(&attr);
            pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_PRIVATE);
            pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST);
            pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ADAPTIVE_NP);
            pthread_mutex_init(&mutex, &attr);
            pthread_mutexattr_destroy(&attr);
        }
        void Lock() { pthread_mutex_lock(&mutex); }
        bool TryLock() { return pthread_mutex_trylock(&mutex) == 0; }
        void Unlock() { pthread_mutex_unlock(&mutex); }
    };
    struct LocalLock {
        pthread_spinlock_t lock;
        ~LocalLock() { pthread_spin_destroy(&lock); }
        void Init() { pthread_spin_init(&lock, PTHREAD_PROCESS_PRIVATE); }
        void Lock() { pthread_spin_lock(&lock); }
        bool TryLock() { return pthread_spin_trylock(&lock) == 0; }
        void Unlock() { pthread_spin_unlock(&lock); }
    };

    bool ioDirect_{false};
    BufferHeader header_;
    LocalMutex bucketLocks_[nHashTableBucket];
    std::unique_ptr<LocalLock[]> nodeLocks_;
    std::unique_ptr<BufferMetaNode[]> meta_;
    std::shared_ptr<void> data_;

public:
    LocalBufferStrategy(int32_t deviceId, size_t nodeSize, size_t totalSize, size_t reservedNumber,
                        bool ioDirect)
        : BufferStrategy(deviceId, nodeSize, totalSize, reservedNumber), ioDirect_(ioDirect)
    {
    }
    Status Setup() override
    {
        const auto deviceId = base_.deviceId;
        const auto totalSize = base_.totalSize;
        const auto nodeSize = base_.nodeSize;
        auto nNode = totalSize / nodeSize;
        try {
            nodeLocks_ = std::make_unique<LocalLock[]>(nNode);
            meta_ = std::make_unique<BufferMetaNode[]>(nNode);
            for (size_t i = 0; i < nHashTableBucket; i++) { bucketLocks_[i].Init(); }
            for (size_t i = 0; i < nNode; i++) { nodeLocks_[i].Init(); }
        } catch (const std::exception& e) {
            UC_ERROR("Failed({}) to alloc buffer.", e.what());
            return Status::Error(e.what());
        }
        Trans::Device device;
        auto s = device.Setup(deviceId);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to setup device({}).", s, deviceId);
            return s;
        }
        auto buffer = device.MakeBuffer();
        if (!buffer) [[unlikely]] {
            UC_ERROR("Failed to make buffer on device({}).", deviceId);
            return Status::Error();
        }
        data_ = ioDirect_ ? buffer->MakeHostBuffer4DirectIo(nodeSize * nNode)
                          : buffer->MakeHostBuffer(nodeSize * nNode);
        if (!data_) [[unlikely]] {
            UC_ERROR("Failed to make pinned({}) for device({}).", nodeSize * nNode, deviceId);
            return Status::OutOfMemory();
        }
        for (size_t i = 0; i < nHashTableBucket; i++) { header_.buckets[i] = invalidIndex; }
        for (size_t i = 0; i < nNode; i++) { meta_[i].Init(); }
        header_.freeHead = 0;
        header_.nodeSize = nodeSize;
        header_.nNode = nNode;
        return Status::OK();
    }
    void BucketLock(size_t iBucket) override { bucketLocks_[iBucket].Lock(); }
    bool BucketTryLock(size_t iBucket) override { return bucketLocks_[iBucket].TryLock(); }
    void BucketUnlock(size_t iBucket) override { bucketLocks_[iBucket].Unlock(); }
    void NodeLock(size_t iNode) override { nodeLocks_[iNode].Lock(); }
    void NodeUnlock(size_t iNode) override { nodeLocks_[iNode].Unlock(); }
    size_t& FirstAt(size_t iBucket) override { return header_.buckets[iBucket]; }
    size_t FetchNode(bool allowReserved) override
    {
        const auto limit = header_.nNode - (allowReserved ? 0 : base_.reservedNumber);
        if (header_.freeHead >= limit) { header_.freeHead = 0; }
        return header_.freeHead++;
    }
    void* DataAt(size_t iNode) override
    {
        return ((std::byte*)data_.get()) + header_.nodeSize * iNode;
    }
    BufferMetaNode* MetaAt(size_t iNode) override { return meta_.get() + iNode; }
};

class HugePageShm {
private:
    std::string name_{};
    std::string path_{};
    int32_t handle_{-1};

    static std::string FileName(std::string name)
    {
        for (auto& ch : name) {
            if (ch == '/' || ch == '\\') { ch = '_'; }
        }
        return name;
    }

    static std::string MakePath(const std::string& name)
    {
        return (std::filesystem::path{"/hugepages-2Mi"} / FileName(name)).string();
    }

public:
    explicit HugePageShm(std::string name) : name_{std::move(name)}, path_{MakePath(name_)} {}
    ~HugePageShm()
    {
        if (handle_ != -1) { close(handle_); }
    }
    const std::string& Path() const noexcept { return path_; }
    std::string Readme() const { return path_; }
    Status Create(size_t size)
    {
        size_ = size;
        static constexpr auto NewFilePerm = (S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
        handle_ = open(path_.c_str(), O_CREAT | O_EXCL | O_RDWR, NewFilePerm);
        if (handle_ < 0) {
            const auto eno = errno;
            if (eno == EEXIST) { return Status::DuplicateKey(); }
            return Status{eno, std::to_string(eno)};
        }
        const auto ret = ftruncate64(handle_, size);
        const auto truncEno = errno;
        if (ret == 0) { return Status::OK(); }
        Unlink();
        return Status{truncEno, std::to_string(truncEno)};
    }
    Status Open()
    {
        handle_ = open(path_.c_str(), O_RDWR);
        const auto eno = errno;
        if (handle_ >= 0) { return Status::OK(); }
        return Status{eno, std::to_string(eno)};
    }
    Status Attach(void*& addr)
    {
        constexpr auto prot = PROT_READ | PROT_WRITE;
        constexpr auto flags = MAP_SHARED | MAP_POPULATE;
        addr = mmap(nullptr, size_, prot, flags, handle_, 0);
        const auto eno = errno;
        if (addr != MAP_FAILED) { return Status::OK(); }
        addr = nullptr;
        return Status{eno, std::to_string(eno)};
    }
    void SetSize(size_t size) noexcept { size_ = size; }
    static void Detach(void* addr, size_t size) { munmap(addr, size); }
    void Unlink()
    {
        if (handle_ != -1) {
            close(handle_);
            handle_ = -1;
        }
        unlink(path_.c_str());
    }

private:
    size_t size_{0};
};

class SharedBufferStrategy : public BufferStrategy {
protected:
    struct ShareMutex {
        pthread_mutex_t mutex;
        ~ShareMutex() = delete;
        void Init()
        {
            pthread_mutexattr_t attr;
            pthread_mutexattr_init(&attr);
            pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
            pthread_mutexattr_setrobust(&attr, PTHREAD_MUTEX_ROBUST);
            pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ADAPTIVE_NP);
            pthread_mutex_init(&mutex, &attr);
            pthread_mutexattr_destroy(&attr);
        }
        void Lock() { pthread_mutex_lock(&mutex); }
        bool TryLock() { return pthread_mutex_trylock(&mutex) == 0; }
        void Unlock() { pthread_mutex_unlock(&mutex); }
    };
    struct ShareLock {
        pthread_spinlock_t lock;
        ~ShareLock() = delete;
        void Init() { pthread_spin_init(&lock, PTHREAD_PROCESS_SHARED); }
        void Lock() { pthread_spin_lock(&lock); }
        bool TryLock() { return pthread_spin_trylock(&lock) == 0; }
        void Unlock() { pthread_spin_unlock(&lock); }
    };
    static constexpr size_t sharedBufferMagic = (('S' << 16) | ('b' << 8) | 1);
    struct BufferHeader {
        std::atomic<size_t> magic;
        uint64_t nameHash;
        ShareLock lock;
        size_t nNode;
        size_t freeHead;
        size_t buckets[nHashTableBucket];
        ShareMutex bucketLocks[nHashTableBucket];
        ShareLock nodeLocks[0];
    };

    BufferHeader* header_{nullptr};
    BufferMetaNode* meta_{nullptr};
    std::byte* data_{nullptr};
    std::byte* dataOnDevice_{nullptr};
    const std::string& uuid_;
    std::string shmName_;
    std::string hugeShmName_;
    size_t nodeSize_{0};
    size_t nNode_{0};
    void* addrress_{nullptr};
    size_t totalSize_{0};
    bool useHugeShm_{false};
    bool ownHugeShm_{false};

    static constexpr size_t kHugePageSize = 2UL << 20;
    static size_t AlignUp(size_t size, size_t alignment) noexcept
    {
        return (size + alignment - 1) / alignment * alignment;
    }
    size_t MetaOffset() const noexcept { return sizeof(BufferHeader) + sizeof(ShareLock) * nNode_; }
    size_t DataOffset() const noexcept
    {
        const auto size = MetaOffset() + sizeof(BufferMetaNode) * nNode_;
        return AlignUp(size, kHugePageSize);
    }
    size_t DataSize() const noexcept { return nodeSize_ * nNode_; }
    static size_t HugeMapSize(size_t size) noexcept { return AlignUp(size, kHugePageSize); }
    static uint64_t ShmNameHash(const std::string& name) noexcept
    {
        uint64_t hash = 1469598103934665603ULL;
        for (const auto ch : name) {
            hash ^= static_cast<unsigned char>(ch);
            hash *= 1099511628211ULL;
        }
        return hash;
    }
    static const std::string& ShmPrefix() noexcept
    {
        static std::string prefix{"uc_shm_cache_"};
        return prefix;
    }
    static void CleanUpShmFileExceptMe(const std::string& me)
    {
        namespace fs = std::filesystem;
        std::string_view prefix = ShmPrefix();
        const auto now = fs::file_time_type::clock::now();
        const auto keepThreshold = std::chrono::minutes(10);
        auto cleanDir = [&](const fs::path& dir) {
            try {
                if (!fs::exists(dir)) { return; }
                for (const auto& entry : fs::directory_iterator(dir)) {
                    const auto& path = entry.path();
                    const auto& name = path.filename().string();
                    if (!entry.is_regular_file() || name.compare(0, prefix.size(), prefix) != 0 ||
                        name == me) {
                        continue;
                    }
                    const auto lwt = fs::last_write_time(path);
                    if (now - lwt <= keepThreshold) { continue; }
                    fs::remove(path);
                }
            } catch (...) {
            }
        };
        cleanDir("/dev/shm");
        cleanDir("/hugepages-2Mi");
    }
    static Status MmapShmFile(PosixShm& shmFile, const size_t size, void*& addr,
                              bool needTrunc = true)
    {
        auto s = Status::OK();
        if (needTrunc) {
            s = shmFile.Truncate(size);
            if (s.Failure()) [[unlikely]] {
                UC_ERROR("Failed({}) to trunc file({}) with size({}).", s, shmFile.ShmName(), size);
                return s;
            }
        }
        s = shmFile.MMap(addr, size, true, true, true);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to mmap file({}) with size({}).", s, shmFile.ShmName(), size);
            addr = nullptr;
            return s;
        }
        s = PosixShm::MAdvise(addr, size, MADV_HUGEPAGE);
        if (s.Success()) {
            UC_DEBUG("Advise shm file({}) to use transparent huge pages, size({}).",
                     shmFile.ShmName(), size);
        } else {
            UC_DEBUG("Failed({}) to advise shm file({}) to use transparent huge pages.",
                     s, shmFile.ShmName());
        }
        return Status::OK();
    }
    static Status AttachHugeShm(HugePageShm& shmFile, size_t size, void*& addr)
    {
        shmFile.SetSize(size);
        auto s = shmFile.Attach(addr);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to attach huge shm({}).", s, shmFile.Readme());
            addr = nullptr;
            return s;
        }
        return Status::OK();
    }
    static Status WaitShmHeaderReady(BufferHeader* header)
    {
        constexpr auto retryInterval = std::chrono::milliseconds(100);
        constexpr auto maxTryTime = 100;
        auto tryTime = 0;
        do {
            if (header->magic == sharedBufferMagic) { break; }
            if (tryTime > maxTryTime) { return Status::Retry(); }
            std::this_thread::sleep_for(retryInterval);
            tryTime++;
        } while (true);
        return Status::OK();
    }
    Status WaitCurrentShmHeaderReady(BufferHeader* header) const
    {
        auto s = WaitShmHeaderReady(header);
        if (s.Failure()) { return s; }
        if (header->nameHash != ShmNameHash(shmName_)) {
            return Status::InvalidParam("shared buffer name hash mismatch");
        }
        return Status::OK();
    }
    Status InitShmBuffer(PosixShm& shmFile)
    {
        auto s = MmapShmFile(shmFile, totalSize_, addrress_);
        if (s.Failure()) [[unlikely]] { return s; }
        header_ = static_cast<BufferHeader*>(addrress_);
        meta_ = (BufferMetaNode*)(static_cast<std::byte*>(addrress_) + MetaOffset());
        header_->lock.Init();
        header_->nameHash = ShmNameHash(shmName_);
        header_->nNode = nNode_;
        header_->freeHead = 0;
        for (size_t i = 0; i < nHashTableBucket; i++) {
            header_->buckets[i] = invalidIndex;
            header_->bucketLocks[i].Init();
        }
        for (size_t i = 0; i < nNode_; i++) {
            header_->nodeLocks[i].Init();
            meta_[i].Init();
        }
        header_->magic = sharedBufferMagic;
        return Status::OK();
    }
    Status LoadShmBuffer(PosixShm& shmFile)
    {
        auto s = shmFile.ShmOpen(PosixShm::OpenFlag::READ_WRITE);
        if (s.Failure()) {
            UC_ERROR("Failed({}) to open file({}).", s, shmFile.ShmName());
            return s;
        }
        s = MmapShmFile(shmFile, totalSize_, addrress_, false);
        if (s.Failure()) [[unlikely]] { return s; }
        header_ = static_cast<BufferHeader*>(addrress_);
        s = WaitCurrentShmHeaderReady(header_);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Shm file({}) not ready.", shmFile.ShmName());
            return s;
        }
        meta_ = (BufferMetaNode*)(static_cast<std::byte*>(addrress_) + MetaOffset());
        return Status::OK();
    }
    Status InitHugeShmBuffer(HugePageShm& shmFile)
    {
        auto s = AttachHugeShm(shmFile, totalSize_, addrress_);
        if (s.Failure()) [[unlikely]] { return s; }
        useHugeShm_ = true;
        ownHugeShm_ = true;
        UC_INFO("Use HugeTLB shm({}) for CacheStore shared buffer.", shmFile.Readme());
        header_ = static_cast<BufferHeader*>(addrress_);
        meta_ = (BufferMetaNode*)(static_cast<std::byte*>(addrress_) + MetaOffset());
        header_->lock.Init();
        header_->nameHash = ShmNameHash(shmName_);
        header_->nNode = nNode_;
        header_->freeHead = 0;
        for (size_t i = 0; i < nHashTableBucket; i++) {
            header_->buckets[i] = invalidIndex;
            header_->bucketLocks[i].Init();
        }
        for (size_t i = 0; i < nNode_; i++) {
            header_->nodeLocks[i].Init();
            meta_[i].Init();
        }
        header_->magic = sharedBufferMagic;
        return Status::OK();
    }
    Status LoadHugeShmBuffer(HugePageShm& shmFile)
    {
        auto s = shmFile.Open();
        if (s.Failure()) {
            if (s.Underlying() != ENOENT) {
                UC_ERROR("Failed({}) to open huge shm({}).", s, shmFile.Readme());
            }
            return s;
        }
        s = AttachHugeShm(shmFile, totalSize_, addrress_);
        if (s.Failure()) [[unlikely]] { return s; }
        useHugeShm_ = true;
        UC_INFO("Use HugeTLB shm({}) for CacheStore shared buffer.", shmFile.Readme());
        header_ = static_cast<BufferHeader*>(addrress_);
        s = WaitCurrentShmHeaderReady(header_);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Huge shm({}) not ready.", shmFile.Readme());
            return s;
        }
        meta_ = (BufferMetaNode*)(static_cast<std::byte*>(addrress_) + MetaOffset());
        return Status::OK();
    }
    Status LoadHugeShmHeader(HugePageShm& shmFile)
    {
        auto s = shmFile.Open();
        if (s.Failure()) {
            if (s.Underlying() != ENOENT) {
                UC_ERROR("Failed({}) to open huge shm({}).", s, shmFile.Readme());
            }
            return s;
        }
        void* addr = nullptr;
        const auto size = HugeMapSize(sizeof(BufferHeader));
        s = AttachHugeShm(shmFile, size, addr);
        if (s.Failure()) [[unlikely]] { return s; }
        auto header = static_cast<BufferHeader*>(addr);
        s = WaitCurrentShmHeaderReady(header);
        if (s.Failure()) [[unlikely]] {
            HugePageShm::Detach(addr, size);
            UC_ERROR("Huge shm({}) not ready.", shmFile.Readme());
            return s;
        }
        nNode_ = header->nNode;
        HugePageShm::Detach(addr, size);
        return Status::OK();
    }
    Status LoadHugeShmMeta(HugePageShm& shmFile)
    {
        auto s = LoadHugeShmHeader(shmFile);
        if (s.Failure()) [[unlikely]] { return s; }
        totalSize_ = DataOffset();
        s = AttachHugeShm(shmFile, totalSize_, addrress_);
        if (s.Failure()) [[unlikely]] { return s; }
        useHugeShm_ = true;
        UC_INFO("Use HugeTLB shm({}) for CacheStore shared buffer watcher.",
                shmFile.Readme());
        header_ = static_cast<BufferHeader*>(addrress_);
        meta_ = (BufferMetaNode*)(static_cast<std::byte*>(addrress_) + MetaOffset());
        return Status::OK();
    }
    Status RegisterBuffer(int32_t deviceId)
    {
        data_ = static_cast<std::byte*>(addrress_) + DataOffset();
        Trans::Device device;
        auto s = device.Setup(deviceId);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to setup device({}).", s, deviceId);
            return s;
        }
        const auto dataSize = DataSize();
        s = Trans::Buffer::RegisterHostBuffer((void*)data_, dataSize, (void**)&dataOnDevice_);
        if (s.Failure()) [[unlikely]] {
            UC_ERROR("Failed({}) to register buffer({}) to device({}).", s, dataSize, deviceId);
            return s;
        }
        return Status::OK();
    }

public:
    SharedBufferStrategy(const std::string& uuid, int32_t deviceId, size_t nodeSize,
                         size_t totalSize, size_t reservedNumber)
        : BufferStrategy(deviceId, nodeSize, totalSize, reservedNumber), uuid_(uuid)
    {
    }
    ~SharedBufferStrategy() override
    {
        if (data_) { Trans::Buffer::UnregisterHostBuffer(data_); }
        if (addrress_) {
            if (useHugeShm_) {
                HugePageShm::Detach(addrress_, totalSize_);
            } else {
                PosixShm::MUnmap(addrress_, totalSize_);
            }
        }
        if (useHugeShm_) {
            if (ownHugeShm_) { HugePageShm{hugeShmName_}.Unlink(); }
        } else {
            PosixShm{shmName_}.ShmUnlink();
        }
    }
    Status Setup() override
    {
        const auto& uuid = uuid_;
        const auto deviceId = base_.deviceId;
        const auto nodeSize = base_.nodeSize;
        const auto totalSize = base_.totalSize;
        shmName_ = ShmPrefix() + uuid;
        nodeSize_ = nodeSize;
        nNode_ = totalSize / nodeSize;
        CleanUpShmFileExceptMe(shmName_);
        const auto dataOffset = DataOffset();
        totalSize_ = HugeMapSize(dataOffset + DataSize());
        hugeShmName_ = shmName_;
        HugePageShm hugeFile{hugeShmName_};
        auto s = hugeFile.Create(totalSize_);
        if (s.Success()) {
            s = InitHugeShmBuffer(hugeFile);
            if (s.Success()) { return RegisterBuffer(deviceId); }
            hugeFile.Unlink();
            UC_WARN("Failed({}) to create HugeTLB shm buffer({}), fallback to POSIX shm.",
                    s, hugeFile.Readme());
        } else if (s == Status::DuplicateKey()) {
            s = LoadHugeShmBuffer(hugeFile);
            if (s.Success()) { return RegisterBuffer(deviceId); }
            return s;
        } else {
            UC_WARN("Failed({}) to create HugeTLB shm({}), fallback to POSIX shm.",
                    s, hugeFile.Readme());
            addrress_ = nullptr;
            useHugeShm_ = false;
        }

        PosixShm shmFile{shmName_};
        const auto flags =
            PosixShm::OpenFlag::CREATE | PosixShm::OpenFlag::EXCL | PosixShm::OpenFlag::READ_WRITE;
        s = shmFile.ShmOpen(flags);
        if (s.Success()) {
            s = InitShmBuffer(shmFile);
        } else if (s == Status::DuplicateKey()) {
            s = LoadShmBuffer(shmFile);
        } else {
            UC_ERROR("Failed({}) to open file({}) with flags({}).", s, shmName_, flags);
            return s;
        }
        return RegisterBuffer(deviceId);
    }
    void BucketLock(size_t iBucket) override { header_->bucketLocks[iBucket].Lock(); }
    bool BucketTryLock(size_t iBucket) override { return header_->bucketLocks[iBucket].TryLock(); }
    void BucketUnlock(size_t iBucket) override { header_->bucketLocks[iBucket].Unlock(); }
    void NodeLock(size_t iNode) override { header_->nodeLocks[iNode].Lock(); }
    void NodeUnlock(size_t iNode) override { header_->nodeLocks[iNode].Unlock(); }
    size_t& FirstAt(size_t iBucket) override { return header_->buckets[iBucket]; }
    size_t FetchNode(bool allowReserved) override
    {
        const auto limit = header_->nNode - (allowReserved ? 0 : base_.reservedNumber);
        header_->lock.Lock();
        if (header_->freeHead >= limit) { header_->freeHead = 0; }
        const auto iNode = header_->freeHead++;
        header_->lock.Unlock();
        return iNode;
    }
    void* DataAt(size_t iNode) override { return data_ + nodeSize_ * iNode; }
    BufferMetaNode* MetaAt(size_t iNode) override { return meta_ + iNode; }
};

class SharedBufferWatcherStrategy : public SharedBufferStrategy {
public:
    explicit SharedBufferWatcherStrategy(const std::string& uuid)
        : SharedBufferStrategy(uuid, -1, 0, 0, 0)
    {
    }
    Status Setup() override
    {
        shmName_ = ShmPrefix() + uuid_;
        CleanUpShmFileExceptMe(shmName_);
        hugeShmName_ = shmName_;
        HugePageShm hugeFile{hugeShmName_};
        auto s = LoadHugeShmMeta(hugeFile);
        if (s.Success()) { return Status::OK(); }
        if (s.Underlying() != ENOENT) { return s; }

        PosixShm shmFile{shmName_};
        s = shmFile.ShmOpen(PosixShm::OpenFlag::READ_WRITE);
        if (s.Failure()) {
            UC_ERROR("Failed({}) to open file({}).", s, shmFile.ShmName());
            return s;
        }
        void* addr = nullptr;
        auto size = sizeof(BufferHeader);
        s = MmapShmFile(shmFile, size, addr, false);
        if (s.Failure()) [[unlikely]] { return s; }
        auto header = static_cast<BufferHeader*>(addr);
        s = WaitCurrentShmHeaderReady(header);
        if (s.Failure()) [[unlikely]] {
            PosixShm::MUnmap(addr, size);
            UC_ERROR("Shm file({}) not ready.", shmFile.ShmName());
            return s;
        }
        nNode_ = header->nNode;
        shmFile.MUnmap(addr, size);
        totalSize_ = DataOffset();
        s = MmapShmFile(shmFile, totalSize_, addrress_, false);
        if (s.Failure()) [[unlikely]] { return s; }
        header_ = static_cast<BufferHeader*>(addrress_);
        meta_ = (BufferMetaNode*)(static_cast<std::byte*>(addrress_) + MetaOffset());
        return Status::OK();
    }
    void* DataAt(size_t iNode) override { return nullptr; }
};

Status TransBuffer::Setup(const Config& config)
{
    bypassHitOnLoad_ = config.cacheLoadBackendOnly;
    try {
        if (!config.shareBufferEnable) {
            strategy_ = std::make_shared<LocalBufferStrategy>(
                config.deviceId, config.shardSize, config.bufferCapacity,
                config.loadExclusiveBufferNumber, config.ioDirect);
        } else if (config.deviceId >= 0) {
            strategy_ = std::make_shared<SharedBufferStrategy>(
                config.uniqueId, config.deviceId, config.shardSize, config.bufferCapacity,
                config.loadExclusiveBufferNumber);
        } else {
            strategy_ = std::make_shared<SharedBufferWatcherStrategy>(config.uniqueId);
        }
    } catch (const std::exception& e) {
        return Status::Error(fmt::format("failed({}) to make buffer strategy", e.what()));
    }
    return strategy_->Setup();
}

TransBuffer::Handle TransBuffer::Get(const Detail::BlockId& blockId, size_t shardIdx,
                                     bool allowReserved, bool isLoad)
{
    auto iBucket = Hash(blockId, shardIdx);
    bool owner = false;
    strategy_->BucketLock(iBucket);
    auto iNode = FindAt(iBucket, blockId, shardIdx, owner);
    if (iNode != invalidIndex) {
        if (bypassHitOnLoad_ && isLoad && owner && Ready(iNode)) { MarkNotReady(iNode); }
        strategy_->BucketUnlock(iBucket);
        return Handle{this, iNode, owner};
    }
    iNode = Alloc(blockId, shardIdx, iBucket, allowReserved);
    strategy_->BucketUnlock(iBucket);
    return Handle(this, iNode, true);
}

bool TransBuffer::Exist(const Detail::BlockId& blockId, size_t shardIdx)
{
    auto iBucket = Hash(blockId, shardIdx);
    strategy_->BucketLock(iBucket);
    auto exist = ExistAt(iBucket, blockId, shardIdx);
    strategy_->BucketUnlock(iBucket);
    return exist;
}

bool TransBuffer::ExistAt(size_t iBucket, const Detail::BlockId& blockId, size_t shardIdx)
{
    auto iNode = strategy_->FirstAt(iBucket);
    while (iNode != invalidIndex) {
        auto meta = strategy_->MetaAt(iNode);
        strategy_->NodeLock(iNode);
        if (meta->block == blockId && meta->shard == shardIdx) {
            strategy_->NodeUnlock(iNode);
            return true;
        }
        auto next = meta->next;
        strategy_->NodeUnlock(iNode);
        iNode = next;
    }
    return false;
}

size_t TransBuffer::FindAt(size_t iBucket, const Detail::BlockId& blockId, size_t shardIdx,
                           bool& owner)
{
    auto iNode = strategy_->FirstAt(iBucket);
    while (iNode != invalidIndex) {
        auto meta = strategy_->MetaAt(iNode);
        strategy_->NodeLock(iNode);
        if (meta->block == blockId && meta->shard == shardIdx) {
            owner = meta->reference == 0;
            ++meta->reference;
            strategy_->NodeUnlock(iNode);
            break;
        }
        auto next = meta->next;
        strategy_->NodeUnlock(iNode);
        iNode = next;
    }
    return iNode;
}

size_t TransBuffer::Alloc(const Detail::BlockId& blockId, size_t shardIdx, size_t iBucket,
                          bool allowReserved)
{
    for (;;) {
        auto iNode = strategy_->FetchNode(allowReserved);
        auto meta = strategy_->MetaAt(iNode);
        strategy_->NodeLock(iNode);
        if (meta->reference > 0) {
            strategy_->NodeUnlock(iNode);
            continue;
        }
        const auto oldBucket = meta->hash;
        if (oldBucket != iBucket) {
            if (oldBucket != invalidIndex) {
                if (!strategy_->BucketTryLock(oldBucket)) {
                    strategy_->NodeUnlock(iNode);
                    continue;
                }
                Remove(oldBucket, iNode);
                strategy_->BucketUnlock(oldBucket);
            }
            MoveTo(iBucket, iNode);
        }
        ++meta->reference;
        meta->block = blockId;
        meta->shard = shardIdx;
        meta->ready = false;
        strategy_->NodeUnlock(iNode);
        return iNode;
    }
}

void TransBuffer::MoveTo(size_t iBucket, size_t iNode)
{
    auto meta = strategy_->MetaAt(iNode);
    auto& head = strategy_->FirstAt(iBucket);
    auto n = head;
    meta->next = n;
    if (n != invalidIndex) {
        auto next = strategy_->MetaAt(n);
        strategy_->NodeLock(n);
        next->prev = iNode;
        strategy_->NodeUnlock(n);
    }
    meta->hash = iBucket;
    head = iNode;
}

void TransBuffer::Remove(size_t iBucket, size_t iNode)
{
    auto meta = strategy_->MetaAt(iNode);
    auto p = meta->prev;
    if (p != invalidIndex) {
        auto prev = strategy_->MetaAt(p);
        strategy_->NodeLock(p);
        prev->next = meta->next;
        strategy_->NodeUnlock(p);
    }
    auto n = meta->next;
    if (n != invalidIndex) {
        auto next = strategy_->MetaAt(n);
        strategy_->NodeLock(n);
        next->prev = meta->prev;
        strategy_->NodeUnlock(n);
    }
    if (strategy_->FirstAt(iBucket) == iNode) { strategy_->FirstAt(iBucket) = n; }
    meta->prev = meta->next = invalidIndex;
    meta->hash = invalidIndex;
}

void* TransBuffer::DataAt(Index pos) { return strategy_->DataAt(pos); }

void TransBuffer::Acquire(Index pos)
{
    strategy_->NodeLock(pos);
    ++strategy_->MetaAt(pos)->reference;
    strategy_->NodeUnlock(pos);
}

void TransBuffer::Release(Index pos)
{
    strategy_->NodeLock(pos);
    --strategy_->MetaAt(pos)->reference;
    strategy_->NodeUnlock(pos);
}

bool TransBuffer::Ready(Index pos)
{
    strategy_->NodeLock(pos);
    auto ready = strategy_->MetaAt(pos)->ready;
    strategy_->NodeUnlock(pos);
    return ready;
}

void TransBuffer::MarkReady(Index pos)
{
    strategy_->NodeLock(pos);
    strategy_->MetaAt(pos)->ready = true;
    strategy_->NodeUnlock(pos);
}

void TransBuffer::MarkNotReady(Index pos)
{
    strategy_->NodeLock(pos);
    strategy_->MetaAt(pos)->ready = false;
    strategy_->NodeUnlock(pos);
}

}  // namespace UC::CacheStore
