#pragma once

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <utility>
#include <vector>

// A fixed-size pool of interchangeable resources handed out one at a time.
// acquire() blocks until one is free, so at most N holders exist at once;
// the RAII Handle returns its resource when it goes out of scope. The
// resource type is a small copyable token (an integer id, a raw pointer) --
// the pool tracks availability, it does not own whatever the token refers
// to. Shared by DenseIndex's usearch search-slot pool and the read-only
// SQLite connection pool.
namespace retrieval_engine::detail {

template <typename Resource>
class BlockingPool {
public:
    explicit BlockingPool(std::vector<Resource> resources) : free_(std::move(resources)) {}

    BlockingPool(const BlockingPool&) = delete;
    BlockingPool& operator=(const BlockingPool&) = delete;

    class Handle {
    public:
        Handle() = default;
        Handle(BlockingPool* pool, Resource resource) : pool_(pool), resource_(std::move(resource)) {}
        Handle(Handle&& other) noexcept : pool_(other.pool_), resource_(std::move(other.resource_)) {
            other.pool_ = nullptr;
        }
        Handle& operator=(Handle&&) = delete;
        Handle(const Handle&) = delete;
        Handle& operator=(const Handle&) = delete;
        ~Handle() {
            if (pool_) pool_->Release(std::move(resource_));
        }

        bool valid() const { return pool_ != nullptr; }
        const Resource& value() const { return resource_; }

    private:
        BlockingPool* pool_ = nullptr;
        Resource resource_{};
    };

    Handle Acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        available_.wait(lock, [this] { return !free_.empty(); });
        Resource resource = std::move(free_.back());
        free_.pop_back();
        return Handle(this, std::move(resource));
    }

private:
    void Release(Resource resource) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            free_.push_back(std::move(resource));
        }
        available_.notify_one();
    }

    std::mutex mutex_;
    std::condition_variable available_;
    std::vector<Resource> free_;
};

}  // namespace retrieval_engine::detail
