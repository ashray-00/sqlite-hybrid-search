#pragma once

#include <usearch/index_dense.hpp>

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// Small usearch helpers shared by every store that owns a usearch index.
// Internal implementation detail, not part of the public API.
namespace retrieval_engine::detail {

// How many search threads a usearch index is provisioned for -- also the
// size of the SearchSlotPool below. Clamped to >= 1 for hosts where
// std::thread::hardware_concurrency() reports 0.
inline std::size_t UsearchSearchThreadCount() {
    return std::max<std::size_t>(1, std::thread::hardware_concurrency());
}

// Grows `index`'s capacity if it's about to be exceeded (usearch requires
// `size() < capacity()` ahead of every add()). Amortized-doubling, same
// strategy as std::vector, for stores that add one vector at a time. The
// two-arg index_limits_t also sizes the per-thread `contexts_` buffers for
// `search_threads` concurrent searchers (a mid-add reserve would otherwise
// reset that to the default and shrink `contexts_`).
inline void EnsureCapacity(unum::usearch::index_dense_t& index, std::size_t search_threads, const std::string& context) {
    if (index.size() != index.capacity()) return;
    const std::size_t new_capacity = index.capacity() == 0 ? 64 : index.capacity() * 2;
    if (!index.reserve(unum::usearch::index_limits_t(new_capacity, search_threads)))
        throw std::runtime_error(context + ": failed to reserve usearch index capacity");
}

// Ensures `index` has `contexts_` / capacity provisioned for
// `search_threads` concurrent searchers, without growing the member count.
// Call after construction, load(), and clear(): load() re-derives the
// thread count from the file (which may have been written on a host with a
// different core count) and clear()/reset() drop it to the default.
inline void ProvisionSearchThreads(unum::usearch::index_dense_t& index, std::size_t search_threads,
                                   const std::string& context) {
    const std::size_t members = std::max<std::size_t>(index.capacity(), 1);
    if (!index.reserve(unum::usearch::index_limits_t(members, search_threads)))
        throw std::runtime_error(context + ": failed to provision usearch search threads");
}

// A fixed pool of usearch "thread" slot ids in [0, size). usearch's
// search(vector, k) with the default any_thread() picks a private
// context_t slot and *returns it to the pool the moment search() returns*
// -- but the search_result_t it hands back still references that slot's
// buffers, which we read afterwards via dump_to(). Two concurrent
// DenseIndex::Search calls then race on one context_t. Leasing an explicit
// slot id for the whole Search() call (search + dump) fixes it; a
// (size+1)-th concurrent searcher blocks here rather than exhausting
// usearch's own free list (undefined behaviour).
class SearchSlotPool {
public:
    explicit SearchSlotPool(std::size_t size) {
        free_.reserve(size);
        for (std::size_t i = 0; i < size; ++i) free_.push_back(i);
    }

    SearchSlotPool(const SearchSlotPool&) = delete;
    SearchSlotPool& operator=(const SearchSlotPool&) = delete;

    class Lease {
    public:
        Lease(SearchSlotPool* pool, std::size_t id) : pool_(pool), id_(id) {}
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        ~Lease() {
            if (pool_) pool_->Release(id_);
        }
        std::size_t id() const { return id_; }

    private:
        SearchSlotPool* pool_;
        std::size_t id_;
    };

    Lease Acquire() {
        std::unique_lock<std::mutex> lock(mutex_);
        available_.wait(lock, [this] { return !free_.empty(); });
        const std::size_t id = free_.back();
        free_.pop_back();
        return Lease(this, id);
    }

private:
    void Release(std::size_t id) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            free_.push_back(id);
        }
        available_.notify_one();
    }

    std::mutex mutex_;
    std::condition_variable available_;
    std::vector<std::size_t> free_;
};

}  // namespace retrieval_engine::detail
