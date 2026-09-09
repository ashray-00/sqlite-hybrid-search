#pragma once

#include <usearch/index_dense.hpp>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <thread>

// Capacity/thread provisioning helpers for a usearch index, shared by every
// store that owns one. Internal implementation detail, not part of the
// public API.
namespace retrieval_engine::detail {

// How many concurrent searches a usearch index is provisioned for -- also
// the number of search-slot ids DenseIndex leases out. Clamped to >= 1 for
// hosts where std::thread::hardware_concurrency() reports 0.
inline std::size_t UsearchSearchThreadCount() { return std::max<std::size_t>(1, std::thread::hardware_concurrency()); }

// Grows `index`'s capacity if it's about to be exceeded (usearch requires
// `size() < capacity()` ahead of every add()). Amortized-doubling, same
// strategy as std::vector, for stores that add one vector at a time. The
// two-arg index_limits_t also sizes the per-thread `contexts_` buffers for
// `search_threads` concurrent searchers (a mid-add reserve would otherwise
// reset that to the default and shrink `contexts_`).
inline void EnsureCapacity(unum::usearch::index_dense_t& index, std::size_t search_threads,
                           const std::string& context) {
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

}  // namespace retrieval_engine::detail
