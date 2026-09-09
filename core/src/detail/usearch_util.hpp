#pragma once

#include <usearch/index_dense.hpp>

#include <cstddef>
#include <stdexcept>
#include <string>

// Small usearch helper shared by every store that owns a usearch index.
// Internal implementation detail, not part of the public API.
namespace retrieval_engine::detail {

// Grows `index`'s capacity if it's about to be exceeded (usearch requires
// `size() < capacity()` ahead of every add()). Amortized-doubling, same
// strategy as std::vector, for stores that add one vector at a time.
inline void EnsureCapacity(unum::usearch::index_dense_t& index, const std::string& context) {
    if (index.size() != index.capacity()) return;
    const std::size_t new_capacity = index.capacity() == 0 ? 64 : index.capacity() * 2;
    if (!index.reserve(unum::usearch::index_limits_t(new_capacity)))
        throw std::runtime_error(context + ": failed to reserve usearch index capacity");
}

}  // namespace retrieval_engine::detail
