#pragma once

#include <usearch/index_dense.hpp>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

// A thin, storage-agnostic wrapper around a cosine-similarity usearch index:
// knows nothing about SQLite or the chunks schema, only integer keys and
// vectors. Rebuilding this from persisted state on open (usearch is a
// rebuildable sidecar; SQLite is authoritative) is the caller's job --
// see ChunkRepository::ForEachChunk() and ChunkStore's constructor.
namespace retrieval_engine::detail {

class DenseIndex {
public:
    explicit DenseIndex(std::size_t dim);

    std::size_t dimensions() const { return dimensions_; }

    // Throws std::invalid_argument if `embedding.size() != dimensions()`,
    // or std::runtime_error on a usearch failure.
    void Add(std::uint64_t key, const std::vector<float>& embedding);

    // Returns (up to) `k` nearest (key, distance) pairs, nearest-first
    // (lower distance = more similar). Throws std::invalid_argument if
    // `query.size() != dimensions()`, or std::runtime_error on a usearch
    // failure.
    std::vector<std::pair<std::uint64_t, float>> Search(const std::vector<float>& query, std::size_t k) const;

private:
    std::size_t dimensions_;
    unum::usearch::index_dense_t index_;
};

}  // namespace retrieval_engine::detail
