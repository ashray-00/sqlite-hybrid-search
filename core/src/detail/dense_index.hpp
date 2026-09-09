#pragma once

#include <usearch/index_dense.hpp>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// A thin, storage-agnostic wrapper around a cosine-similarity usearch index:
// knows nothing about SQLite or the chunks schema, only integer keys and
// vectors. The graph can be serialised to a sidecar file (Save/Load) so a
// reopened engine skips the O(N) rebuild from SQLite; SQLite stays
// authoritative and can always reconstruct the index if the sidecar is
// missing or stale (see ChunkStore's constructor).
namespace retrieval_engine::detail {

class DenseIndex {
public:
    explicit DenseIndex(std::size_t dim);

    std::size_t dimensions() const { return dimensions_; }

    // Number of vectors currently indexed.
    std::size_t size() const { return index_.size(); }

    // Throws std::invalid_argument if `embedding.size() != dimensions()`,
    // or std::runtime_error on a usearch failure.
    void Add(std::uint64_t key, const std::vector<float>& embedding);

    // Returns (up to) `k` nearest (key, distance) pairs, nearest-first
    // (lower distance = more similar). Throws std::invalid_argument if
    // `query.size() != dimensions()`, or std::runtime_error on a usearch
    // failure.
    std::vector<std::pair<std::uint64_t, float>> Search(const std::vector<float>& query, std::size_t k) const;

    // Serialises the whole HNSW graph to `path` in usearch's native format.
    // Written to `path + ".tmp"` and renamed into place so a crash mid-write
    // cannot leave a torn sidecar. Throws std::runtime_error on failure.
    void Save(const std::string& path) const;

    // Replaces the in-memory graph with the one serialised at `path`.
    // Throws std::invalid_argument if the loaded graph's dimensionality does
    // not match this index, or std::runtime_error on an I/O or
    // deserialisation failure.
    void Load(const std::string& path);

    // Drops every vector, returning the index to its freshly-constructed
    // state -- used to discard a stale or unreadable sidecar before
    // rebuilding from SQLite.
    void Clear();

private:
    static unum::usearch::index_dense_t MakeIndex(std::size_t dim);

    std::size_t dimensions_;
    unum::usearch::index_dense_t index_;
};

}  // namespace retrieval_engine::detail
