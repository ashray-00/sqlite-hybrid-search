#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace retrieval_engine {

// RetrievalEngine ties a usearch HNSW index (in-memory acceleration
// structure) to a SQLite database (source of truth for vectors/metadata),
// per the architecture in BUILD_PLAN.md section 5.
//
// Uses the Pimpl idiom so usearch/SQLite types never leak into this public
// header, matching the "thin C++ API" note in BUILD_PLAN.md section 5.
//
// Not thread-safe: concurrent calls into the same instance (from multiple
// threads) are not synchronized. Confine an instance to one thread, or add
// external locking, until a later stage addresses concurrency explicitly.
class RetrievalEngine {
public:
    // Opens (or creates) the SQLite database at `db_path` and prepares a
    // usearch index for dense vectors of dimensionality `dim`.
    // Throws std::invalid_argument if `dim` is zero, or std::runtime_error
    // if the SQLite database cannot be opened/prepared.
    RetrievalEngine(const std::string& db_path, std::size_t dim);
    ~RetrievalEngine();

    // Adds `vector` to the usearch index under key `id`, and records a
    // corresponding row in the dummy SQLite table so the two stores can be
    // cross-checked for consistency. Throws std::invalid_argument if
    // `vector.size() != dim`, or std::runtime_error on a usearch/SQLite
    // failure.
    void add_vector(std::uint64_t id, const std::vector<float>& vector);

    // Returns the ids of (up to) the `k` nearest neighbours of `query`,
    // ordered nearest-first. Throws std::invalid_argument if
    // `query.size() != dim`, or std::runtime_error on a usearch failure.
    std::vector<std::uint64_t> search(const std::vector<float>& query, std::size_t k) const;

    // Number of rows currently present in the dummy SQLite table -- used to
    // verify that SQLite bookkeeping stays in sync with the usearch index.
    std::size_t dummy_table_row_count() const;

    RetrievalEngine(const RetrievalEngine&) = delete;
    RetrievalEngine& operator=(const RetrievalEngine&) = delete;

    // Movable: ownership of the underlying index/connection transfers
    // cleanly (defined out-of-line where Impl is a complete type).
    RetrievalEngine(RetrievalEngine&&) noexcept;
    RetrievalEngine& operator=(RetrievalEngine&&) noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace retrieval_engine
