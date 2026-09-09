#pragma once

#include <usearch/index_dense.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

struct sqlite3;

// Stage 0's scaffolding: proves SQLite + usearch are linked and stay
// consistent with each other. Deliberately kept separate from ChunkStore
// (Stage 1's real feature) so it can be deleted cleanly in a future
// cleanup stage without touching the real retrieval path -- see
// docs/DECISIONS.md, "kept Stage 0's dummy_vectors/... untouched".
//
// Non-owning: does not open or close `db` -- RetrievalEngine::Impl owns the
// connection and outlives every store built on top of it.
namespace retrieval_engine::detail {

class DummyVectorStore {
public:
    DummyVectorStore(sqlite3* db, std::size_t dim);

    void add_vector(std::uint64_t id, const std::vector<float>& vector);
    std::vector<std::uint64_t> search(const std::vector<float>& query, std::size_t k) const;
    std::size_t row_count() const;

private:
    sqlite3* db_;
    std::size_t dimensions_;
    unum::usearch::index_dense_t index_;
};

}  // namespace retrieval_engine::detail
