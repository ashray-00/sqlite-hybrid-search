#pragma once

#include "retrieval_engine/retrieval_engine.hpp"  // DocumentInput, ChunkSearchResult

#include <usearch/index_dense.hpp>

#include <cstddef>
#include <vector>

struct sqlite3;

// Stage 1's real feature: chunk storage + metadata in SQLite, embeddings in
// a cosine-similarity usearch index, kept in sync per the "SQLite is
// authoritative, usearch is a rebuildable sidecar" architecture in
// BUILD_PLAN.md section 5.
//
// Non-owning: does not open or close `db` -- RetrievalEngine::Impl owns the
// connection and outlives every store built on top of it.
namespace retrieval_engine::detail {

class ChunkStore {
public:
    ChunkStore(sqlite3* db, std::size_t dim);

    void add_documents(const std::vector<DocumentInput>& documents);
    std::vector<ChunkSearchResult> search_chunks(const std::vector<float>& query, std::size_t k) const;
    std::size_t chunk_count() const;

private:
    // Reloads `index_` from whatever is already in the `chunks` table --
    // the "rebuild the index from SQLite" recovery path from
    // BUILD_PLAN.md section 5. Run unconditionally in the constructor
    // (not just after corruption), since this store doesn't persist the
    // usearch index itself, only SQLite. A no-op on a fresh/empty database.
    void RebuildIndexFromSqlite();

    sqlite3* db_;
    std::size_t dimensions_;
    unum::usearch::index_dense_t index_;
};

}  // namespace retrieval_engine::detail
