#pragma once

#include "retrieval_engine/retrieval_engine.hpp"  // DocumentInput, ChunkSearchResult, SearchExplanation

#include <usearch/index_dense.hpp>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

struct sqlite3;

// Stage 1 + Stage 2's real feature: chunk storage + metadata in SQLite,
// dense embeddings in a cosine-similarity usearch index, sparse keyword
// search via a synced SQLite FTS5 index, and Reciprocal Rank Fusion (RRF)
// combining the two. Kept in sync per the "SQLite is authoritative, usearch
// is a rebuildable sidecar" architecture in BUILD_PLAN.md section 5 --
// note FTS5 doesn't need the same rebuild-on-open treatment as usearch: a
// native SQLite virtual table, its contents are already durable.
//
// Non-owning: does not open or close `db` -- RetrievalEngine::Impl owns the
// connection and outlives every store built on top of it.
namespace retrieval_engine::detail {

class ChunkStore {
public:
    ChunkStore(sqlite3* db, std::size_t dim);

    void add_documents(const std::vector<DocumentInput>& documents);
    std::size_t chunk_count() const;

    std::vector<ChunkSearchResult> search_dense(const std::vector<float>& query, std::size_t k) const;
    std::vector<ChunkSearchResult> search_sparse(const std::string& query_text, std::size_t k) const;
    std::vector<ChunkSearchResult> search_hybrid(const std::string& query_text, const std::vector<float>& query_vec,
                                                  std::size_t k) const;
    std::vector<SearchExplanation> search_explained(const std::string& query_text,
                                                      const std::vector<float>& query_vec, std::size_t k) const;

private:
    // Reloads `index_` from whatever is already in the `chunks` table --
    // the "rebuild the index from SQLite" recovery path from
    // BUILD_PLAN.md section 5. Run unconditionally in the constructor
    // (not just after corruption), since this store doesn't persist the
    // usearch index itself, only SQLite. A no-op on a fresh/empty database.
    void RebuildIndexFromSqlite();

    // One chunk's fusion state while search_hybrid()/search_explained() are
    // combining a dense and a sparse ranking via RRF (BUILD_PLAN.md Stage
    // 2): which ranking(s) it appeared in, its rank (1-based) in each, and
    // the resulting fused score. Shared internal representation for both
    // public methods so the fetch/fuse/sort/truncate logic lives once.
    struct FusionEntry {
        std::string document_id;
        std::size_t chunk_index = 0;
        std::string text;

        bool dense_present = false;
        float dense_distance = 0.0f;
        std::size_t dense_rank = 0;  // 1-based; 0 means "not present"

        bool sparse_present = false;
        float sparse_bm25_score = 0.0f;
        std::size_t sparse_rank = 0;  // 1-based; 0 means "not present"

        float fused_score = 0.0f;
    };

    // Runs search_dense()/search_sparse(), fuses them via RRF (k=60), and
    // returns the top `k` fused entries ordered highest-fused-score-first.
    std::vector<FusionEntry> FuseAndRank(const std::string& query_text, const std::vector<float>& query_vec,
                                          std::size_t k) const;

    sqlite3* db_;
    std::size_t dimensions_;
    unum::usearch::index_dense_t index_;
};

}  // namespace retrieval_engine::detail
