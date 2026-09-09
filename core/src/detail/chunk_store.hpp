#pragma once

#include "retrieval_engine/retrieval_engine.hpp"  // DocumentInput, ChunkSearchResult, SearchExplanation

#include "chunk_repository.hpp"
#include "dense_index.hpp"

#include <cstddef>
#include <string>
#include <vector>

struct sqlite3;

// Orchestrates chunk storage and retrieval: composes a ChunkRepository
// (SQLite persistence: documents/chunks/chunks_fts) and a DenseIndex (the
// usearch cosine-similarity sidecar), and implements Reciprocal Rank Fusion
// (via rrf_fusion.hpp) to combine their results for search_hybrid()/
// search_explained(). Each collaborator owns one concern and is
// independently testable -- see their own headers -- ChunkStore's job is
// only coordination: dimension validation up front, and (per the "SQLite is
// authoritative, usearch is a rebuildable sidecar" architecture)
// populating DenseIndex only *after* ChunkRepository's SQLite transaction
// has committed, never before.
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
    std::size_t dimensions_;
    ChunkRepository repository_;
    DenseIndex dense_index_;
};

}  // namespace retrieval_engine::detail
