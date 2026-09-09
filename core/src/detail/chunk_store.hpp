#pragma once

#include "retrieval_engine/retrieval_engine.hpp"  // DocumentInput, ChunkSearchResult, SearchExplanation

#include "chunk_repository.hpp"
#include "dense_index.hpp"
#include "rrf_fusion.hpp"  // FusionEntry

#include <cstddef>
#include <cstdint>
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

    // The agent memory layer: search_hybrid()'s fused score, discounted by
    // exponential recency decay and re-ranked by the result -- see
    // RetrievalEngine::search_memory()'s doc comment for the formula.
    std::vector<ChunkSearchResult> search_memory(const std::string& query_text, const std::vector<float>& query_vec,
                                                  std::size_t k, float decay_lambda) const;
    std::vector<SearchExplanation> search_memory_explained(const std::string& query_text,
                                                             const std::vector<float>& query_vec, std::size_t k,
                                                             float decay_lambda) const;

private:
    // One fused candidate plus its recency-decay outcome, produced by
    // FuseRankAndDecay() and shared by search_memory()/
    // search_memory_explained() so the fetch/fuse/decay/re-sort logic lives
    // once (mirroring FuseAndRank()'s role for search_hybrid()/
    // search_explained()).
    struct DecayedEntry {
        FusionEntry entry;
        std::int64_t created_at_unix_seconds;
        double age_seconds;
        double recency_factor;
        float decayed_score;
    };

    std::vector<DecayedEntry> FuseRankAndDecay(const std::string& query_text, const std::vector<float>& query_vec,
                                                std::size_t k, float decay_lambda) const;

    std::size_t dimensions_;
    ChunkRepository repository_;
    DenseIndex dense_index_;
};

}  // namespace retrieval_engine::detail
