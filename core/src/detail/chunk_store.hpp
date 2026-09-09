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
// independently testable. ChunkStore's job is only coordination: dimension
// validation up front, and populating DenseIndex only *after*
// ChunkRepository's SQLite transaction has committed, never before (SQLite
// is authoritative; the usearch index is a rebuildable sidecar).
//
// Non-owning: does not open or close `db` -- RetrievalEngine::Impl owns the
// connection and outlives every store built on top of it.
namespace retrieval_engine::detail {

class ChunkStore {
public:
    // `index_sidecar_path` is where the usearch graph is persisted between
    // opens (typically "<db_path>.usearch"). An empty string disables
    // sidecar persistence and forces a rebuild from SQLite on every open
    // (used for in-memory databases, which have no stable path to sit
    // beside).
    ChunkStore(sqlite3* db, std::size_t dim, std::string index_sidecar_path);

    void add_documents(const std::vector<DocumentInput>& documents);
    std::size_t chunk_count() const;

    // True when this instance loaded its dense index from the sidecar file
    // on construction; false when it had to rebuild from SQLite (no
    // sidecar, or a stale/unreadable one). Purely diagnostic.
    bool loaded_index_from_sidecar() const { return loaded_index_from_sidecar_; }

    std::vector<ChunkSearchResult> search_dense(const std::vector<float>& query, std::size_t k) const;
    std::vector<ChunkSearchResult> search_sparse(const std::string& query_text, std::size_t k) const;
    std::vector<ChunkSearchResult> search_hybrid(const std::string& query_text, const std::vector<float>& query_vec,
                                                 std::size_t k) const;
    std::vector<SearchExplanation> search_explained(const std::string& query_text, const std::vector<float>& query_vec,
                                                    std::size_t k) const;

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
    // once (mirroring Fuse()'s role for search_hybrid()/search_explained()).
    struct DecayedEntry {
        FusionEntry entry;
        std::int64_t created_at_unix_seconds;
        double age_seconds;
        double recency_factor;
        float decayed_score;
    };

    // Runs search_dense()+search_sparse()+RrfFuse() -- the one fuse step
    // every search_*() method builds on -- so it's written once instead of
    // once per caller (search_hybrid(), search_explained(), and
    // FuseRankAndDecay() all used to call RrfFuse() independently).
    std::vector<FusionEntry> Fuse(const std::string& query_text, const std::vector<float>& query_vec,
                                  std::size_t k) const;

    // Builds the shared (non-decay) fields of a SearchExplanation from one
    // fused entry at the given 1-based rank -- the common core of
    // search_explained() and search_memory_explained(), which differ only
    // in which ranking (fused-order vs. decayed-order) supplies `rank` and
    // whether the recency-decay fields get filled in afterwards.
    static SearchExplanation ExplanationFromFusionEntry(const FusionEntry& entry, std::size_t rank);

    std::vector<DecayedEntry> FuseRankAndDecay(const std::string& query_text, const std::vector<float>& query_vec,
                                               std::size_t k, float decay_lambda) const;

    // Rebuilds the dense index from every chunk in SQLite and, if a sidecar
    // path is configured, writes the result so the next open can skip this.
    void RebuildIndexFromRepositoryAndPersist();

    // Persists the dense index to the sidecar file; a no-op when sidecar
    // persistence is disabled (empty path).
    void PersistIndexSidecar() const;

    std::size_t dimensions_;
    std::string index_sidecar_path_;
    bool loaded_index_from_sidecar_ = false;
    ChunkRepository repository_;
    DenseIndex dense_index_;
};

}  // namespace retrieval_engine::detail
