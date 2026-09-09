#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace retrieval_engine {

// One chunk of a document, ready to be ingested via add_documents(): its
// text, a caller-supplied embedding for that text (embeddings are always
// supplied by the caller -- this engine does not compute them), and its
// position within the source document (as produced by chunk_text()).
struct DocumentChunkInput {
    std::string text;
    std::vector<float> embedding;
    std::size_t start_token;
    std::size_t end_token;
    // Unix timestamp (seconds since epoch) this chunk should be recorded as
    // created at, for recency-decay scoring (search_memory()). 0 is a
    // sentinel meaning "use the current wall-clock time at insertion" --
    // a real timestamp is never legitimately exactly the Unix epoch.
    std::int64_t created_at_unix_seconds = 0;
};

// A document to ingest: a caller-assigned id, opaque caller-defined metadata
// (not interpreted -- stored and returned as-is), and its already-chunked,
// already-embedded content.
struct DocumentInput {
    std::string document_id;
    std::string metadata;
    std::vector<DocumentChunkInput> chunks;
};

// One chunk returned by search_dense()/search_sparse()/search_hybrid(),
// with enough context to trace it back to its source document without a
// further lookup.
struct ChunkSearchResult {
    std::string document_id;
    std::size_t chunk_index;  // position within that document's chunk list, 0-based
    std::string text;
    // Meaning and "better" direction depend on which search produced this
    // result: search_dense's raw cosine distance and search_sparse's raw
    // bm25 score are both lower-is-better; search_hybrid's fused RRF score
    // is higher-is-better. Named score, not distance, because "distance"
    // would wrongly imply lower-is-better for every case. See each method's
    // doc comment for which convention applies.
    float score;
};

// One result's full per-ranking score breakdown, returned by
// search_explained(): the dense score/rank, the sparse score/rank, the
// fused score, and the final rank.
struct SearchExplanation {
    std::string document_id;
    std::size_t chunk_index;  // position within that document's chunk list, 0-based
    std::string text;

    bool dense_present;      // true if this chunk was among search_dense()'s results
    float dense_distance;    // raw cosine distance from search_dense() (meaningful only if dense_present)
    std::size_t dense_rank;  // 1-based rank within the dense-only result list (0 if dense_present is false)

    bool sparse_present;       // true if this chunk was among search_sparse()'s results
    float sparse_bm25_score;   // raw FTS5 bm25() score, lower = more relevant (meaningful only if sparse_present)
    std::size_t sparse_rank;   // 1-based rank within the sparse-only result list (0 if sparse_present is false)

    float fused_score;       // Reciprocal Rank Fusion score (k=60), higher = more relevant
    std::size_t final_rank;  // 1-based rank within the fused/hybrid result list

    // Recency-decay fields: populated by search_memory_explained(), left at
    // these defaults by search_explained() (which has no notion of decay).
    // See RetrievalEngine::search_memory()'s doc comment for the formula.
    std::int64_t created_at_unix_seconds = 0;  // this chunk's stored timestamp
    double age_seconds = 0.0;                  // time between created_at and the query, in seconds
    double recency_factor = 1.0;               // exp(-recency_weight * age_seconds); 1.0 = no decay
    float decayed_score = 0.0f;                // fused_score * recency_factor
};

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

    // Stores each document's chunks + metadata in SQLite (the authoritative
    // store) and adds each chunk's embedding to a dedicated chunk-search
    // usearch index (cosine similarity), keyed by a
    // chunk id this call assigns internally. Persists across re-opening the
    // same `db_path`: a freshly-constructed RetrievalEngine reloads existing
    // chunk rows from SQLite and rebuilds the chunk-search index from them,
    // per the "SQLite is authoritative, usearch is a rebuildable sidecar"
    // architecture in BUILD_PLAN.md section 5.
    //
    // Throws std::invalid_argument if any chunk's embedding size doesn't
    // match `dim`, or std::runtime_error on a usearch/SQLite failure.
    void add_documents(const std::vector<DocumentInput>& documents);

    // Number of chunk rows currently present in SQLite -- used to verify
    // ingestion persisted everything add_documents() was given.
    std::size_t chunk_count() const;

    // Dense-only search: returns (up to) the `k` nearest chunks to `query`
    // by cosine similarity, ordered nearest-first. `score` in each result is
    // the raw cosine distance (lower = more similar). Throws
    // std::invalid_argument if `query.size() != dim`, or std::runtime_error
    // on a usearch failure.
    std::vector<ChunkSearchResult> search_dense(const std::vector<float>& query, std::size_t k) const;

    // Sparse keyword search over chunk text via SQLite FTS5's bm25()
    // ranking function. `score` in each result holds the raw bm25 score
    // (lower = more relevant, FTS5's convention). Only chunks matching
    // `query_text` are returned, so the result may have fewer than `k`
    // entries -- or none. Throws std::runtime_error on an invalid FTS5
    // query (e.g. unbalanced quotes) or other SQLite failure.
    std::vector<ChunkSearchResult> search_sparse(const std::string& query_text, std::size_t k) const;

    // Combines search_dense() and search_sparse() via Reciprocal Rank Fusion
    // (RRF, k=60): each chunk's fused score is the sum, over whichever of
    // the two rankings it appears in, of 1 / (60 + rank-in-that-ranking) (0
    // for a ranking it's absent from), ordered highest-fused-score-first.
    // `score` in each result holds this fused score (higher = more relevant
    // -- unlike search_dense()'s and search_sparse()'s own conventions).
    // Throws under the same conditions as search_dense() and
    // search_sparse().
    std::vector<ChunkSearchResult> search_hybrid(const std::string& query_text, const std::vector<float>& query_vec,
                                                  std::size_t k) const;

    // Like search_hybrid(), but returns the full per-result score breakdown
    // (dense distance/rank, sparse bm25/rank, fused score, final rank)
    // instead of just the fused ranking -- for debugging and tuning
    // relevance. Throws under the same conditions as search_hybrid().
    std::vector<SearchExplanation> search_explained(const std::string& query_text,
                                                      const std::vector<float>& query_vec, std::size_t k) const;

    // The agent memory layer: applies exponential recency decay on top of
    // search_hybrid()'s fused score and re-ranks by the result, so a
    // recent, relevant memory can outrank an older one even when the older
    // one has a slightly higher raw hybrid score --
    //     decayed_score = base_score * e^(-recency_weight * age_seconds)
    // where base_score is search_hybrid()'s fused RRF score, age_seconds is
    // the time (in seconds) between each chunk's stored created_at (see
    // DocumentChunkInput::created_at_unix_seconds) and now, and
    // recency_weight is lambda: 0 disables decay entirely (identical
    // ranking to search_hybrid()); larger values discount older memories
    // more aggressively. `score` in each result holds decayed_score.
    // Throws under the same conditions as search_hybrid().
    std::vector<ChunkSearchResult> search_memory(const std::string& query_text, const std::vector<float>& query_vec,
                                                  std::size_t k, float recency_weight) const;

    // Like search_memory(), but returns the full per-result score breakdown
    // (as search_explained(), plus the recency-decay fields: timestamp,
    // age_seconds, recency_factor, decayed_score) -- for debugging and
    // tuning recency weighting. Throws under the same conditions as
    // search_memory().
    std::vector<SearchExplanation> search_memory_explained(const std::string& query_text,
                                                             const std::vector<float>& query_vec, std::size_t k,
                                                             float recency_weight) const;

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
