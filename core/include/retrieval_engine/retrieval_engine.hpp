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

// A document for the raw-text ingestion path (add_text()): like
// DocumentInput, but with no caller-supplied embedding -- the engine's
// built-in embedding model (see load_embedding_model()) computes the
// vector from `text`, and the whole text becomes a single chunk. Used only
// when a model has been attached; the caller-supplied-vector path
// (DocumentInput / add_documents()) is unaffected and still available.
struct TextDocumentInput {
    std::string document_id;
    std::string text;
    std::string metadata;
    // Same sentinel semantics as DocumentChunkInput::created_at_unix_seconds:
    // 0 means "use the current wall-clock time at insertion".
    std::int64_t created_at_unix_seconds = 0;
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

    bool sparse_present;      // true if this chunk was among search_sparse()'s results
    float sparse_bm25_score;  // raw FTS5 bm25() score, lower = more relevant (meaningful only if sparse_present)
    std::size_t sparse_rank;  // 1-based rank within the sparse-only result list (0 if sparse_present is false)

    float fused_score;       // Reciprocal Rank Fusion score (k=60), higher = more relevant
    std::size_t final_rank;  // 1-based rank within the fused/hybrid result list

    // Recency-decay fields: populated by search_memory_explained(), left at
    // these defaults by search_explained() (which has no notion of decay).
    // See RetrievalEngine::search_memory()'s doc comment for the formula.
    std::int64_t created_at_unix_seconds = 0;  // this chunk's stored timestamp
    double age_seconds = 0.0;                  // time between created_at and the query, in seconds
    double recency_factor = 1.0;               // e^(-decay_lambda * age_seconds / 86400); 1.0 = no decay
    float decayed_score = 0.0f;                // fused_score * recency_factor
};

// RetrievalEngine ties a usearch HNSW index (an in-memory acceleration
// structure) to a SQLite database (the source of truth for vectors and
// metadata; the index is a rebuildable sidecar).
//
// Uses the Pimpl idiom so usearch/SQLite types never leak into this public
// header.
//
// Not thread-safe: concurrent calls into the same instance are not
// synchronized. Confine an instance to one thread, or add external locking.
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
    // architecture.
    //
    // Throws std::invalid_argument if any chunk's embedding size doesn't
    // match `dim`, or std::runtime_error on a usearch/SQLite failure.
    void add_documents(const std::vector<DocumentInput>& documents);

    // Number of chunk rows currently present in SQLite -- used to verify
    // ingestion persisted everything add_documents() was given.
    std::size_t chunk_count() const;

    // True when this instance loaded its dense index from the on-disk
    // "<db_path>.usearch" sidecar at construction (the fast path), false
    // when it rebuilt the index from SQLite (no sidecar yet, or one that
    // was stale or unreadable). Diagnostic only -- results are identical
    // either way.
    bool loaded_index_from_sidecar() const;

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
    std::vector<SearchExplanation> search_explained(const std::string& query_text, const std::vector<float>& query_vec,
                                                    std::size_t k) const;

    // The agent memory layer: applies exponential recency decay on top of
    // search_hybrid()'s fused score and re-ranks by the result, so a
    // recent, relevant memory can outrank an older one even when the older
    // one has a slightly higher raw hybrid score --
    //     decayed_score = base_score * e^(-decay_lambda * age_seconds / 86400)
    // where base_score is search_hybrid()'s fused RRF score, age_seconds is
    // the time (in seconds) between each chunk's stored created_at (see
    // DocumentChunkInput::created_at_unix_seconds) and now (normalized to
    // days by the /86400, so decay_lambda values around 0.05-0.5 correspond
    // to meaningful day-scale half-lives), and decay_lambda is lambda: 0
    // disables decay entirely (identical ranking to search_hybrid());
    // larger values discount older memories more aggressively, clamped to
    // never fully zero out an old memory (see ComputeRecencyFactor's own
    // doc comment for the exact floor). `score` in each result holds
    // decayed_score. Throws under the same conditions as search_hybrid().
    std::vector<ChunkSearchResult> search_memory(const std::string& query_text, const std::vector<float>& query_vec,
                                                 std::size_t k, float decay_lambda) const;

    // Like search_memory(), but returns the full per-result score breakdown
    // (as search_explained(), plus the recency-decay fields: timestamp,
    // age_seconds, recency_factor, decayed_score) -- for debugging and
    // tuning the decay parameter. Throws under the same conditions as
    // search_memory().
    std::vector<SearchExplanation> search_memory_explained(const std::string& query_text,
                                                           const std::vector<float>& query_vec, std::size_t k,
                                                           float decay_lambda) const;

    // --- Built-in local embedding model ---------------------------------
    // The zero-setup "just give it text" path.
    // With a model attached, callers pass raw strings and the engine
    // computes the vectors itself, instead of supplying floats from Python
    // or an external runtime. All of the above (add_documents() /
    // search_dense() / search_hybrid() / search_memory() with
    // caller-supplied vectors) keeps working unchanged whether or not a
    // model is attached.

    // Loads a local embedding model from `model_path` (e.g. an ONNX or
    // GGUF file for all-MiniLM-L6-v2 / nomic-embed-text) and attaches it to
    // this engine, enabling embed(), add_text() and search_text(). The
    // model's output dimensionality must equal the `dim` passed to the
    // constructor. Throws std::runtime_error if the file cannot be read or
    // parsed, or std::invalid_argument if its output dimension != dim.
    void load_embedding_model(const std::string& model_path);

    // True once load_embedding_model() has successfully attached a model.
    bool has_embedding_model() const;

    // Output dimensionality of the attached model (always equal to the
    // engine's `dim`). Throws std::logic_error if no model is attached.
    std::size_t embedding_dim() const;

    // Computes an embedding for `text` with the attached model. The result
    // always has exactly embedding_dim() elements and is never empty for
    // non-empty input. Throws std::logic_error if no model is attached.
    std::vector<float> embed(const std::string& text) const;

    // Raw-text ingestion: embeds each document's `text` with the attached
    // model and stores it as a single-chunk document -- equivalent to
    // embedding each text yourself and calling add_documents(). Throws
    // std::logic_error if no model is attached, or std::runtime_error on a
    // storage failure.
    void add_text(const std::vector<TextDocumentInput>& documents);

    // Raw-text query: embeds `query_text` with the attached model and runs
    // the memory search (search_memory()) with that vector -- the same text
    // also drives the sparse/BM25 side. `decay_lambda` is passed straight
    // through to search_memory() (0, the default, disables recency decay,
    // making this exactly a hybrid search). `score` in each result holds
    // search_memory()'s decayed fused score (higher = more relevant).
    // Throws std::logic_error if no model is attached.
    std::vector<ChunkSearchResult> search_text(const std::string& query_text, std::size_t k,
                                               float decay_lambda = 0.0f) const;

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
