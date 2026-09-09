#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace retrieval_engine {

// One chunk of a document, ready to be ingested via add_documents(): its
// text, a caller-supplied embedding for that text (Stage 1 scope: "embeddings
// supplied by caller for now", per BUILD_PLAN.md Stage 1), and its position
// within the source document (as produced by chunk_text()).
struct DocumentChunkInput {
    std::string text;
    std::vector<float> embedding;
    std::size_t start_token;
    std::size_t end_token;
};

// A document to ingest: a caller-assigned id, opaque caller-defined metadata
// (Stage 1 does not interpret it -- stored and returned as-is), and its
// already-chunked, already-embedded content.
struct DocumentInput {
    std::string document_id;
    std::string metadata;
    std::vector<DocumentChunkInput> chunks;
};

// One chunk returned by search_chunks(), with enough context to trace it
// back to its source document without a further lookup.
struct ChunkSearchResult {
    std::string document_id;
    std::size_t chunk_index;  // position within that document's chunk list, 0-based
    std::string text;
    float distance;  // raw index distance for the chunk-search metric (lower = more similar)
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

    // Stage 1 (BUILD_PLAN.md): stores each document's chunks + metadata in
    // SQLite (the authoritative store) and adds each chunk's embedding to a
    // dedicated chunk-search usearch index (cosine similarity), keyed by a
    // chunk id this call assigns internally. Persists across re-opening the
    // same `db_path`: a freshly-constructed RetrievalEngine reloads existing
    // chunk rows from SQLite and rebuilds the chunk-search index from them,
    // per the "SQLite is authoritative, usearch is a rebuildable sidecar"
    // architecture in BUILD_PLAN.md section 5.
    //
    // Throws std::invalid_argument if any chunk's embedding size doesn't
    // match `dim`, or std::runtime_error on a usearch/SQLite failure.
    void add_documents(const std::vector<DocumentInput>& documents);

    // Returns (up to) the `k` nearest chunks to `query` by cosine similarity,
    // ordered nearest-first. Throws std::invalid_argument if
    // `query.size() != dim`, or std::runtime_error on a usearch failure.
    std::vector<ChunkSearchResult> search_chunks(const std::vector<float>& query, std::size_t k) const;

    // Number of chunk rows currently present in SQLite -- used to verify
    // ingestion persisted everything add_documents() was given.
    std::size_t chunk_count() const;

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
