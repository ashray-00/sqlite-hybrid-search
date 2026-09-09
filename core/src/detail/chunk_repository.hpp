#pragma once

#include "retrieval_engine/retrieval_engine.hpp"  // DocumentInput, ChunkSearchResult

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

struct sqlite3;
struct sqlite3_stmt;

// SQLite persistence for chunks: the `documents`/`chunks` tables and the
// `chunks_fts` FTS5 sparse index. All three are grouped in one class
// because all three are plain, durable, transactional SQLite constructs
// written together inside a single BEGIN/COMMIT in add_documents() -- unlike
// the usearch dense index (see dense_index.hpp), which is not transactional
// and must only be populated once that transaction has committed. See
// docs/DECISIONS.md for the SOLID-split rationale.
//
// Non-owning: does not open or close `db` -- RetrievalEngine::Impl owns the
// connection and outlives every store built on top of it.
namespace retrieval_engine::detail {

class ChunkRepository {
public:
    ChunkRepository(sqlite3* db, std::size_t dim);

    // Writes each document's chunks + metadata into `documents`/`chunks`,
    // and each chunk's text into `chunks_fts`, inside one SQLite
    // transaction (rolled back atomically if anything fails partway
    // through). Returns each newly-inserted chunk's (chunk_id, embedding)
    // pair -- pointers into `documents`, valid for as long as it is -- so
    // the caller can populate a DenseIndex once this call returns
    // successfully. Throws std::runtime_error on a SQLite failure.
    std::vector<std::pair<std::uint64_t, const std::vector<float>*>> add_documents(
        const std::vector<DocumentInput>& documents);

    std::size_t chunk_count() const;

    // Resolves a chunk_id (e.g. a DenseIndex search hit) back to its
    // document_id/chunk_index/text. Throws std::runtime_error if no such
    // row exists (index and store have desynced).
    ChunkSearchResult Resolve(std::uint64_t chunk_id) const;

    // Streams every currently-stored chunk's (chunk_id, embedding) to
    // `visitor`, in no particular order -- for ChunkStore's constructor to
    // rebuild a DenseIndex from persisted state (the rebuildable-sidecar
    // architecture). A no-op on a fresh/empty
    // database. Throws std::runtime_error if a stored embedding's size
    // doesn't match this repository's dimensionality (this database may
    // have been created with a different `dim`), or on any other SQLite
    // failure.
    void ForEachChunk(
        const std::function<void(std::uint64_t chunk_id, const std::vector<float>& embedding)>& visitor) const;

    // Sparse keyword search over chunk text via SQLite FTS5's bm25()
    // ranking function, ordered most-relevant-first. `query_text` is
    // sanitized (see fts5_query.hpp) so raw, untrusted search text can
    // never be interpreted as FTS5 query syntax. Only chunks matching are
    // returned, so the result may have fewer than `k` entries -- or none.
    // Throws std::runtime_error on a SQLite failure.
    std::vector<ChunkSearchResult> SearchSparse(const std::string& query_text, std::size_t k) const;

    // Looks up a chunk's stored `created_at` (Unix epoch seconds) by
    // (document_id, chunk_index) -- the same identity key used throughout
    // the RRF fusion pipeline (see rrf_fusion.hpp), for the agent memory
    // layer's recency decay (RetrievalEngine::search_memory()). Throws
    // std::runtime_error if no such chunk exists.
    std::int64_t GetCreatedAt(const std::string& document_id, std::size_t chunk_index) const;

private:
    // The three per-row inserts add_documents() performs, factored out so
    // its own loop reads as "insert the document row, then insert each
    // chunk's row and FTS row" rather than interleaving all three
    // statements' bind/step calls inline. Each takes an already-prepared,
    // caller-owned statement (reset and rebound here) so the statements
    // themselves are still prepared once per batch, not once per row.
    void InsertDocumentRow(sqlite3_stmt* statement, const DocumentInput& document) const;
    std::uint64_t InsertChunkRow(sqlite3_stmt* statement, const std::string& document_id, std::size_t chunk_index,
                                 const DocumentChunkInput& chunk) const;
    void InsertChunkFtsRow(sqlite3_stmt* statement, std::uint64_t chunk_id, const std::string& text) const;

    sqlite3* db_;
    std::size_t dimensions_;
};

}  // namespace retrieval_engine::detail
