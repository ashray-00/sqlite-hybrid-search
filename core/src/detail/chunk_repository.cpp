#include "chunk_repository.hpp"

#include "fts5_query.hpp"
#include "sqlite_util.hpp"
#include "time_util.hpp"

#include <sqlite3.h>

#include <stdexcept>

namespace retrieval_engine::detail {

ChunkRepository::ChunkRepository(sqlite3* db, std::size_t dim) : db_(db), dimensions_(dim) {
    // `chunk_id` is a plain SQLite rowid alias (INTEGER PRIMARY KEY, no
    // AUTOINCREMENT -- this store never deletes rows, so plain monotonic
    // reuse-after-max-rowid semantics are irrelevant) so it can double as
    // the usearch vector key with no separate id table: last_insert_rowid()
    // after each chunk insert *is* the key add_documents() hands to
    // DenseIndex, and chunks_fts's rowid, keeping all three trivially
    // joinable/lookup-able by the same integer.
    ThrowIfSqliteError(sqlite3_exec(db_,
                                     "CREATE TABLE IF NOT EXISTS documents ("
                                     "  document_id TEXT PRIMARY KEY,"
                                     "  metadata TEXT NOT NULL"
                                     ");",
                                     nullptr, nullptr, nullptr),
                        db_, "ChunkRepository: failed to create documents table");

    // `created_at`/`last_accessed_at` are Unix epoch seconds (UTC --
    // std::time()-based, see time_util.hpp), for the agent memory layer's
    // recency decay (RetrievalEngine::search_memory()). `last_accessed_at`
    // is initialized to `created_at` at insertion and stored for future use
    // (e.g. access-pattern-aware forgetting policies) but is not currently
    // refreshed on retrieval -- search_memory() decays against `created_at`
    // only. Deliberately scoped this way: refreshing it on every search
    // would make search_memory() a non-idempotent "read" (two identical
    // calls could return different results), which needs its own explicit
    // design, not a side effect of this stage. See docs/DECISIONS.md.
    ThrowIfSqliteError(sqlite3_exec(db_,
                                     "CREATE TABLE IF NOT EXISTS chunks ("
                                     "  chunk_id INTEGER PRIMARY KEY,"
                                     "  document_id TEXT NOT NULL REFERENCES documents(document_id),"
                                     "  chunk_index INTEGER NOT NULL,"
                                     "  text TEXT NOT NULL,"
                                     "  start_token INTEGER NOT NULL,"
                                     "  end_token INTEGER NOT NULL,"
                                     "  embedding BLOB NOT NULL,"
                                     "  created_at INTEGER NOT NULL,"
                                     "  last_accessed_at INTEGER NOT NULL"
                                     ");",
                                     nullptr, nullptr, nullptr),
                        db_, "ChunkRepository: failed to create chunks table");

    // The sparse index. A plain (not "external content") FTS5 table,
    // populated explicitly with `rowid` set to the matching chunk_id -- see
    // add_documents() -- rather than sourcing content from `chunks` (SQLite's
    // recommended pattern for indexing an existing column without
    // duplicating it), because that mode requires triggers to stay in sync
    // across every write path, and add_documents() is currently the only
    // one. Costs one extra copy of each chunk's text; simpler to reason
    // about for now.
    ThrowIfSqliteError(
        sqlite3_exec(db_, "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(text);", nullptr, nullptr,
                     nullptr),
        db_, "ChunkRepository: failed to create chunks_fts table");
}

std::vector<std::pair<std::uint64_t, const std::vector<float>*>> ChunkRepository::add_documents(
    const std::vector<DocumentInput>& documents) {
    std::vector<std::pair<std::uint64_t, const std::vector<float>*>> pending_index_adds;

    // One transaction for the whole batch: ingesting is dominated by
    // per-statement fsync overhead if each row auto-commits individually,
    // which would make ingesting large corpora (tens of thousands of
    // chunks) far slower than it needs to be. If anything below throws,
    // the catch block rolls back so SQLite never ends up with a
    // partially-committed batch (across documents, chunks, *and*
    // chunks_fts -- all three are plain SQLite tables inside this one
    // transaction).
    ThrowIfSqliteError(sqlite3_exec(db_, "BEGIN;", nullptr, nullptr, nullptr), db_,
                        "ChunkRepository::add_documents: failed to begin transaction");

    try {
        SqliteStatement insert_document(db_, "INSERT INTO documents (document_id, metadata) VALUES (?, ?);");
        SqliteStatement insert_chunk(db_,
                                      "INSERT INTO chunks (document_id, chunk_index, text, start_token, end_token, "
                                      "embedding, created_at, last_accessed_at) "
                                      "VALUES (?, ?, ?, ?, ?, ?, ?, ?);");
        SqliteStatement insert_chunk_fts(db_, "INSERT INTO chunks_fts(rowid, text) VALUES (?, ?);");

        for (const auto& document : documents) {
            sqlite3_reset(insert_document.get());
            sqlite3_bind_text(insert_document.get(), 1, document.document_id.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(insert_document.get(), 2, document.metadata.c_str(), -1, SQLITE_TRANSIENT);
            if (sqlite3_step(insert_document.get()) != SQLITE_DONE) {
                throw std::runtime_error(std::string("ChunkRepository::add_documents: failed to insert document '") +
                                          document.document_id + "': " + sqlite3_errmsg(db_));
            }

            for (std::size_t chunk_position = 0; chunk_position < document.chunks.size(); ++chunk_position) {
                const DocumentChunkInput& chunk = document.chunks[chunk_position];

                sqlite3_reset(insert_chunk.get());
                sqlite3_bind_text(insert_chunk.get(), 1, document.document_id.c_str(), -1, SQLITE_TRANSIENT);
                sqlite3_bind_int64(insert_chunk.get(), 2, static_cast<sqlite3_int64>(chunk_position));
                sqlite3_bind_text(insert_chunk.get(), 3, chunk.text.c_str(), -1, SQLITE_TRANSIENT);
                sqlite3_bind_int64(insert_chunk.get(), 4, static_cast<sqlite3_int64>(chunk.start_token));
                sqlite3_bind_int64(insert_chunk.get(), 5, static_cast<sqlite3_int64>(chunk.end_token));
                sqlite3_bind_blob(insert_chunk.get(), 6, chunk.embedding.data(),
                                   static_cast<int>(chunk.embedding.size() * sizeof(float)), SQLITE_TRANSIENT);
                // 0 is DocumentChunkInput::created_at_unix_seconds's sentinel
                // for "use the current time" -- a real timestamp is never
                // legitimately exactly the Unix epoch.
                const std::int64_t created_at =
                    chunk.created_at_unix_seconds != 0 ? chunk.created_at_unix_seconds : CurrentUnixTimeSeconds();
                sqlite3_bind_int64(insert_chunk.get(), 7, static_cast<sqlite3_int64>(created_at));
                sqlite3_bind_int64(insert_chunk.get(), 8, static_cast<sqlite3_int64>(created_at));

                if (sqlite3_step(insert_chunk.get()) != SQLITE_DONE) {
                    throw std::runtime_error(
                        std::string("ChunkRepository::add_documents: failed to insert chunk: ") +
                        sqlite3_errmsg(db_));
                }

                // The chunk's SQLite rowid *is* its usearch key: assigned by
                // SQLite (not chosen by us), guaranteed unique, and cheap to
                // map back (WHERE chunk_id = ?) in Resolve(). Reused as-is
                // for chunks_fts's rowid, so the two stay trivially
                // joinable.
                const auto chunk_id = static_cast<std::uint64_t>(sqlite3_last_insert_rowid(db_));
                pending_index_adds.emplace_back(chunk_id, &chunk.embedding);

                sqlite3_reset(insert_chunk_fts.get());
                sqlite3_bind_int64(insert_chunk_fts.get(), 1, static_cast<sqlite3_int64>(chunk_id));
                sqlite3_bind_text(insert_chunk_fts.get(), 2, chunk.text.c_str(), -1, SQLITE_TRANSIENT);
                if (sqlite3_step(insert_chunk_fts.get()) != SQLITE_DONE) {
                    throw std::runtime_error(std::string("ChunkRepository::add_documents: failed to insert into "
                                                          "chunks_fts: ") +
                                              sqlite3_errmsg(db_));
                }
            }
        }
    } catch (...) {
        sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
        throw;
    }

    ThrowIfSqliteError(sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, nullptr), db_,
                        "ChunkRepository::add_documents: failed to commit transaction");

    return pending_index_adds;
}

std::size_t ChunkRepository::chunk_count() const {
    SqliteStatement statement(db_, "SELECT COUNT(*) FROM chunks;");
    if (sqlite3_step(statement.get()) != SQLITE_ROW)
        throw std::runtime_error("ChunkRepository::chunk_count: failed to execute row count query");
    return static_cast<std::size_t>(sqlite3_column_int64(statement.get(), 0));
}

ChunkSearchResult ChunkRepository::Resolve(std::uint64_t chunk_id) const {
    SqliteStatement statement(db_, "SELECT document_id, chunk_index, text FROM chunks WHERE chunk_id = ?;");
    sqlite3_bind_int64(statement.get(), 1, static_cast<sqlite3_int64>(chunk_id));

    if (sqlite3_step(statement.get()) != SQLITE_ROW) {
        throw std::runtime_error(
            "ChunkRepository::Resolve: no chunk row for this chunk_id (index and store have desynced)");
    }

    ChunkSearchResult result;
    result.document_id = ColumnText(statement.get(), 0);
    result.chunk_index = static_cast<std::size_t>(sqlite3_column_int64(statement.get(), 1));
    result.text = ColumnText(statement.get(), 2);
    result.score = 0.0f;  // caller's job to fill in -- see DenseIndex/SearchSparse callers
    return result;
}

void ChunkRepository::ForEachChunk(
    const std::function<void(std::uint64_t chunk_id, const std::vector<float>& embedding)>& visitor) const {
    SqliteStatement statement(db_, "SELECT chunk_id, embedding FROM chunks;");
    const std::size_t expected_bytes = dimensions_ * sizeof(float);

    int step_rc;
    while ((step_rc = sqlite3_step(statement.get())) == SQLITE_ROW) {
        const auto chunk_id = static_cast<std::uint64_t>(sqlite3_column_int64(statement.get(), 0));
        const void* embedding_blob = sqlite3_column_blob(statement.get(), 1);
        const auto blob_bytes = static_cast<std::size_t>(sqlite3_column_bytes(statement.get(), 1));

        if (blob_bytes != expected_bytes) {
            throw std::runtime_error(
                "ChunkRepository::ForEachChunk: a stored chunk embedding's size does not match this "
                "repository's dimensionality (this database may have been created with a different `dim`)");
        }

        const auto* floats = static_cast<const float*>(embedding_blob);
        visitor(chunk_id, std::vector<float>(floats, floats + dimensions_));
    }

    if (step_rc != SQLITE_DONE) {
        throw std::runtime_error(std::string("ChunkRepository::ForEachChunk: failed while reading the chunks "
                                              "table: ") +
                                  sqlite3_errmsg(db_));
    }
}

std::vector<ChunkSearchResult> ChunkRepository::SearchSparse(const std::string& query_text, std::size_t k) const {
    const std::string match_query = BuildSafeFts5MatchQuery(query_text);
    if (match_query.empty()) return {};  // no search terms -- nothing can match

    SqliteStatement statement(db_,
                               "SELECT c.document_id, c.chunk_index, c.text, bm25(chunks_fts) "
                               "FROM chunks_fts JOIN chunks c ON c.chunk_id = chunks_fts.rowid "
                               "WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) ASC LIMIT ?;");
    sqlite3_bind_text(statement.get(), 1, match_query.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(statement.get(), 2, static_cast<sqlite3_int64>(k));

    std::vector<ChunkSearchResult> results;
    int step_rc;
    while ((step_rc = sqlite3_step(statement.get())) == SQLITE_ROW) {
        ChunkSearchResult result;
        result.document_id = ColumnText(statement.get(), 0);
        result.chunk_index = static_cast<std::size_t>(sqlite3_column_int64(statement.get(), 1));
        result.text = ColumnText(statement.get(), 2);
        result.score = static_cast<float>(sqlite3_column_double(statement.get(), 3));
        results.push_back(std::move(result));
    }
    if (step_rc != SQLITE_DONE) {
        throw std::runtime_error(std::string("ChunkRepository::SearchSparse: FTS5 query failed: ") +
                                  sqlite3_errmsg(db_));
    }

    return results;
}

std::int64_t ChunkRepository::GetCreatedAt(const std::string& document_id, std::size_t chunk_index) const {
    SqliteStatement statement(db_, "SELECT created_at FROM chunks WHERE document_id = ? AND chunk_index = ?;");
    sqlite3_bind_text(statement.get(), 1, document_id.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(statement.get(), 2, static_cast<sqlite3_int64>(chunk_index));

    if (sqlite3_step(statement.get()) != SQLITE_ROW) {
        throw std::runtime_error("ChunkRepository::GetCreatedAt: no chunk row for this (document_id, chunk_index)");
    }

    return static_cast<std::int64_t>(sqlite3_column_int64(statement.get(), 0));
}

}  // namespace retrieval_engine::detail
