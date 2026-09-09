#include "chunk_store.hpp"

#include "sqlite_util.hpp"
#include "usearch_util.hpp"

#include <sqlite3.h>

#include <stdexcept>
#include <utility>

namespace retrieval_engine::detail {

using unum::usearch::index_dense_t;
using unum::usearch::metric_kind_t;
using unum::usearch::metric_punned_t;

ChunkStore::ChunkStore(sqlite3* db, std::size_t dim)
    : db_(db), dimensions_(dim), index_(index_dense_t::make(metric_punned_t(dim, metric_kind_t::cos_k))) {
    // `chunk_id` is a plain SQLite rowid alias (INTEGER PRIMARY KEY, no
    // AUTOINCREMENT -- this store never deletes rows, so plain monotonic
    // reuse-after-max-rowid semantics are irrelevant) so it can double as
    // the usearch vector key with no separate id table:
    // sqlite3_last_insert_rowid() after each chunk insert *is* the key
    // add_documents() hands to the index.
    ThrowIfSqliteError(sqlite3_exec(db_,
                                     "CREATE TABLE IF NOT EXISTS documents ("
                                     "  document_id TEXT PRIMARY KEY,"
                                     "  metadata TEXT NOT NULL"
                                     ");",
                                     nullptr, nullptr, nullptr),
                        db_, "ChunkStore: failed to create documents table");

    ThrowIfSqliteError(sqlite3_exec(db_,
                                     "CREATE TABLE IF NOT EXISTS chunks ("
                                     "  chunk_id INTEGER PRIMARY KEY,"
                                     "  document_id TEXT NOT NULL REFERENCES documents(document_id),"
                                     "  chunk_index INTEGER NOT NULL,"
                                     "  text TEXT NOT NULL,"
                                     "  start_token INTEGER NOT NULL,"
                                     "  end_token INTEGER NOT NULL,"
                                     "  embedding BLOB NOT NULL"
                                     ");",
                                     nullptr, nullptr, nullptr),
                        db_, "ChunkStore: failed to create chunks table");

    RebuildIndexFromSqlite();
}

void ChunkStore::RebuildIndexFromSqlite() {
    SqliteStatement statement(db_, "SELECT chunk_id, embedding FROM chunks;");
    const std::size_t expected_bytes = dimensions_ * sizeof(float);

    int step_rc;
    while ((step_rc = sqlite3_step(statement.get())) == SQLITE_ROW) {
        const auto chunk_id = static_cast<std::uint64_t>(sqlite3_column_int64(statement.get(), 0));
        const void* embedding_blob = sqlite3_column_blob(statement.get(), 1);
        const auto blob_bytes = static_cast<std::size_t>(sqlite3_column_bytes(statement.get(), 1));

        if (blob_bytes != expected_bytes) {
            throw std::runtime_error(
                "ChunkStore: a stored chunk embedding's size does not match this store's dimensionality "
                "(this database may have been created with a different `dim`)");
        }

        EnsureCapacity(index_, "ChunkStore: index rebuild");
        const auto add_result = index_.add(static_cast<index_dense_t::vector_key_t>(chunk_id),
                                            static_cast<const float*>(embedding_blob));
        if (!add_result) {
            throw std::runtime_error(std::string("ChunkStore: failed to rebuild index: ") +
                                      (add_result.error.what() ? add_result.error.what() : "unknown error"));
        }
    }

    if (step_rc != SQLITE_DONE) {
        throw std::runtime_error(std::string("ChunkStore: failed while reading the chunks table: ") +
                                  sqlite3_errmsg(db_));
    }
}

void ChunkStore::add_documents(const std::vector<DocumentInput>& documents) {
    // Validate everything before writing anything, so a bad chunk deep in
    // the batch doesn't leave earlier documents in this same call partially
    // ingested (the transaction below protects against a mid-write SQLite/
    // usearch failure; this protects against a caller mistake up front).
    for (const auto& document : documents) {
        for (const auto& chunk : document.chunks) {
            if (chunk.embedding.size() != dimensions_) {
                throw std::invalid_argument(
                    "ChunkStore::add_documents: chunk embedding size does not match index dimensionality");
            }
        }
    }

    // usearch has no transaction concept, so its .add() calls must not
    // happen until *after* SQLite's transaction has durably committed:
    // otherwise a later document in this same batch failing (e.g. a
    // duplicate document_id) would ROLLBACK every SQLite row from this call
    // while leaving whatever chunks were already added to `index_` sitting
    // there permanently -- a usearch entry with no backing SQLite row, the
    // exact inconsistency the "SQLite is authoritative, usearch is a
    // rebuildable sidecar" architecture (BUILD_PLAN.md section 5) doesn't
    // support recovering from. Buffer (chunk_id, embedding) pairs here and
    // populate the index only once COMMIT has succeeded.
    std::vector<std::pair<std::uint64_t, const std::vector<float>*>> pending_index_adds;

    // One transaction for the whole batch: ingesting is dominated by
    // per-statement fsync overhead if each row auto-commits individually,
    // which would make "ingest 10k chunks" (BUILD_PLAN.md Stage 1) far
    // slower than it needs to be. If anything below throws, the catch block
    // rolls back so SQLite never ends up with a partially-committed batch.
    ThrowIfSqliteError(sqlite3_exec(db_, "BEGIN;", nullptr, nullptr, nullptr), db_,
                        "ChunkStore::add_documents: failed to begin transaction");

    try {
        SqliteStatement insert_document(db_, "INSERT INTO documents (document_id, metadata) VALUES (?, ?);");
        SqliteStatement insert_chunk(
            db_,
            "INSERT INTO chunks (document_id, chunk_index, text, start_token, end_token, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?);");

        for (const auto& document : documents) {
            sqlite3_reset(insert_document.get());
            sqlite3_bind_text(insert_document.get(), 1, document.document_id.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(insert_document.get(), 2, document.metadata.c_str(), -1, SQLITE_TRANSIENT);
            if (sqlite3_step(insert_document.get()) != SQLITE_DONE) {
                throw std::runtime_error(std::string("ChunkStore::add_documents: failed to insert document '") +
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

                if (sqlite3_step(insert_chunk.get()) != SQLITE_DONE) {
                    throw std::runtime_error(std::string("ChunkStore::add_documents: failed to insert chunk: ") +
                                              sqlite3_errmsg(db_));
                }

                // The chunk's SQLite rowid *is* its usearch key: assigned by
                // SQLite (not chosen by us), guaranteed unique, and cheap to
                // map back (WHERE chunk_id = ?) when search_chunks() resolves
                // usearch's results to their SQLite rows.
                const auto chunk_id = static_cast<std::uint64_t>(sqlite3_last_insert_rowid(db_));
                pending_index_adds.emplace_back(chunk_id, &chunk.embedding);
            }
        }
    } catch (...) {
        sqlite3_exec(db_, "ROLLBACK;", nullptr, nullptr, nullptr);
        throw;
    }

    ThrowIfSqliteError(sqlite3_exec(db_, "COMMIT;", nullptr, nullptr, nullptr), db_,
                        "ChunkStore::add_documents: failed to commit transaction");

    // SQLite now durably has every row from this batch. If a usearch add
    // below fails, the index merely lags what's in SQLite -- recoverable by
    // rebuilding it (BUILD_PLAN.md section 5) -- rather than containing an
    // entry SQLite has no record of, which a mid-transaction failure could
    // not have recovered from.
    for (const auto& [chunk_id, embedding] : pending_index_adds) {
        EnsureCapacity(index_, "ChunkStore::add_documents");
        const auto add_result = index_.add(static_cast<index_dense_t::vector_key_t>(chunk_id), embedding->data());
        if (!add_result) {
            throw std::runtime_error(std::string("ChunkStore::add_documents: usearch insertion failed: ") +
                                      (add_result.error.what() ? add_result.error.what() : "unknown error"));
        }
    }
}

std::vector<ChunkSearchResult> ChunkStore::search_chunks(const std::vector<float>& query, std::size_t k) const {
    if (query.size() != dimensions_)
        throw std::invalid_argument("ChunkStore::search_chunks: query size does not match index dimensionality");

    const auto search_result = index_.search(query.data(), k);
    if (!search_result) {
        throw std::runtime_error(std::string("ChunkStore::search_chunks: usearch query failed: ") +
                                  (search_result.error.what() ? search_result.error.what() : "unknown error"));
    }

    std::vector<std::uint64_t> chunk_ids(search_result.size());
    std::vector<float> distances(search_result.size());
    search_result.dump_to(chunk_ids.data(), distances.data());

    SqliteStatement lookup(db_, "SELECT document_id, chunk_index, text FROM chunks WHERE chunk_id = ?;");

    std::vector<ChunkSearchResult> results;
    results.reserve(chunk_ids.size());
    for (std::size_t i = 0; i < chunk_ids.size(); ++i) {
        sqlite3_reset(lookup.get());
        sqlite3_bind_int64(lookup.get(), 1, static_cast<sqlite3_int64>(chunk_ids[i]));

        if (sqlite3_step(lookup.get()) != SQLITE_ROW) {
            throw std::runtime_error(
                "ChunkStore::search_chunks: usearch returned a chunk id with no matching SQLite row "
                "(index and store have desynced)");
        }

        ChunkSearchResult result;
        result.document_id = ColumnText(lookup.get(), 0);
        result.chunk_index = static_cast<std::size_t>(sqlite3_column_int64(lookup.get(), 1));
        result.text = ColumnText(lookup.get(), 2);
        result.distance = distances[i];
        results.push_back(std::move(result));
    }

    return results;
}

std::size_t ChunkStore::chunk_count() const {
    SqliteStatement statement(db_, "SELECT COUNT(*) FROM chunks;");
    if (sqlite3_step(statement.get()) != SQLITE_ROW)
        throw std::runtime_error("ChunkStore::chunk_count: failed to execute row count query");
    return static_cast<std::size_t>(sqlite3_column_int64(statement.get(), 0));
}

}  // namespace retrieval_engine::detail
