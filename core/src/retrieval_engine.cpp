#include "retrieval_engine/retrieval_engine.hpp"

#include <sqlite3.h>
#include <usearch/index_dense.hpp>

#include <sstream>
#include <stdexcept>
#include <utility>

namespace retrieval_engine {

namespace {

using unum::usearch::index_dense_t;
using unum::usearch::index_limits_t;
using unum::usearch::metric_kind_t;
using unum::usearch::metric_punned_t;

// Throws std::runtime_error naming `context` and SQLite's own error message
// unless `sqlite_rc` is SQLITE_OK. Only meant for calls (open/prepare/exec)
// whose success code is SQLITE_OK -- sqlite3_step()'s SQLITE_DONE/SQLITE_ROW
// are handled at their call sites instead, since "success" means something
// different there.
void ThrowIfSqliteError(int sqlite_rc, sqlite3* db, const std::string& context) {
    if (sqlite_rc == SQLITE_OK) return;
    std::ostringstream message;
    message << context << ": " << (db ? sqlite3_errmsg(db) : "unknown SQLite error");
    throw std::runtime_error(message.str());
}

// RAII wrapper for a prepared sqlite3_stmt*. Stage 1's methods have multiple
// throw points (validation, usearch failures) between preparing a statement
// and finishing with it -- tracking `sqlite3_finalize()` by hand across all
// of them is exactly the kind of thing that silently leaks a statement
// handle on one missed path. Reset-and-rebind for repeated use (e.g. once
// per document/chunk in a loop) via `.get()`.
class SqliteStatement {
public:
    SqliteStatement(sqlite3* db, const char* sql) {
        ThrowIfSqliteError(sqlite3_prepare_v2(db, sql, -1, &statement_, nullptr), db,
                            std::string("failed to prepare statement: ") + sql);
    }
    ~SqliteStatement() { sqlite3_finalize(statement_); }

    SqliteStatement(const SqliteStatement&) = delete;
    SqliteStatement& operator=(const SqliteStatement&) = delete;

    sqlite3_stmt* get() const { return statement_; }

private:
    sqlite3_stmt* statement_ = nullptr;
};

// sqlite3_column_text() can return nullptr (a SQL NULL, or on OOM); building
// a std::string from a null `const char*` is undefined behaviour. Our schema
// declares these columns NOT NULL, so nullptr here should only mean OOM, but
// guard it regardless per CLAUDE.md's "defensive programming at boundaries".
std::string ColumnText(sqlite3_stmt* statement, int column) {
    const unsigned char* text = sqlite3_column_text(statement, column);
    return text ? std::string(reinterpret_cast<const char*>(text)) : std::string();
}

}  // namespace

// The private implementation (Pimpl idiom) -- keeps usearch/SQLite types out
// of the public header.
struct RetrievalEngine::Impl {
    std::size_t dimensions;
    sqlite3* db = nullptr;
    index_dense_t index;  // Stage 0's dummy_vectors index (kept as-is; see retrieval_engine.hpp)

    // Stage 1's real chunk-search index. Cosine similarity (rather than
    // Stage 0's l2sq) per the Stage 1 test's stated requirement and because
    // it's the standard choice for text-embedding retrieval. Kept as a
    // second, separate index rather than repurposing `index` above so
    // Stage 0's already-shipped add_vector()/search() contract doesn't
    // change; see docs/DECISIONS.md for the tradeoff.
    index_dense_t chunk_index;

    explicit Impl(std::size_t dim)
        : dimensions(dim),
          index(index_dense_t::make(metric_punned_t(dim, metric_kind_t::l2sq_k))),
          chunk_index(index_dense_t::make(metric_punned_t(dim, metric_kind_t::cos_k))) {}

    ~Impl() {
        if (db) sqlite3_close(db);
    }

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    // Grows `idx`'s capacity if it's about to be exceeded (usearch requires
    // `size() < capacity()` ahead of every add()). Amortized-doubling, same
    // strategy as std::vector, since callers add one vector at a time.
    static void ReserveIfFull(index_dense_t& idx, const char* context) {
        if (idx.size() != idx.capacity()) return;
        const std::size_t new_capacity = idx.capacity() == 0 ? 64 : idx.capacity() * 2;
        if (!idx.reserve(index_limits_t(new_capacity)))
            throw std::runtime_error(std::string(context) + ": failed to reserve usearch index capacity");
    }

    // Reloads `chunk_index` from whatever is already in the `chunks` table.
    // This is the "rebuild the index from SQLite" recovery path from
    // BUILD_PLAN.md section 5 -- Stage 1 doesn't persist the usearch index
    // itself (only SQLite is durable), so every open runs this, not just
    // recovery from corruption. A no-op on a fresh/empty database. Defined
    // out-of-line below (a member so it can name the private `Impl` type;
    // a free function taking `Impl&` cannot, even from this same file).
    void RebuildChunkIndexFromSqlite();
};

void RetrievalEngine::Impl::RebuildChunkIndexFromSqlite() {
    SqliteStatement statement(db, "SELECT chunk_id, embedding FROM chunks;");
    const std::size_t expected_bytes = dimensions * sizeof(float);

    int step_rc;
    while ((step_rc = sqlite3_step(statement.get())) == SQLITE_ROW) {
        const auto chunk_id = static_cast<std::uint64_t>(sqlite3_column_int64(statement.get(), 0));
        const void* embedding_blob = sqlite3_column_blob(statement.get(), 1);
        const auto blob_bytes = static_cast<std::size_t>(sqlite3_column_bytes(statement.get(), 1));

        if (blob_bytes != expected_bytes) {
            throw std::runtime_error(
                "RetrievalEngine: a stored chunk embedding's size does not match this engine's dimensionality "
                "(this database may have been created with a different `dim`)");
        }

        ReserveIfFull(chunk_index, "RetrievalEngine: chunk index rebuild");
        const auto add_result = chunk_index.add(static_cast<index_dense_t::vector_key_t>(chunk_id),
                                                  static_cast<const float*>(embedding_blob));
        if (!add_result) {
            throw std::runtime_error(std::string("RetrievalEngine: failed to rebuild chunk index: ") +
                                      (add_result.error.what() ? add_result.error.what() : "unknown error"));
        }
    }

    if (step_rc != SQLITE_DONE) {
        throw std::runtime_error(std::string("RetrievalEngine: failed while reading the chunks table: ") +
                                  sqlite3_errmsg(db));
    }
}

RetrievalEngine::RetrievalEngine(const std::string& db_path, std::size_t dim) {
    if (dim == 0) throw std::invalid_argument("RetrievalEngine: dim must be greater than zero");

    impl_ = std::make_unique<Impl>(dim);

    sqlite3* db = nullptr;
    const int open_rc = sqlite3_open(db_path.c_str(), &db);
    impl_->db = db;  // assign immediately so ~Impl() closes it even if we throw below
    ThrowIfSqliteError(open_rc, db, "RetrievalEngine: failed to open SQLite database at '" + db_path + "'");

    ThrowIfSqliteError(
        sqlite3_exec(db, "CREATE TABLE IF NOT EXISTS dummy_vectors (id INTEGER PRIMARY KEY);", nullptr, nullptr,
                     nullptr),
        db, "RetrievalEngine: failed to create dummy_vectors table");

    // Stage 1's real schema. `chunks.chunk_id` is a plain SQLite rowid alias
    // (INTEGER PRIMARY KEY, no AUTOINCREMENT -- Stage 1 never deletes rows,
    // so plain monotonic reuse-after-max-rowid semantics are irrelevant) so
    // it can double as the usearch vector key with no separate id table:
    // sqlite3_last_insert_rowid() after each chunk insert *is* the key
    // add_documents() hands to the chunk-search index below.
    ThrowIfSqliteError(sqlite3_exec(db,
                                     "CREATE TABLE IF NOT EXISTS documents ("
                                     "  document_id TEXT PRIMARY KEY,"
                                     "  metadata TEXT NOT NULL"
                                     ");",
                                     nullptr, nullptr, nullptr),
                        db, "RetrievalEngine: failed to create documents table");

    ThrowIfSqliteError(sqlite3_exec(db,
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
                        db, "RetrievalEngine: failed to create chunks table");

    impl_->RebuildChunkIndexFromSqlite();
}

RetrievalEngine::~RetrievalEngine() = default;
RetrievalEngine::RetrievalEngine(RetrievalEngine&&) noexcept = default;
RetrievalEngine& RetrievalEngine::operator=(RetrievalEngine&&) noexcept = default;

void RetrievalEngine::add_vector(std::uint64_t id, const std::vector<float>& vector) {
    if (vector.size() != impl_->dimensions)
        throw std::invalid_argument("RetrievalEngine::add_vector: vector size does not match index dimensionality");

    // Write SQLite (the architecture's authoritative store, see
    // BUILD_PLAN.md section 5) *before* touching the usearch index (a
    // rebuildable acceleration sidecar). If the SQLite write fails, we throw
    // having mutated nothing; if it succeeds but the usearch insert below
    // fails, the index merely lags what SQLite already durably recorded --
    // recoverable by rebuilding it from SQLite. Doing this in the opposite
    // order could leave a vector live in the index with no backing row in
    // the store of record, which the architecture doesn't support recovering
    // from.
    sqlite3_stmt* statement = nullptr;
    ThrowIfSqliteError(
        sqlite3_prepare_v2(impl_->db, "INSERT INTO dummy_vectors (id) VALUES (?);", -1, &statement, nullptr),
        impl_->db, "RetrievalEngine::add_vector: failed to prepare insert statement");

    // `id` is caller-controlled and unsigned; SQLite's column is a signed
    // 64-bit integer. Values above INT64_MAX would round-trip as negative
    // numbers. Harmless today (dummy_table_row_count() only counts rows, it
    // never reads ids back), but flagged here for whoever later reads ids
    // out of this table and compares them against usearch's uint64 keys.
    sqlite3_bind_int64(statement, 1, static_cast<sqlite3_int64>(id));
    const int step_rc = sqlite3_step(statement);
    sqlite3_finalize(statement);
    if (step_rc != SQLITE_DONE) {
        throw std::runtime_error(std::string("RetrievalEngine::add_vector: failed to insert dummy row: ") +
                                  sqlite3_errmsg(impl_->db));
    }

    Impl::ReserveIfFull(impl_->index, "RetrievalEngine::add_vector");

    const auto add_result = impl_->index.add(static_cast<index_dense_t::vector_key_t>(id), vector.data());
    if (!add_result) {
        throw std::runtime_error(std::string("RetrievalEngine::add_vector: usearch insertion failed: ") +
                                  (add_result.error.what() ? add_result.error.what() : "unknown error"));
    }
}

std::vector<std::uint64_t> RetrievalEngine::search(const std::vector<float>& query, std::size_t k) const {
    if (query.size() != impl_->dimensions)
        throw std::invalid_argument("RetrievalEngine::search: query size does not match index dimensionality");

    const auto search_result = impl_->index.search(query.data(), k);
    if (!search_result) {
        throw std::runtime_error(std::string("RetrievalEngine::search: usearch query failed: ") +
                                  (search_result.error.what() ? search_result.error.what() : "unknown error"));
    }

    std::vector<std::uint64_t> neighbour_ids(search_result.size());
    search_result.dump_to(neighbour_ids.data());
    return neighbour_ids;
}

std::size_t RetrievalEngine::dummy_table_row_count() const {
    sqlite3_stmt* statement = nullptr;
    ThrowIfSqliteError(sqlite3_prepare_v2(impl_->db, "SELECT COUNT(*) FROM dummy_vectors;", -1, &statement, nullptr),
                        impl_->db, "RetrievalEngine::dummy_table_row_count: failed to prepare query");

    const int step_rc = sqlite3_step(statement);
    if (step_rc != SQLITE_ROW) {
        sqlite3_finalize(statement);
        throw std::runtime_error("RetrievalEngine::dummy_table_row_count: failed to execute row count query");
    }

    const std::size_t count = static_cast<std::size_t>(sqlite3_column_int64(statement, 0));
    sqlite3_finalize(statement);
    return count;
}

void RetrievalEngine::add_documents(const std::vector<DocumentInput>& documents) {
    // Validate everything before writing anything, so a bad chunk deep in
    // the batch doesn't leave earlier documents in this same call partially
    // ingested (the transaction below protects against a mid-write SQLite/
    // usearch failure; this protects against a caller mistake up front).
    for (const auto& document : documents) {
        for (const auto& chunk : document.chunks) {
            if (chunk.embedding.size() != impl_->dimensions) {
                throw std::invalid_argument(
                    "RetrievalEngine::add_documents: chunk embedding size does not match index dimensionality");
            }
        }
    }

    // usearch has no transaction concept, so its .add() calls must not
    // happen until *after* SQLite's transaction has durably committed:
    // otherwise a later document in this same batch failing (e.g. a
    // duplicate document_id) would ROLLBACK every SQLite row from this call
    // while leaving whatever chunks were already added to chunk_index
    // sitting there permanently -- a usearch entry with no backing SQLite
    // row, the exact inconsistency the "SQLite is authoritative, usearch is
    // a rebuildable sidecar" architecture (BUILD_PLAN.md section 5) doesn't
    // support recovering from. Buffer (chunk_id, embedding) pairs here and
    // populate the index only once COMMIT has succeeded.
    std::vector<std::pair<std::uint64_t, const std::vector<float>*>> pending_index_adds;

    // One transaction for the whole batch: ingesting is dominated by
    // per-statement fsync overhead if each row auto-commits individually,
    // which would make "ingest 10k chunks" (BUILD_PLAN.md Stage 1) far
    // slower than it needs to be. If anything below throws, the catch block
    // rolls back so SQLite never ends up with a partially-committed batch.
    ThrowIfSqliteError(sqlite3_exec(impl_->db, "BEGIN;", nullptr, nullptr, nullptr), impl_->db,
                        "RetrievalEngine::add_documents: failed to begin transaction");

    try {
        SqliteStatement insert_document(impl_->db, "INSERT INTO documents (document_id, metadata) VALUES (?, ?);");
        SqliteStatement insert_chunk(
            impl_->db,
            "INSERT INTO chunks (document_id, chunk_index, text, start_token, end_token, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?);");

        for (const auto& document : documents) {
            sqlite3_reset(insert_document.get());
            sqlite3_bind_text(insert_document.get(), 1, document.document_id.c_str(), -1, SQLITE_TRANSIENT);
            sqlite3_bind_text(insert_document.get(), 2, document.metadata.c_str(), -1, SQLITE_TRANSIENT);
            if (sqlite3_step(insert_document.get()) != SQLITE_DONE) {
                throw std::runtime_error(std::string("RetrievalEngine::add_documents: failed to insert document '") +
                                          document.document_id + "': " + sqlite3_errmsg(impl_->db));
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
                    throw std::runtime_error(std::string("RetrievalEngine::add_documents: failed to insert chunk: ") +
                                              sqlite3_errmsg(impl_->db));
                }

                // The chunk's SQLite rowid *is* its usearch key: assigned by
                // SQLite (not chosen by us), guaranteed unique, and cheap to
                // map back (WHERE chunk_id = ?) when search_chunks() resolves
                // usearch's results to their SQLite rows.
                const auto chunk_id = static_cast<std::uint64_t>(sqlite3_last_insert_rowid(impl_->db));
                pending_index_adds.emplace_back(chunk_id, &chunk.embedding);
            }
        }
    } catch (...) {
        sqlite3_exec(impl_->db, "ROLLBACK;", nullptr, nullptr, nullptr);
        throw;
    }

    ThrowIfSqliteError(sqlite3_exec(impl_->db, "COMMIT;", nullptr, nullptr, nullptr), impl_->db,
                        "RetrievalEngine::add_documents: failed to commit transaction");

    // SQLite now durably has every row from this batch. If a usearch add
    // below fails, the index merely lags what's in SQLite -- recoverable by
    // rebuilding it (BUILD_PLAN.md section 5) -- rather than containing an
    // entry SQLite has no record of, which a mid-transaction failure could
    // not have recovered from.
    for (const auto& [chunk_id, embedding] : pending_index_adds) {
        Impl::ReserveIfFull(impl_->chunk_index, "RetrievalEngine::add_documents");
        const auto add_result =
            impl_->chunk_index.add(static_cast<index_dense_t::vector_key_t>(chunk_id), embedding->data());
        if (!add_result) {
            throw std::runtime_error(std::string("RetrievalEngine::add_documents: usearch insertion failed: ") +
                                      (add_result.error.what() ? add_result.error.what() : "unknown error"));
        }
    }
}

std::vector<ChunkSearchResult> RetrievalEngine::search_chunks(const std::vector<float>& query, std::size_t k) const {
    if (query.size() != impl_->dimensions)
        throw std::invalid_argument("RetrievalEngine::search_chunks: query size does not match index dimensionality");

    const auto search_result = impl_->chunk_index.search(query.data(), k);
    if (!search_result) {
        throw std::runtime_error(std::string("RetrievalEngine::search_chunks: usearch query failed: ") +
                                  (search_result.error.what() ? search_result.error.what() : "unknown error"));
    }

    std::vector<std::uint64_t> chunk_ids(search_result.size());
    std::vector<float> distances(search_result.size());
    search_result.dump_to(chunk_ids.data(), distances.data());

    SqliteStatement lookup(impl_->db, "SELECT document_id, chunk_index, text FROM chunks WHERE chunk_id = ?;");

    std::vector<ChunkSearchResult> results;
    results.reserve(chunk_ids.size());
    for (std::size_t i = 0; i < chunk_ids.size(); ++i) {
        sqlite3_reset(lookup.get());
        sqlite3_bind_int64(lookup.get(), 1, static_cast<sqlite3_int64>(chunk_ids[i]));

        if (sqlite3_step(lookup.get()) != SQLITE_ROW) {
            throw std::runtime_error(
                "RetrievalEngine::search_chunks: usearch returned a chunk id with no matching SQLite row "
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

std::size_t RetrievalEngine::chunk_count() const {
    SqliteStatement statement(impl_->db, "SELECT COUNT(*) FROM chunks;");
    if (sqlite3_step(statement.get()) != SQLITE_ROW)
        throw std::runtime_error("RetrievalEngine::chunk_count: failed to execute row count query");
    return static_cast<std::size_t>(sqlite3_column_int64(statement.get(), 0));
}

}  // namespace retrieval_engine
