#include "chunk_store.hpp"

#include "sqlite_util.hpp"
#include "usearch_util.hpp"

#include <sqlite3.h>

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string_view>
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

    // Stage 2's sparse index. A plain (not "external content") FTS5 table,
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
        db_, "ChunkStore: failed to create chunks_fts table");

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
        // Unlike the usearch index below, chunks_fts is a genuine SQLite
        // table: writing to it here, inside the same BEGIN/COMMIT as
        // `chunks`, makes it atomic with the main insert for free -- no
        // buffer-until-commit dance required.
        SqliteStatement insert_chunk_fts(db_, "INSERT INTO chunks_fts(rowid, text) VALUES (?, ?);");

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
                // map back (WHERE chunk_id = ?) when search_dense() resolves
                // usearch's results to their SQLite rows. Reused as-is for
                // chunks_fts's rowid, so the two stay trivially joinable.
                const auto chunk_id = static_cast<std::uint64_t>(sqlite3_last_insert_rowid(db_));
                pending_index_adds.emplace_back(chunk_id, &chunk.embedding);

                sqlite3_reset(insert_chunk_fts.get());
                sqlite3_bind_int64(insert_chunk_fts.get(), 1, static_cast<sqlite3_int64>(chunk_id));
                sqlite3_bind_text(insert_chunk_fts.get(), 2, chunk.text.c_str(), -1, SQLITE_TRANSIENT);
                if (sqlite3_step(insert_chunk_fts.get()) != SQLITE_DONE) {
                    throw std::runtime_error(std::string("ChunkStore::add_documents: failed to insert into "
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

std::vector<ChunkSearchResult> ChunkStore::search_dense(const std::vector<float>& query, std::size_t k) const {
    if (query.size() != dimensions_)
        throw std::invalid_argument("ChunkStore::search_dense: query size does not match index dimensionality");

    const auto search_result = index_.search(query.data(), k);
    if (!search_result) {
        throw std::runtime_error(std::string("ChunkStore::search_dense: usearch query failed: ") +
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
                "ChunkStore::search_dense: usearch returned a chunk id with no matching SQLite row "
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

namespace {

// FTS5's MATCH operand isn't a literal string -- it's parsed by FTS5's own
// query grammar (AND/OR/NOT, quoted phrases, a leading '-' meaning NOT,
// '*' prefix queries, "column:" filters, NEAR()). Passing raw, untrusted
// search text straight through means an ordinary hyphenated word or an
// unbalanced quote -- both entirely normal in real search input -- throws
// a SQLite error instead of matching literal text (verified against our
// exact linked SQLite3: `"ZXQ7742 -stock"` raises "no such column: stock",
// and `hello "world` raises "unterminated string").
//
// Splits `query_text` on whitespace and wraps each token in its own
// double-quoted phrase (escaping embedded '"' by doubling, FTS5's string
// escape -- confirmed against our linked build), OR'd together. Wrapping in
// quotes takes every character out of FTS5's operator grammar entirely: a
// quoted phrase is tokenized like ordinary document text, not parsed for
// operators, so nothing the caller types can be interpreted as query syntax
// -- it can only ever mean "search for this literal text". OR (rather than
// implicit AND) matches typical keyword-search UX: find chunks containing
// any of the given terms, ranked by BM25, not requiring every term present.
//
// Returns an empty string if `query_text` has no non-whitespace content.
std::string BuildSafeFts5MatchQuery(const std::string& query_text) {
    std::vector<std::string> quoted_terms;

    const std::string_view remaining(query_text);
    std::size_t pos = 0;
    while (pos < remaining.size()) {
        pos = remaining.find_first_not_of(" \t\n\r\f\v", pos);
        if (pos == std::string_view::npos) break;

        std::size_t end = remaining.find_first_of(" \t\n\r\f\v", pos);
        if (end == std::string_view::npos) end = remaining.size();

        std::string escaped_term;
        escaped_term.reserve(end - pos);
        for (std::size_t i = pos; i < end; ++i) {
            escaped_term += remaining[i];
            if (remaining[i] == '"') escaped_term += '"';  // "" is FTS5's escape for a literal quote
        }
        quoted_terms.push_back("\"" + escaped_term + "\"");

        pos = end;
    }

    if (quoted_terms.empty()) return {};

    std::string match_query = quoted_terms[0];
    for (std::size_t i = 1; i < quoted_terms.size(); ++i) match_query += " OR " + quoted_terms[i];
    return match_query;
}

}  // namespace

std::vector<ChunkSearchResult> ChunkStore::search_sparse(const std::string& query_text, std::size_t k) const {
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
        result.distance = static_cast<float>(sqlite3_column_double(statement.get(), 3));
        results.push_back(std::move(result));
    }
    if (step_rc != SQLITE_DONE) {
        throw std::runtime_error(std::string("ChunkStore::search_sparse: FTS5 query failed: ") +
                                  sqlite3_errmsg(db_));
    }

    return results;
}

namespace {
constexpr double kRrfK = 60.0;
}  // namespace

std::vector<ChunkStore::FusionEntry> ChunkStore::FuseAndRank(const std::string& query_text,
                                                              const std::vector<float>& query_vec,
                                                              std::size_t k) const {
    const std::vector<ChunkSearchResult> dense_results = search_dense(query_vec, k);
    const std::vector<ChunkSearchResult> sparse_results = search_sparse(query_text, k);

    // Keyed by (document_id, chunk_index) -- unique within this store's
    // schema (chunk_index is 0-based position within its document) -- so a
    // chunk hit by both rankings merges into one entry instead of two.
    std::map<std::pair<std::string, std::size_t>, FusionEntry> entries;

    for (std::size_t i = 0; i < dense_results.size(); ++i) {
        const ChunkSearchResult& r = dense_results[i];
        FusionEntry& entry = entries[{r.document_id, r.chunk_index}];
        entry.document_id = r.document_id;
        entry.chunk_index = r.chunk_index;
        entry.text = r.text;
        entry.dense_present = true;
        entry.dense_distance = r.distance;
        entry.dense_rank = i + 1;  // 1-based
    }

    for (std::size_t i = 0; i < sparse_results.size(); ++i) {
        const ChunkSearchResult& r = sparse_results[i];
        FusionEntry& entry = entries[{r.document_id, r.chunk_index}];
        entry.document_id = r.document_id;
        entry.chunk_index = r.chunk_index;
        entry.text = r.text;
        entry.sparse_present = true;
        entry.sparse_bm25_score = r.distance;
        entry.sparse_rank = i + 1;  // 1-based
    }

    std::vector<FusionEntry> sorted_entries;
    sorted_entries.reserve(entries.size());
    for (auto& [key, entry] : entries) {
        entry.fused_score = 0.0f;
        if (entry.dense_present)
            entry.fused_score += static_cast<float>(1.0 / (kRrfK + static_cast<double>(entry.dense_rank)));
        if (entry.sparse_present)
            entry.fused_score += static_cast<float>(1.0 / (kRrfK + static_cast<double>(entry.sparse_rank)));
        sorted_entries.push_back(std::move(entry));
    }

    // Break ties on fused_score deterministically (by document_id then
    // chunk_index) rather than leaving them to std::sort's unspecified
    // handling of equal elements -- an explainability feature (BUILD_PLAN.md
    // Stage 2) should give the same, reproducible ordering for the same
    // input every time, not one that happens to depend on std::map's
    // iteration order or the sort algorithm's internal behavior.
    std::sort(sorted_entries.begin(), sorted_entries.end(), [](const FusionEntry& a, const FusionEntry& b) {
        if (a.fused_score != b.fused_score) return a.fused_score > b.fused_score;
        if (a.document_id != b.document_id) return a.document_id < b.document_id;
        return a.chunk_index < b.chunk_index;
    });
    if (sorted_entries.size() > k) sorted_entries.resize(k);

    return sorted_entries;
}

std::vector<ChunkSearchResult> ChunkStore::search_hybrid(const std::string& query_text,
                                                          const std::vector<float>& query_vec, std::size_t k) const {
    const std::vector<FusionEntry> fused = FuseAndRank(query_text, query_vec, k);

    std::vector<ChunkSearchResult> results;
    results.reserve(fused.size());
    for (const FusionEntry& entry : fused) {
        results.push_back(ChunkSearchResult{entry.document_id, entry.chunk_index, entry.text, entry.fused_score});
    }
    return results;
}

std::vector<SearchExplanation> ChunkStore::search_explained(const std::string& query_text,
                                                             const std::vector<float>& query_vec,
                                                             std::size_t k) const {
    const std::vector<FusionEntry> fused = FuseAndRank(query_text, query_vec, k);

    std::vector<SearchExplanation> explanations;
    explanations.reserve(fused.size());
    for (std::size_t i = 0; i < fused.size(); ++i) {
        const FusionEntry& entry = fused[i];
        SearchExplanation explanation;
        explanation.document_id = entry.document_id;
        explanation.chunk_index = entry.chunk_index;
        explanation.text = entry.text;
        explanation.dense_present = entry.dense_present;
        explanation.dense_distance = entry.dense_distance;
        explanation.dense_rank = entry.dense_rank;
        explanation.sparse_present = entry.sparse_present;
        explanation.sparse_bm25_score = entry.sparse_bm25_score;
        explanation.sparse_rank = entry.sparse_rank;
        explanation.fused_score = entry.fused_score;
        explanation.final_rank = i + 1;  // 1-based
        explanations.push_back(std::move(explanation));
    }
    return explanations;
}

std::size_t ChunkStore::chunk_count() const {
    SqliteStatement statement(db_, "SELECT COUNT(*) FROM chunks;");
    if (sqlite3_step(statement.get()) != SQLITE_ROW)
        throw std::runtime_error("ChunkStore::chunk_count: failed to execute row count query");
    return static_cast<std::size_t>(sqlite3_column_int64(statement.get(), 0));
}

}  // namespace retrieval_engine::detail
