#include "dummy_vector_store.hpp"

#include "sqlite_util.hpp"
#include "usearch_util.hpp"

#include <sqlite3.h>

#include <stdexcept>

namespace retrieval_engine::detail {

using unum::usearch::index_dense_t;
using unum::usearch::metric_kind_t;
using unum::usearch::metric_punned_t;

DummyVectorStore::DummyVectorStore(sqlite3* db, std::size_t dim)
    : db_(db), dimensions_(dim), index_(index_dense_t::make(metric_punned_t(dim, metric_kind_t::l2sq_k))) {
    ThrowIfSqliteError(
        sqlite3_exec(db_, "CREATE TABLE IF NOT EXISTS dummy_vectors (id INTEGER PRIMARY KEY);", nullptr, nullptr,
                     nullptr),
        db_, "DummyVectorStore: failed to create dummy_vectors table");
}

void DummyVectorStore::add_vector(std::uint64_t id, const std::vector<float>& vector) {
    if (vector.size() != dimensions_)
        throw std::invalid_argument("DummyVectorStore::add_vector: vector size does not match index dimensionality");

    // Write SQLite (the architecture's authoritative store, see
    // BUILD_PLAN.md section 5) *before* touching the usearch index (a
    // rebuildable acceleration sidecar) -- see docs/DECISIONS.md for why.
    SqliteStatement statement(db_, "INSERT INTO dummy_vectors (id) VALUES (?);");

    // `id` is caller-controlled and unsigned; SQLite's column is a signed
    // 64-bit integer. Values above INT64_MAX would round-trip as negative
    // numbers. Harmless today (row_count() only counts rows, it never reads
    // ids back), but flagged here for whoever later reads ids out of this
    // table and compares them against usearch's uint64 keys.
    sqlite3_bind_int64(statement.get(), 1, static_cast<sqlite3_int64>(id));
    if (sqlite3_step(statement.get()) != SQLITE_DONE) {
        throw std::runtime_error(std::string("DummyVectorStore::add_vector: failed to insert dummy row: ") +
                                  sqlite3_errmsg(db_));
    }

    EnsureCapacity(index_, "DummyVectorStore::add_vector");
    const auto add_result = index_.add(static_cast<index_dense_t::vector_key_t>(id), vector.data());
    if (!add_result) {
        throw std::runtime_error(std::string("DummyVectorStore::add_vector: usearch insertion failed: ") +
                                  (add_result.error.what() ? add_result.error.what() : "unknown error"));
    }
}

std::vector<std::uint64_t> DummyVectorStore::search(const std::vector<float>& query, std::size_t k) const {
    if (query.size() != dimensions_)
        throw std::invalid_argument("DummyVectorStore::search: query size does not match index dimensionality");

    const auto search_result = index_.search(query.data(), k);
    if (!search_result) {
        throw std::runtime_error(std::string("DummyVectorStore::search: usearch query failed: ") +
                                  (search_result.error.what() ? search_result.error.what() : "unknown error"));
    }

    std::vector<std::uint64_t> neighbour_ids(search_result.size());
    search_result.dump_to(neighbour_ids.data());
    return neighbour_ids;
}

std::size_t DummyVectorStore::row_count() const {
    SqliteStatement statement(db_, "SELECT COUNT(*) FROM dummy_vectors;");
    if (sqlite3_step(statement.get()) != SQLITE_ROW)
        throw std::runtime_error("DummyVectorStore::row_count: failed to execute row count query");
    return static_cast<std::size_t>(sqlite3_column_int64(statement.get(), 0));
}

}  // namespace retrieval_engine::detail
