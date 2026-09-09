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

}  // namespace

// The private implementation (Pimpl idiom) -- keeps usearch/SQLite types out
// of the public header.
struct RetrievalEngine::Impl {
    std::size_t dimensions;
    sqlite3* db = nullptr;
    index_dense_t index;

    explicit Impl(std::size_t dim)
        : dimensions(dim), index(index_dense_t::make(metric_punned_t(dim, metric_kind_t::l2sq_k))) {}

    ~Impl() {
        if (db) sqlite3_close(db);
    }

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;
};

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

    // usearch requires capacity to be reserved ahead of insertion
    // (`size() < capacity()`); grow geometrically, same amortized-cost
    // strategy as std::vector, since callers add one vector at a time.
    if (impl_->index.size() == impl_->index.capacity()) {
        const std::size_t new_capacity = impl_->index.capacity() == 0 ? 64 : impl_->index.capacity() * 2;
        if (!impl_->index.reserve(index_limits_t(new_capacity)))
            throw std::runtime_error("RetrievalEngine::add_vector: failed to reserve usearch index capacity");
    }

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

}  // namespace retrieval_engine
