// Infrastructure sanity checks. Not about retrieval_engine's own logic:
// these exercise usearch and SQLite3 directly, independent of
// RetrievalEngine, so a failure here points at the toolchain/environment
// rather than our code (see docs/dev-log.md for the history).
#include <gtest/gtest.h>
#include <sqlite3.h>
#include <usearch/index_dense.hpp>

#include <cstdint>
#include <vector>

TEST(UsearchLibrary, IndexesAndQueriesASingleVectorDirectly) {
    using unum::usearch::index_dense_t;
    using unum::usearch::index_limits_t;
    using unum::usearch::metric_kind_t;
    using unum::usearch::metric_punned_t;

    constexpr std::size_t kDim = 4;
    constexpr std::uint64_t kKey = 42;

    auto index = index_dense_t::make(metric_punned_t(kDim, metric_kind_t::l2sq_k));
    // usearch requires capacity to be reserved ahead of insertion
    // (`size() < capacity()`); skipping this segfaults.
    ASSERT_TRUE(index.reserve(index_limits_t(1)));

    const std::vector<float> vector = {1.0f, 2.0f, 3.0f, 4.0f};
    const auto add_result = index.add(kKey, vector.data());
    ASSERT_TRUE(bool(add_result));

    const auto search_result = index.search(vector.data(), /*wanted=*/1);
    ASSERT_TRUE(bool(search_result));
    ASSERT_EQ(search_result.size(), 1u);

    std::uint64_t nearest_key = 0;
    search_result.dump_to(&nearest_key);
    EXPECT_EQ(nearest_key, kKey);
}

TEST(SqliteLibrary, CreatesDummyTableAndCountsInsertedRows) {
    sqlite3* db = nullptr;
    ASSERT_EQ(sqlite3_open(":memory:", &db), SQLITE_OK);

    char* error_message = nullptr;
    ASSERT_EQ(sqlite3_exec(db, "CREATE TABLE dummy (id INTEGER PRIMARY KEY);", nullptr, nullptr, &error_message),
              SQLITE_OK)
        << (error_message ? error_message : "unknown error");

    ASSERT_EQ(sqlite3_exec(db, "INSERT INTO dummy (id) VALUES (1), (2), (3);", nullptr, nullptr, &error_message),
              SQLITE_OK)
        << (error_message ? error_message : "unknown error");

    sqlite3_stmt* statement = nullptr;
    ASSERT_EQ(sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM dummy;", -1, &statement, nullptr), SQLITE_OK);
    ASSERT_EQ(sqlite3_step(statement), SQLITE_ROW);
    EXPECT_EQ(sqlite3_column_int(statement, 0), 3);
    sqlite3_finalize(statement);

    sqlite3_close(db);
}
