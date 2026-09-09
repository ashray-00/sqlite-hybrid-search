// Stage 0 (BUILD_PLAN.md).
//
// This file has the following test cases:
//
//  1. UsearchLibrary.*  -- exercises usearch directly (index a vector, query
//     it back). Proves the FetchContent-based usearch dependency actually
//     compiles and works on this platform.
//  2. SqliteLibrary.*   -- exercises SQLite directly (create a dummy table,
//     insert rows, count them). Proves SQLite3 is linked correctly.
//  3. RetrievalEngineStage0.* -- exercises retrieval_engine::RetrievalEngine
//     (core/include/retrieval_engine/retrieval_engine.hpp,
//     core/src/retrieval_engine.cpp), the actual Stage 0 deliverable: the
//     happy path (insert vectors, query back the known nearest neighbour,
//     SQLite row count matches) plus its defensive-validation edge cases.
#include <gtest/gtest.h>
#include <sqlite3.h>
#include <usearch/index_dense.hpp>

#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "retrieval_engine/retrieval_engine.hpp"

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

TEST(RetrievalEngineStage0, InsertsRandomVectorsAndReturnsKnownNearestNeighbour) {
    const std::string db_path = "stage0_test.sqlite3";
    std::remove(db_path.c_str());  // start from a clean database each run

    constexpr std::size_t kDim = 8;
    constexpr std::size_t kNumRandomVectors = 20;
    constexpr std::uint64_t kTargetId = 999;

    retrieval_engine::RetrievalEngine engine(db_path, kDim);

    // Insert a cluster of random "noise" vectors near the origin.
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> noise(-1.0f, 1.0f);
    for (std::uint64_t id = 0; id < kNumRandomVectors; ++id) {
        std::vector<float> vector(kDim);
        for (auto& value : vector) value = noise(rng);
        engine.add_vector(id, vector);
    }

    // A vector far outside the noise cluster: querying with an exact copy of
    // it must unambiguously return itself as the nearest neighbour.
    const std::vector<float> target_vector(kDim, 10.0f);
    engine.add_vector(kTargetId, target_vector);

    const std::vector<std::uint64_t> neighbours = engine.search(target_vector, /*k=*/1);
    ASSERT_EQ(neighbours.size(), 1u);
    EXPECT_EQ(neighbours[0], kTargetId);

    EXPECT_EQ(engine.dummy_table_row_count(), kNumRandomVectors + 1);
}

TEST(RetrievalEngineStage0, ConstructorRejectsZeroDimension) {
    const std::string db_path = "stage0_test_zero_dim.sqlite3";
    std::remove(db_path.c_str());

    EXPECT_THROW(retrieval_engine::RetrievalEngine(db_path, /*dim=*/0), std::invalid_argument);
}

TEST(RetrievalEngineStage0, AddVectorRejectsMismatchedDimension) {
    const std::string db_path = "stage0_test_add_mismatch.sqlite3";
    std::remove(db_path.c_str());

    retrieval_engine::RetrievalEngine engine(db_path, /*dim=*/8);
    const std::vector<float> wrong_size_vector(4, 1.0f);

    EXPECT_THROW(engine.add_vector(0, wrong_size_vector), std::invalid_argument);
}

TEST(RetrievalEngineStage0, SearchRejectsMismatchedDimension) {
    const std::string db_path = "stage0_test_search_mismatch.sqlite3";
    std::remove(db_path.c_str());

    retrieval_engine::RetrievalEngine engine(db_path, /*dim=*/8);
    const std::vector<float> wrong_size_query(4, 1.0f);

    EXPECT_THROW(engine.search(wrong_size_query, /*k=*/1), std::invalid_argument);
}
