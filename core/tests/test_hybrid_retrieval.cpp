// Hybrid retrieval: sparse (FTS5/BM25) search fused with dense search via
// Reciprocal Rank Fusion, plus the explainable score breakdown.
//
//  1. FtsLibrary.* -- exercises SQLite FTS5 directly (create a virtual
//     table, index rows, rank by bm25()). FTS5 is an optional, compile-time
//     SQLite feature -- not guaranteed by every build -- so this proves the
//     linked SQLite3 actually has it, independent of anything below.
//  2. RetrievalEngineHybridSearch.* -- exercises RetrievalEngine's
//     search_dense()/search_sparse()/search_hybrid()/search_explained()
//     (core/src/detail/chunk_store.*, chunk_repository.*, rrf_fusion.*,
//     fts5_query.*).
#include <gtest/gtest.h>
#include <sqlite3.h>

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "retrieval_engine/retrieval_engine.hpp"

TEST(FtsLibrary, RanksRowsByBm25Directly) {
    sqlite3* db = nullptr;
    ASSERT_EQ(sqlite3_open(":memory:", &db), SQLITE_OK);

    char* error_message = nullptr;
    ASSERT_EQ(sqlite3_exec(db, "CREATE VIRTUAL TABLE docs USING fts5(text);", nullptr, nullptr, &error_message),
              SQLITE_OK)
        << (error_message ? error_message : "unknown error");

    ASSERT_EQ(sqlite3_exec(db,
                           "INSERT INTO docs(rowid, text) VALUES "
                           "(1, 'the quick brown fox'), "
                           "(2, 'a lazy dog sleeps'), "
                           "(3, 'fox fox fox everywhere');",
                           nullptr, nullptr, &error_message),
              SQLITE_OK)
        << (error_message ? error_message : "unknown error");

    sqlite3_stmt* statement = nullptr;
    ASSERT_EQ(sqlite3_prepare_v2(db, "SELECT rowid, bm25(docs) FROM docs WHERE docs MATCH 'fox' ORDER BY bm25(docs);",
                                 -1, &statement, nullptr),
              SQLITE_OK);

    // Row 3 mentions "fox" three times, so BM25 must rank it best (lowest
    // score -- FTS5's convention: lower is more relevant) ahead of row 1's
    // single mention. Row 2 never mentions "fox" and must not appear at all.
    ASSERT_EQ(sqlite3_step(statement), SQLITE_ROW);
    EXPECT_EQ(sqlite3_column_int64(statement, 0), 3);
    const double best_score = sqlite3_column_double(statement, 1);

    ASSERT_EQ(sqlite3_step(statement), SQLITE_ROW);
    EXPECT_EQ(sqlite3_column_int64(statement, 0), 1);
    const double second_score = sqlite3_column_double(statement, 1);
    EXPECT_LT(best_score, second_score);

    EXPECT_EQ(sqlite3_step(statement), SQLITE_DONE);  // no third row: row 2 never matched

    sqlite3_finalize(statement);
    sqlite3_close(db);
}

namespace {

constexpr std::size_t kDim = 4;
const std::vector<float> kQueryVector = {1.0f, 0.0f, 0.0f, 0.0f};

// A corpus crafted so dense and sparse search sharply disagree: four
// "noise" documents whose embeddings sit close to the query direction but
// whose text has nothing to do with it, and one "target" document whose
// text contains a unique, exact term a keyword search finds trivially, but
// whose embedding is orthogonal to the query -- dense search alone ranks it
// dead last. Hybrid (RRF) fusion is expected to recover it to rank 1: see
// the RRF arithmetic in RetrievalEngineHybridSearch test comments below.
std::vector<retrieval_engine::DocumentInput> MakeCorpusWhereSparseAndDenseDisagree() {
    std::vector<retrieval_engine::DocumentInput> documents;

    const std::vector<std::string> noise_texts = {
        "the weather today is mild and pleasant",
        "stock markets closed slightly higher today",
        "the recipe calls for two cups of flour",
        "a gentle breeze moved through the trees",
    };
    for (std::size_t i = 0; i < noise_texts.size(); ++i) {
        retrieval_engine::DocumentInput document;
        document.document_id = "noise-" + std::to_string(i);
        document.metadata = "source:test";
        // Near-parallel to the query direction {1,0,0,0}, with a tiny
        // per-document perturbation in an unused dimension so dense search
        // ranks them deterministically: smaller index -> closer -> better
        // dense rank (1 through 4).
        document.chunks.push_back(
            retrieval_engine::DocumentChunkInput{noise_texts[i],
                                                 {1.0f, 0.001f * static_cast<float>(i + 1), 0.0f, 0.0f},
                                                 /*start_token=*/0,
                                                 /*end_token=*/1});
        documents.push_back(std::move(document));
    }

    retrieval_engine::DocumentInput target;
    target.document_id = "target";
    target.metadata = "source:test";
    // Orthogonal to the query direction -- cosine similarity exactly 0, the
    // worst of all five documents, so dense search ranks it 5th -- but its
    // text contains a unique exact term a keyword search finds unambiguously.
    target.chunks.push_back(retrieval_engine::DocumentChunkInput{"reference part number ZXQ7742 in stock",
                                                                 {0.0f, 1.0f, 0.0f, 0.0f},
                                                                 /*start_token=*/0,
                                                                 /*end_token=*/4});
    documents.push_back(std::move(target));

    return documents;
}

}  // namespace

TEST(RetrievalEngineHybridSearch, EveryChunkIsSearchableBySparseText) {
    const std::string db_path = "hybrid_retrieval_test_sync.sqlite3";
    std::remove(db_path.c_str());

    retrieval_engine::RetrievalEngine engine(db_path, kDim);
    engine.add_documents(MakeCorpusWhereSparseAndDenseDisagree());

    // Every chunk add_documents() wrote must be findable by a word unique
    // to its own text -- proof the FTS5 index was synced for every row, not
    // just the one with the obviously distinctive term.
    EXPECT_EQ(engine.search_sparse("weather", 5).size(), 1u);
    EXPECT_EQ(engine.search_sparse("markets", 5).size(), 1u);
    EXPECT_EQ(engine.search_sparse("recipe", 5).size(), 1u);
    EXPECT_EQ(engine.search_sparse("breeze", 5).size(), 1u);
    EXPECT_EQ(engine.search_sparse("ZXQ7742", 5).size(), 1u);
}

// Regression test for a BLOCKER caught in Phase 3's independent review:
// FTS5's MATCH operand is parsed by FTS5's own query grammar (AND/OR/NOT, a
// leading '-' meaning NOT, quoted phrases, "column:" filters), not treated
// as literal text. A hyphenated word or an unbalanced quote -- both
// completely ordinary in real search input -- must not throw, and must
// still find the intended chunk by its literal text.
TEST(RetrievalEngineHybridSearch, SearchSparseTreatsSpecialCharactersAsLiteralText) {
    const std::string db_path = "hybrid_retrieval_test_sanitization.sqlite3";
    std::remove(db_path.c_str());

    retrieval_engine::RetrievalEngine engine(db_path, kDim);
    engine.add_documents(MakeCorpusWhereSparseAndDenseDisagree());

    // A leading '-' is FTS5's NOT operator outside quotes; unescaped, this
    // raises a SQLite error ("no such column: ZXQ7742") rather than
    // matching. Hyphenated compound terms (product codes, negative
    // numbers...) are completely ordinary in real search input.
    std::vector<retrieval_engine::ChunkSearchResult> results;
    EXPECT_NO_THROW(results = engine.search_sparse("-ZXQ7742", 5));
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].document_id, "target");

    // An unbalanced double quote raises "unterminated string" outside a
    // properly escaped phrase.
    EXPECT_NO_THROW(results = engine.search_sparse("ZXQ7742 \"unterminated", 5));
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].document_id, "target");

    // A query with no terms at all (only whitespace) matches nothing rather
    // than producing a malformed empty MATCH expression.
    EXPECT_NO_THROW(results = engine.search_sparse("   ", 5));
    EXPECT_TRUE(results.empty());
}

TEST(RetrievalEngineHybridSearch, HybridRanksExactTermMatchHigherThanDenseAlone) {
    const std::string db_path = "hybrid_retrieval_test_hybrid.sqlite3";
    std::remove(db_path.c_str());

    retrieval_engine::RetrievalEngine engine(db_path, kDim);
    engine.add_documents(MakeCorpusWhereSparseAndDenseDisagree());

    constexpr std::size_t kTopK = 5;

    // Dense-only: "target" (orthogonal to the query) ranks last of five.
    const std::vector<retrieval_engine::ChunkSearchResult> dense_results = engine.search_dense(kQueryVector, kTopK);
    ASSERT_EQ(dense_results.size(), kTopK);
    EXPECT_EQ(dense_results.back().document_id, "target");

    // Sparse-only: "target" is the only chunk containing "ZXQ7742" at all.
    const std::vector<retrieval_engine::ChunkSearchResult> sparse_results = engine.search_sparse("ZXQ7742", kTopK);
    ASSERT_EQ(sparse_results.size(), 1u);
    EXPECT_EQ(sparse_results[0].document_id, "target");

    // RRF (k=60) arithmetic: target's fused score is
    // 1/(60+5) [dense rank 5] + 1/(60+1) [sparse rank 1] =~ 0.03177.
    // Every noise document only appears in the dense ranking (1..4), giving
    // at most 1/(60+1) =~ 0.01639 -- roughly half of target's score. So
    // target must come out on top once fused, despite dense search alone
    // ranking it dead last.
    const std::vector<retrieval_engine::ChunkSearchResult> hybrid_results =
        engine.search_hybrid("ZXQ7742", kQueryVector, kTopK);
    ASSERT_FALSE(hybrid_results.empty());
    EXPECT_EQ(hybrid_results.front().document_id, "target");
}

TEST(RetrievalEngineHybridSearch, SearchExplainedReturnsFullScoreBreakdown) {
    const std::string db_path = "hybrid_retrieval_test_explain.sqlite3";
    std::remove(db_path.c_str());

    retrieval_engine::RetrievalEngine engine(db_path, kDim);
    engine.add_documents(MakeCorpusWhereSparseAndDenseDisagree());

    constexpr std::size_t kTopK = 5;
    const std::vector<retrieval_engine::SearchExplanation> explanations =
        engine.search_explained("ZXQ7742", kQueryVector, kTopK);
    ASSERT_FALSE(explanations.empty());

    // Results must be ordered by final_rank: 1-based and contiguous.
    for (std::size_t i = 0; i < explanations.size(); ++i) {
        EXPECT_EQ(explanations[i].final_rank, i + 1);
    }

    const auto target_it =
        std::find_if(explanations.begin(), explanations.end(),
                     [](const retrieval_engine::SearchExplanation& e) { return e.document_id == "target"; });
    ASSERT_NE(target_it, explanations.end());

    // "target" hits both rankings: full breakdown on both sides.
    EXPECT_EQ(target_it->final_rank, 1u);
    EXPECT_TRUE(target_it->dense_present);
    EXPECT_EQ(target_it->dense_rank, 5u);
    EXPECT_TRUE(target_it->sparse_present);
    EXPECT_EQ(target_it->sparse_rank, 1u);
    EXPECT_GT(target_it->fused_score, 0.0f);

    // A noise document: present in the dense ranking, absent from sparse --
    // and correctly reported as such rather than a default/garbage value.
    const auto noise_it =
        std::find_if(explanations.begin(), explanations.end(),
                     [](const retrieval_engine::SearchExplanation& e) { return e.document_id == "noise-0"; });
    ASSERT_NE(noise_it, explanations.end());
    EXPECT_TRUE(noise_it->dense_present);
    EXPECT_FALSE(noise_it->sparse_present);
    EXPECT_EQ(noise_it->sparse_rank, 0u);

    // RRF must have actually mattered here, not just echoed dense order.
    EXPECT_LT(noise_it->fused_score, target_it->fused_score);
}
