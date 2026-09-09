// Agent memory: exponential recency decay + temporal reranking.
//
// Tests two not-yet-implemented methods on RetrievalEngine:
// search_memory() and search_memory_explained(), declared in
// retrieval_engine.hpp but with NO implementation anywhere yet. Calling
// them is expected to make the build FAIL with an undefined-symbol
// (linker) error -- that is the correct, expected result for this pass.
// Do not "fix" it by adding an implementation; that is Phase 2 (GREEN).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <ctime>
#include <string>
#include <vector>

#include "retrieval_engine/retrieval_engine.hpp"

namespace {

constexpr std::size_t kDim = 4;

std::int64_t Now() { return static_cast<std::int64_t>(std::time(nullptr)); }

// A 2-document corpus, shared by both tests: "old" is closer to the query
// embedding (so it wins on *raw* hybrid similarity) but was created
// kOldAgeSeconds ago; "recent" is a hair further from the query embedding
// but created right now. Neither chunk's text matches the query text used
// below ("unrelated"), so search_sparse() contributes nothing to either --
// the fused score is dense-only, making "old"'s raw-similarity edge exact
// and easy to reason about by hand.
constexpr std::int64_t kOldAgeSeconds = 20;

void AddTwoChunkCorpus(retrieval_engine::RetrievalEngine& engine) {
    const std::int64_t now = Now();

    retrieval_engine::DocumentInput old_document;
    old_document.document_id = "old";
    old_document.metadata = "";
    old_document.chunks.push_back(retrieval_engine::DocumentChunkInput{
        "an old memory", {1.0f, 0.0f, 0.0f, 0.0f}, /*start_token=*/0, /*end_token=*/3,
        /*created_at_unix_seconds=*/now - kOldAgeSeconds});

    retrieval_engine::DocumentInput recent_document;
    recent_document.document_id = "recent";
    recent_document.metadata = "";
    recent_document.chunks.push_back(retrieval_engine::DocumentChunkInput{
        "a recent memory", {0.999f, 0.001f, 0.0f, 0.0f}, /*start_token=*/0, /*end_token=*/3,
        /*created_at_unix_seconds=*/now});

    engine.add_documents({old_document, recent_document});
}

}  // namespace

TEST(RetrievalEngineMemorySearch, RecencyFactorAndDecayedScoreMatchTheExponentialFormula) {
    const std::string db_path = "memory_test_decay.sqlite3";
    std::remove(db_path.c_str());

    retrieval_engine::RetrievalEngine engine(db_path, kDim);
    AddTwoChunkCorpus(engine);

    const std::vector<float> query_vec = {1.0f, 0.0f, 0.0f, 0.0f};
    constexpr float kRecencyWeight = 0.1f;

    const std::vector<retrieval_engine::SearchExplanation> explanations =
        engine.search_memory_explained("unrelated", query_vec, /*k=*/2, kRecencyWeight);
    ASSERT_EQ(explanations.size(), 2u);

    const auto find = [&](const std::string& document_id) {
        const auto it = std::find_if(explanations.begin(), explanations.end(),
                                      [&](const retrieval_engine::SearchExplanation& e) {
                                          return e.document_id == document_id;
                                      });
        return *it;
    };

    const retrieval_engine::SearchExplanation old_explanation = find("old");
    const retrieval_engine::SearchExplanation recent_explanation = find("recent");

    // "old" really is the higher-raw-similarity chunk before any decay is
    // applied -- confirms the test corpus is set up as intended.
    EXPECT_GT(old_explanation.fused_score, recent_explanation.fused_score);

    // "recent" was just created: age ~0, essentially no decay.
    EXPECT_NEAR(recent_explanation.age_seconds, 0.0, 2.0);
    EXPECT_NEAR(recent_explanation.recency_factor, 1.0, 0.01);

    // "old" is kOldAgeSeconds in the past -- verify the *exact* formula
    // (score = base_score * e^(-lambda * age)), not just "it decreased".
    EXPECT_NEAR(old_explanation.age_seconds, static_cast<double>(kOldAgeSeconds), 2.0);
    const double expected_old_recency_factor =
        std::exp(-static_cast<double>(kRecencyWeight) * static_cast<double>(kOldAgeSeconds));
    EXPECT_NEAR(old_explanation.recency_factor, expected_old_recency_factor, 1e-6);
    EXPECT_NEAR(static_cast<double>(old_explanation.decayed_score),
                static_cast<double>(old_explanation.fused_score) * expected_old_recency_factor, 1e-6);

    // The reranker's whole point: decay flips what raw similarity alone
    // ranked first.
    EXPECT_LT(old_explanation.decayed_score, recent_explanation.decayed_score);
}

TEST(RetrievalEngineMemorySearch, SearchMemoryRanksRecentDocumentAboveSlightlyMoreSimilarOlderOne) {
    const std::string db_path = "memory_test_rerank.sqlite3";
    std::remove(db_path.c_str());

    retrieval_engine::RetrievalEngine engine(db_path, kDim);
    AddTwoChunkCorpus(engine);

    const std::vector<float> query_vec = {1.0f, 0.0f, 0.0f, 0.0f};

    // recency_weight = 0 disables decay entirely: raw similarity wins,
    // "old" ranks first (same ranking search_hybrid() would give).
    const std::vector<retrieval_engine::ChunkSearchResult> undecayed =
        engine.search_memory("unrelated", query_vec, /*k=*/2, /*recency_weight=*/0.0f);
    ASSERT_EQ(undecayed.size(), 2u);
    EXPECT_EQ(undecayed.front().document_id, "old");

    // With a non-zero recency_weight, the fresher document overtakes it --
    // this is the temporal reranker actually reordering results, not just
    // adjusting scores in place.
    const std::vector<retrieval_engine::ChunkSearchResult> decayed =
        engine.search_memory("unrelated", query_vec, /*k=*/2, /*recency_weight=*/0.1f);
    ASSERT_EQ(decayed.size(), 2u);
    EXPECT_EQ(decayed.front().document_id, "recent");
}
