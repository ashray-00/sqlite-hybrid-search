// Agent memory: exponential recency decay + temporal reranking.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
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
//
// The decay formula normalizes age to days (age_seconds / 86400), so a
// meaningful demonstration needs a day-scale gap, not a few seconds: one
// simulated day (86400 seconds) with decay_lambda=0.1 gives
// e^(-0.1*1) = e^-0.1 ~= 0.905, comfortably enough to flip the ranking
// against "old"'s small raw-similarity edge (worked out by hand below).
constexpr std::int64_t kOldAgeSeconds = 86400;

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
    constexpr float kDecayLambda = 0.1f;

    const std::vector<retrieval_engine::SearchExplanation> explanations =
        engine.search_memory_explained("unrelated", query_vec, /*k=*/2, kDecayLambda);
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
    // (decayed_score = base_score * e^(-lambda * age_days)), not just "it
    // decreased".
    EXPECT_NEAR(old_explanation.age_seconds, static_cast<double>(kOldAgeSeconds), 2.0);
    constexpr double kSecondsPerDay = 86400.0;
    const double expected_old_recency_factor = std::exp(
        -static_cast<double>(kDecayLambda) * (static_cast<double>(kOldAgeSeconds) / kSecondsPerDay));
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

    // decay_lambda = 0 disables decay entirely: raw similarity wins, "old"
    // ranks first (same ranking search_hybrid() would give).
    const std::vector<retrieval_engine::ChunkSearchResult> undecayed =
        engine.search_memory("unrelated", query_vec, /*k=*/2, /*decay_lambda=*/0.0f);
    ASSERT_EQ(undecayed.size(), 2u);
    EXPECT_EQ(undecayed.front().document_id, "old");

    // With a non-zero decay_lambda, the fresher document overtakes it --
    // this is the temporal reranker actually reordering results, not just
    // adjusting scores in place.
    const std::vector<retrieval_engine::ChunkSearchResult> decayed =
        engine.search_memory("unrelated", query_vec, /*k=*/2, /*decay_lambda=*/0.1f);
    ASSERT_EQ(decayed.size(), 2u);
    EXPECT_EQ(decayed.front().document_id, "recent");
}

// Independent review regression: a clock that stepped backwards (or a
// caller-supplied created_at that turns out to be in the future) produces a
// negative age_seconds. Left unguarded, e^(-lambda * negative) > 1 would let
// "decay" actually *inflate* a score above its undecayed value -- verify
// that can never happen.
TEST(RetrievalEngineMemorySearch, FutureCreatedAtNeverInflatesTheRecencyFactorAboveOne) {
    const std::string db_path = "memory_test_future_timestamp.sqlite3";
    std::remove(db_path.c_str());

    retrieval_engine::RetrievalEngine engine(db_path, kDim);

    const std::int64_t ten_days_in_the_future = Now() + 10 * kOldAgeSeconds;
    retrieval_engine::DocumentInput future_document;
    future_document.document_id = "from_the_future";
    future_document.metadata = "";
    future_document.chunks.push_back(retrieval_engine::DocumentChunkInput{
        "a memory with a clock-skewed timestamp", {1.0f, 0.0f, 0.0f, 0.0f}, /*start_token=*/0, /*end_token=*/6,
        /*created_at_unix_seconds=*/ten_days_in_the_future});
    engine.add_documents({future_document});

    const std::vector<retrieval_engine::SearchExplanation> explanations =
        engine.search_memory_explained("unrelated", {1.0f, 0.0f, 0.0f, 0.0f}, /*k=*/1, /*decay_lambda=*/0.5f);
    ASSERT_EQ(explanations.size(), 1u);

    // The raw age is still reported honestly (negative, for debugging/
    // transparency)...
    EXPECT_LT(explanations.front().age_seconds, 0.0);
    // ...but the decay math itself must clamp it: no boost above 1.0, and
    // therefore no decayed_score above the undecayed fused_score.
    EXPECT_LE(explanations.front().recency_factor, 1.0);
    EXPECT_LE(explanations.front().decayed_score, explanations.front().fused_score);
}

// Independent review regression: an old memory decayed with a large
// decay_lambda must be heavily discounted, but never scored as if it did
// not exist at all -- the recency factor has a floor.
TEST(RetrievalEngineMemorySearch, VeryOldMemoryWithAggressiveDecayFloorsRatherThanZeroingOut) {
    const std::string db_path = "memory_test_floor.sqlite3";
    std::remove(db_path.c_str());

    retrieval_engine::RetrievalEngine engine(db_path, kDim);

    constexpr std::int64_t kOneYearInSeconds = 365 * kOldAgeSeconds;
    retrieval_engine::DocumentInput ancient_document;
    ancient_document.document_id = "ancient";
    ancient_document.metadata = "";
    ancient_document.chunks.push_back(retrieval_engine::DocumentChunkInput{
        "a very old critical memory", {1.0f, 0.0f, 0.0f, 0.0f}, /*start_token=*/0, /*end_token=*/5,
        /*created_at_unix_seconds=*/Now() - kOneYearInSeconds});
    engine.add_documents({ancient_document});

    // decay_lambda=10 over a year of age would drive an unclamped
    // e^(-10 * 365) to a value indistinguishable from 0 in double precision
    // -- exactly the "critical memory zeroed out" failure mode the floor
    // exists to prevent.
    const std::vector<retrieval_engine::SearchExplanation> explanations =
        engine.search_memory_explained("unrelated", {1.0f, 0.0f, 0.0f, 0.0f}, /*k=*/1, /*decay_lambda=*/10.0f);
    ASSERT_EQ(explanations.size(), 1u);

    EXPECT_GE(explanations.front().recency_factor, 0.01);
    EXPECT_TRUE(std::isfinite(explanations.front().recency_factor));
    EXPECT_TRUE(std::isfinite(explanations.front().decayed_score));
}
