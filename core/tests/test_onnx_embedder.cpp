// Real ONNX Runtime embedding backend: end-to-end checks against an actual
// all-MiniLM-L6-v2 model.
//
// The 90 MB model is not vendored into the repo, so every test here SKIPS
// (rather than fails) when it can't be found. Point it at a model with the
// RETRIEVAL_ENGINE_TEST_ONNX_MODEL environment variable, or place it at
//   ~/.cache/retrieval-engine/all-MiniLM-L6-v2/model.onnx
// with its vocab.txt beside it. When the whole build is configured without
// ONNX Runtime (-DRETRIEVAL_ENGINE_WITH_ONNX=OFF) the file still compiles,
// as a single skipped test.
#include <gtest/gtest.h>

#ifndef RETRIEVAL_ENGINE_WITH_ONNX

TEST(OnnxEmbedder, SkippedBecauseBuiltWithoutOnnxRuntime) {
    GTEST_SKIP() << "configured with RETRIEVAL_ENGINE_WITH_ONNX=OFF";
}

#else

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "retrieval_engine/retrieval_engine.hpp"

namespace {

constexpr std::size_t kMiniLmDim = 384;

std::string ResolveModelPath() {
    if (const char* from_env = std::getenv("RETRIEVAL_ENGINE_TEST_ONNX_MODEL")) {
        return from_env;
    }
    if (const char* home = std::getenv("HOME")) {
        return std::string(home) + "/.cache/retrieval-engine/all-MiniLM-L6-v2/model.onnx";
    }
    return "";
}

double Cosine(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    return (norm_a == 0.0 || norm_b == 0.0) ? 0.0 : dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

class OnnxEmbedderTest : public ::testing::Test {
protected:
    void SetUp() override {
        model_path_ = ResolveModelPath();
        if (model_path_.empty() || !std::filesystem::exists(model_path_)) {
            GTEST_SKIP() << "all-MiniLM-L6-v2 ONNX model not found at '" << model_path_
                         << "' -- set RETRIEVAL_ENGINE_TEST_ONNX_MODEL or download it to run this test";
        }
    }

    retrieval_engine::RetrievalEngine MakeEngine(const std::string& db_path) {
        std::remove(db_path.c_str());
        retrieval_engine::RetrievalEngine engine(db_path, kMiniLmDim);
        engine.load_embedding_model(model_path_);
        return engine;
    }

    std::string model_path_;
};

}  // namespace

TEST_F(OnnxEmbedderTest, EmbedProducesUnitLength384DimVector) {
    retrieval_engine::RetrievalEngine engine = MakeEngine("onnx_embedder_dim.sqlite3");
    EXPECT_EQ(engine.embedding_dim(), kMiniLmDim);

    const std::vector<float> embedding = engine.embed("Where do I live?");
    ASSERT_EQ(embedding.size(), kMiniLmDim);

    double sum_of_squares = 0.0;
    for (const float value : embedding) sum_of_squares += static_cast<double>(value) * value;
    EXPECT_NEAR(std::sqrt(sum_of_squares), 1.0, 1e-4);
}

TEST_F(OnnxEmbedderTest, EmbedIsDeterministic) {
    retrieval_engine::RetrievalEngine engine = MakeEngine("onnx_embedder_determinism.sqlite3");

    const std::vector<float> first = engine.embed("a repeatable sentence");
    const std::vector<float> second = engine.embed("a repeatable sentence");
    ASSERT_EQ(first.size(), second.size());
    for (std::size_t i = 0; i < first.size(); ++i) {
        EXPECT_FLOAT_EQ(first[i], second[i]);
    }
}

TEST_F(OnnxEmbedderTest, SemanticSimilarityRanksParaphraseAboveUnrelatedText) {
    retrieval_engine::RetrievalEngine engine = MakeEngine("onnx_embedder_semantics.sqlite3");

    const std::vector<float> query = engine.embed("Where do I live?");
    const std::vector<float> paraphrase = engine.embed("Which city is my home in?");
    const std::vector<float> unrelated = engine.embed("The recipe needs two cups of flour.");

    EXPECT_GT(Cosine(query, paraphrase), Cosine(query, unrelated));
    EXPECT_GT(Cosine(query, paraphrase), 0.5);
}

TEST_F(OnnxEmbedderTest, DimensionMismatchBetweenModelAndEngineThrowsInvalidArgument) {
    std::remove("onnx_embedder_dim_mismatch.sqlite3");
    // MiniLM outputs 384; build the engine for something else.
    retrieval_engine::RetrievalEngine engine("onnx_embedder_dim_mismatch.sqlite3", /*dim=*/128);

    EXPECT_THROW(engine.load_embedding_model(model_path_), std::invalid_argument);
    EXPECT_FALSE(engine.has_embedding_model());
}

TEST_F(OnnxEmbedderTest, SearchTextRetrievesTheSemanticallyRelevantDocument) {
    retrieval_engine::RetrievalEngine engine = MakeEngine("onnx_embedder_search.sqlite3");

    engine.add_text({
        retrieval_engine::TextDocumentInput{"home", "I live in Berlin, the capital of Germany.", "", 0},
        retrieval_engine::TextDocumentInput{"weather",
                                            "Heavy rain is expected across the region tomorrow.", "", 0},
        retrieval_engine::TextDocumentInput{"food",
                                            "This pasta recipe calls for garlic and olive oil.", "", 0},
    });
    ASSERT_EQ(engine.chunk_count(), 3u);

    // No shared keywords with "home"'s text -- this only works if the query
    // and document embed to nearby vectors.
    const std::vector<retrieval_engine::ChunkSearchResult> results =
        engine.search_text("Which city is my house located in?", /*k=*/1);

    ASSERT_FALSE(results.empty());
    EXPECT_EQ(results.front().document_id, "home");
}

#endif  // RETRIEVAL_ENGINE_WITH_ONNX
