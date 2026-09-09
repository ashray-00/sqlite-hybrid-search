// Built-in local embedding inference: the zero-setup "just give it text" path.
//
// Tests five not-yet-implemented members of RetrievalEngine --
// load_embedding_model(), has_embedding_model(), embedding_dim(), embed(),
// add_text() and search_text() -- declared in retrieval_engine.hpp but
// with NO implementation anywhere yet. Building this test is expected to
// FAIL at link time with undefined-symbol errors: that is the correct,
// expected result for this pass (Phase 1, RED). Do NOT fix it by writing
// an implementation, a tokenizer, or a text-to-vector pipeline; that is
// Phase 2 (GREEN).
//
// Model file: rather than shipping a real ONNX/GGUF model into the test
// tree, these tests write a tiny *mock* model file (see WriteMockModel()
// and docs/DECISIONS.md for its documented contract). GREEN is expected to
// recognize that mock header and activate a deterministic, dependency-free
// test embedder (lexical-overlap cosine similarity) so the end-to-end text
// path can be exercised here without a multi-megabyte download. Loading a
// genuine ONNX/GGUF model is a separate, model-availability-gated path and
// is out of scope for this pass.
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "retrieval_engine/retrieval_engine.hpp"

namespace {

// all-MiniLM-L6-v2's output width; also the dimension the mock model
// advertises. 768 (nomic-embed-text) would work equally well -- the tests
// only require embed()'s output to match this exactly.
constexpr std::size_t kEmbeddingDim = 384;

// Writes the mock embedding-model file the tests point load_embedding_model()
// at. Contract (also recorded in docs/DECISIONS.md): a file whose first
// line is exactly "RETRIEVAL_ENGINE_MOCK_EMBEDDING_MODEL v1" followed by a
// "dim=<n>" line selects a built-in deterministic test embedder instead of
// a real ONNX/GGUF runtime.
void WriteMockModel(const std::string& path, std::size_t dim) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out.is_open()) << "could not create mock model file at " << path;
    out << "RETRIEVAL_ENGINE_MOCK_EMBEDDING_MODEL v1\n";
    out << "dim=" << dim << "\n";
}

double CosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    EXPECT_EQ(a.size(), b.size());
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
        norm_a += static_cast<double>(a[i]) * static_cast<double>(a[i]);
        norm_b += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    if (norm_a == 0.0 || norm_b == 0.0) {
        return 0.0;
    }
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

retrieval_engine::RetrievalEngine MakeEngineWithMockModel(const std::string& db_path,
                                                          const std::string& model_path) {
    std::remove(db_path.c_str());
    WriteMockModel(model_path, kEmbeddingDim);

    retrieval_engine::RetrievalEngine engine(db_path, kEmbeddingDim);
    engine.load_embedding_model(model_path);
    EXPECT_TRUE(engine.has_embedding_model());
    EXPECT_EQ(engine.embedding_dim(), kEmbeddingDim);
    return engine;
}

}  // namespace

// embed("Hello world") must return a non-empty vector of exactly the
// model's dimension.
TEST(BuiltinEmbedder, EmbedProducesNonEmptyVectorOfExactModelDimension) {
    retrieval_engine::RetrievalEngine engine =
        MakeEngineWithMockModel("builtin_embedder_embed.sqlite3", "builtin_embedder_mock_model.txt");

    const std::vector<float> embedding = engine.embed("Hello world");

    ASSERT_EQ(embedding.size(), kEmbeddingDim);
    const double magnitude = std::sqrt(std::inner_product(embedding.begin(), embedding.end(),
                                                          embedding.begin(), 0.0));
    EXPECT_GT(magnitude, 0.0) << "embedding must not be the all-zeros vector";
}

// The raw-text ingestion + query path: add_text() / search_text() with no
// caller-supplied float vectors anywhere. The document that shares
// vocabulary with the query must come back first.
TEST(BuiltinEmbedder, AddTextAndSearchTextRoundTripWithoutCallerSuppliedVectors) {
    retrieval_engine::RetrievalEngine engine =
        MakeEngineWithMockModel("builtin_embedder_search_text.sqlite3", "builtin_embedder_mock_model.txt");

    engine.add_text({
        retrieval_engine::TextDocumentInput{"install-guide",
                                            "python package installation guide for beginners", "", 0},
        retrieval_engine::TextDocumentInput{"weather-report",
                                            "mountain weather forecast cold and snowy today", "", 0},
        retrieval_engine::TextDocumentInput{"cooking-notes",
                                            "recipe roasted vegetables garlic thyme dinner", "", 0},
    });
    ASSERT_EQ(engine.chunk_count(), 3u);

    const std::vector<retrieval_engine::ChunkSearchResult> results =
        engine.search_text("the python package installation guide", /*k=*/2);

    ASSERT_FALSE(results.empty());
    EXPECT_EQ(results.front().document_id, "install-guide");
}

// Semantically related text (shared vocabulary) must have higher cosine
// similarity than unrelated text, end to end through embed().
TEST(BuiltinEmbedder, SemanticallyRelatedTextHasHigherCosineSimilarity) {
    retrieval_engine::RetrievalEngine engine =
        MakeEngineWithMockModel("builtin_embedder_cosine.sqlite3", "builtin_embedder_mock_model.txt");

    const std::vector<float> query = engine.embed("the python package installation guide");
    const std::vector<float> related = engine.embed("python package installation guide for beginners");
    const std::vector<float> unrelated = engine.embed("mountain weather forecast cold and snowy");

    const double related_similarity = CosineSimilarity(query, related);
    const double unrelated_similarity = CosineSimilarity(query, unrelated);

    EXPECT_GT(related_similarity, unrelated_similarity);
    EXPECT_GT(related_similarity, 0.5);
}

// embed() before any model is attached is a programming error, not a
// silent empty result.
TEST(BuiltinEmbedder, EmbedWithoutAModelThrows) {
    std::remove("builtin_embedder_no_model.sqlite3");
    retrieval_engine::RetrievalEngine engine("builtin_embedder_no_model.sqlite3", kEmbeddingDim);

    EXPECT_FALSE(engine.has_embedding_model());
    EXPECT_THROW(engine.embed("no model loaded"), std::logic_error);
}

// --- Model-path resolution / corrupted-file handling ---------------------

TEST(BuiltinEmbedder, LoadEmbeddingModelWithMissingFileThrowsRuntimeError) {
    std::remove("builtin_embedder_missing.sqlite3");
    retrieval_engine::RetrievalEngine engine("builtin_embedder_missing.sqlite3", kEmbeddingDim);

    EXPECT_THROW(engine.load_embedding_model("no_such_model_file_98765.bin"), std::runtime_error);
    EXPECT_FALSE(engine.has_embedding_model());
}

TEST(BuiltinEmbedder, LoadEmbeddingModelWithUnrecognizedFormatThrowsRuntimeError) {
    const std::string model_path = "builtin_embedder_bad_format.bin";
    {
        std::ofstream out(model_path, std::ios::binary | std::ios::trunc);
        out << "\x00\x01 this is not any embedding model format we know\n";
    }
    std::remove("builtin_embedder_badfmt.sqlite3");
    retrieval_engine::RetrievalEngine engine("builtin_embedder_badfmt.sqlite3", kEmbeddingDim);

    EXPECT_THROW(engine.load_embedding_model(model_path), std::runtime_error);
    EXPECT_FALSE(engine.has_embedding_model());
}

TEST(BuiltinEmbedder, LoadEmbeddingModelWithCorruptDimLineThrowsRuntimeError) {
    const std::string model_path = "builtin_embedder_corrupt_dim.txt";
    {
        std::ofstream out(model_path, std::ios::binary | std::ios::trunc);
        out << "RETRIEVAL_ENGINE_MOCK_EMBEDDING_MODEL v1\n";
        out << "dim=not-a-number\n";
    }
    std::remove("builtin_embedder_corruptdim.sqlite3");
    retrieval_engine::RetrievalEngine engine("builtin_embedder_corruptdim.sqlite3", kEmbeddingDim);

    EXPECT_THROW(engine.load_embedding_model(model_path), std::runtime_error);
    EXPECT_FALSE(engine.has_embedding_model());
}

TEST(BuiltinEmbedder, LoadEmbeddingModelWithDimensionMismatchThrowsInvalidArgument) {
    const std::string model_path = "builtin_embedder_dim_mismatch.txt";
    WriteMockModel(model_path, kEmbeddingDim + 1);  // model says 385, engine wants 384
    std::remove("builtin_embedder_dimmismatch.sqlite3");
    retrieval_engine::RetrievalEngine engine("builtin_embedder_dimmismatch.sqlite3", kEmbeddingDim);

    EXPECT_THROW(engine.load_embedding_model(model_path), std::invalid_argument);
    EXPECT_FALSE(engine.has_embedding_model());
}

// A ".onnx" model needs its vocab.txt beside it. Whether or not this build
// has the ONNX backend, pointing the loader at an .onnx file with no usable
// vocab is a std::runtime_error (never a crash, never silent).
TEST(BuiltinEmbedder, LoadOnnxModelWithoutAUsableSiblingVocabThrowsRuntimeError) {
    namespace fs = std::filesystem;
    const fs::path dir = "builtin_embedder_onnx_no_vocab_dir";
    fs::remove_all(dir);
    fs::create_directory(dir);
    {
        std::ofstream model(dir / "model.onnx", std::ios::binary | std::ios::trunc);
        model << "this is not a real ONNX protobuf";
    }

    std::remove("builtin_embedder_onnx_no_vocab.sqlite3");
    retrieval_engine::RetrievalEngine engine("builtin_embedder_onnx_no_vocab.sqlite3", kEmbeddingDim);

    EXPECT_THROW(engine.load_embedding_model((dir / "model.onnx").string()), std::runtime_error);
    EXPECT_FALSE(engine.has_embedding_model());

    fs::remove_all(dir);
}

// A corrupt .onnx file (vocab present) is likewise a clean std::runtime_error.
TEST(BuiltinEmbedder, LoadCorruptOnnxModelThrowsRuntimeError) {
    namespace fs = std::filesystem;
    const fs::path dir = "builtin_embedder_onnx_corrupt_dir";
    fs::remove_all(dir);
    fs::create_directory(dir);
    {
        std::ofstream model(dir / "model.onnx", std::ios::binary | std::ios::trunc);
        model << std::string("\x00\x01\x02 definitely not protobuf", 27);
    }
    {
        std::ofstream vocab(dir / "vocab.txt", std::ios::binary | std::ios::trunc);
        vocab << "[PAD]\n[UNK]\n[CLS]\n[SEP]\nhello\nworld\n";
    }

    std::remove("builtin_embedder_onnx_corrupt.sqlite3");
    retrieval_engine::RetrievalEngine engine("builtin_embedder_onnx_corrupt.sqlite3", kEmbeddingDim);

    EXPECT_THROW(engine.load_embedding_model((dir / "model.onnx").string()), std::runtime_error);
    EXPECT_FALSE(engine.has_embedding_model());

    fs::remove_all(dir);
}

// A failed re-load must not detach the model that was already working.
TEST(BuiltinEmbedder, LoadEmbeddingModelKeepsThePreviousModelWhenTheNewLoadFails) {
    retrieval_engine::RetrievalEngine engine =
        MakeEngineWithMockModel("builtin_embedder_rollback.sqlite3", "builtin_embedder_mock_model.txt");
    ASSERT_TRUE(engine.has_embedding_model());

    EXPECT_THROW(engine.load_embedding_model("no_such_model_file_98765.bin"), std::runtime_error);

    EXPECT_TRUE(engine.has_embedding_model());
    EXPECT_EQ(engine.embed("the model is still attached").size(), kEmbeddingDim);
}
