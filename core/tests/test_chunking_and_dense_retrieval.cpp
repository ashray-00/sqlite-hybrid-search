// Document ingestion and dense (vector) retrieval.
//
//  1. Chunking.* -- retrieval_engine::chunk_text() (token-window chunker),
//     core/include/retrieval_engine/chunking.hpp / core/src/chunking.cpp.
//  2. RetrievalEngineDenseSearch.* -- retrieval_engine::RetrievalEngine's
//     add_documents()/search_dense()/chunk_count() (core/src/detail/
//     chunk_store.*), including its constructor validation.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

#include "retrieval_engine/chunking.hpp"
#include "retrieval_engine/retrieval_engine.hpp"

namespace {

// Builds a synthetic document body of exactly `token_count` whitespace-
// separated tokens, so tests can reason exactly about how many chunks
// chunk_text() should produce.
std::string MakeTokenizedText(const std::string& prefix, std::size_t token_count) {
    std::string text;
    for (std::size_t i = 0; i < token_count; ++i) {
        if (i != 0) text += ' ';
        text += prefix + "_w" + std::to_string(i);
    }
    return text;
}

}  // namespace

TEST(Chunking, SplitsTextIntoOverlappingTokenWindows) {
    // 9 tokens, window=5, overlap=2 (step=3): windows [0,5) [3,8) [6,9).
    const std::string text = MakeTokenizedText("doc", 9);

    const std::vector<retrieval_engine::Chunk> chunks = retrieval_engine::chunk_text(text, /*window_tokens=*/5,
                                                                                     /*overlap_tokens=*/2);

    ASSERT_EQ(chunks.size(), 3u);

    EXPECT_EQ(chunks[0].start_token, 0u);
    EXPECT_EQ(chunks[0].end_token, 5u);
    EXPECT_EQ(chunks[0].text, "doc_w0 doc_w1 doc_w2 doc_w3 doc_w4");

    EXPECT_EQ(chunks[1].start_token, 3u);
    EXPECT_EQ(chunks[1].end_token, 8u);
    EXPECT_EQ(chunks[1].text, "doc_w3 doc_w4 doc_w5 doc_w6 doc_w7");

    EXPECT_EQ(chunks[2].start_token, 6u);
    EXPECT_EQ(chunks[2].end_token, 9u);
    EXPECT_EQ(chunks[2].text, "doc_w6 doc_w7 doc_w8");
}

TEST(Chunking, EmptyTextProducesNoChunks) {
    EXPECT_TRUE(retrieval_engine::chunk_text("", /*window_tokens=*/5, /*overlap_tokens=*/2).empty());
    EXPECT_TRUE(retrieval_engine::chunk_text("   ", /*window_tokens=*/5, /*overlap_tokens=*/2).empty());
}

TEST(Chunking, RejectsNonAdvancingWindow) {
    EXPECT_THROW(retrieval_engine::chunk_text("a b c", /*window_tokens=*/5, /*overlap_tokens=*/5),
                 std::invalid_argument);
    EXPECT_THROW(retrieval_engine::chunk_text("a b c", /*window_tokens=*/0, /*overlap_tokens=*/0),
                 std::invalid_argument);
}

namespace {

constexpr std::size_t kDim = 4;
constexpr std::size_t kNumDocuments = 10;
constexpr std::size_t kChunksPerDocument = 3;  // 9-token docs, window=5, overlap=2 (see Chunking tests above)

// Builds the 10-document synthetic corpus this file's RetrievalEngineDenseSearch
// tests share: document d's chunks all sit on the same 2D direction
// (angle d * pi/10, spread across documents so cosine similarity strictly
// decreases with angular distance), with a tiny per-chunk perturbation in
// an otherwise-unused dimension so a chunk's own index is the only thing
// that determines its similarity rank *within* its document. This makes
// both cross-document and within-document nearest-neighbour order exactly
// predictable by hand -- see the derivation in RetrievalEngineDenseSearch
// comments below.
std::vector<retrieval_engine::DocumentInput> MakeSyntheticCorpus() {
    std::vector<retrieval_engine::DocumentInput> documents;
    documents.reserve(kNumDocuments);

    for (std::size_t d = 0; d < kNumDocuments; ++d) {
        const std::string document_id = "doc-" + std::to_string(d);
        const float theta = static_cast<float>(d) * static_cast<float>(M_PI) / 10.0f;
        const float base_x = std::cos(theta);
        const float base_y = std::sin(theta);

        const std::string text = MakeTokenizedText(document_id, 9);  // -> 3 chunks, see Chunking tests
        const std::vector<retrieval_engine::Chunk> chunks = retrieval_engine::chunk_text(text, 5, 2);

        retrieval_engine::DocumentInput document;
        document.document_id = document_id;
        document.metadata = "source:synthetic-corpus";
        for (std::size_t c = 0; c < chunks.size(); ++c) {
            retrieval_engine::DocumentChunkInput chunk_input;
            chunk_input.text = chunks[c].text;
            chunk_input.start_token = chunks[c].start_token;
            chunk_input.end_token = chunks[c].end_token;
            // cos(query, chunk) = 1 / sqrt(1 + (0.001*c)^2) when the query is
            // exactly {base_x, base_y, 0, 0} -- strictly decreasing in c.
            chunk_input.embedding = {base_x, base_y, 0.001f * static_cast<float>(c), 0.0f};
            document.chunks.push_back(std::move(chunk_input));
        }
        documents.push_back(std::move(document));
    }

    return documents;
}

std::vector<float> QueryVectorForDocument(std::size_t d) {
    const float theta = static_cast<float>(d) * static_cast<float>(M_PI) / 10.0f;
    return {std::cos(theta), std::sin(theta), 0.0f, 0.0f};
}

}  // namespace

TEST(RetrievalEngineDenseSearch, ConstructorRejectsZeroDimension) {
    const std::string db_path = "dense_retrieval_test_zero_dim.sqlite3";
    std::remove(db_path.c_str());

    EXPECT_THROW(retrieval_engine::RetrievalEngine(db_path, /*dim=*/0), std::invalid_argument);
}

TEST(RetrievalEngineDenseSearch, SearchDenseReturnsExpectedDocumentAndPersistsRowCount) {
    const std::string db_path = "dense_retrieval_test_search.sqlite3";
    std::remove(db_path.c_str());

    const std::vector<retrieval_engine::DocumentInput> documents = MakeSyntheticCorpus();

    retrieval_engine::RetrievalEngine engine(db_path, kDim);
    engine.add_documents(documents);

    ASSERT_EQ(engine.chunk_count(), kNumDocuments * kChunksPerDocument);

    // Querying with document 7's exact base direction must return that
    // document's least-perturbed (chunk_index 0) chunk as the single nearest
    // neighbour: every other document's base direction is at least pi/10
    // radians away (a real angular separation), which dwarfs the 0.001-scale
    // intra-document perturbation.
    const std::vector<retrieval_engine::ChunkSearchResult> top1 = engine.search_dense(QueryVectorForDocument(7), 1);
    ASSERT_EQ(top1.size(), 1u);
    EXPECT_EQ(top1[0].document_id, "doc-7");
    EXPECT_EQ(top1[0].chunk_index, 0u);

    // Requesting all 3 chunks of that same query must return exactly
    // document 7's three chunks, ordered by increasing perturbation
    // (chunk_index 0, 1, 2 -- see MakeSyntheticCorpus for why that ordering
    // is exact, not approximate).
    const std::vector<retrieval_engine::ChunkSearchResult> top3 = engine.search_dense(QueryVectorForDocument(7), 3);
    ASSERT_EQ(top3.size(), 3u);
    for (std::size_t i = 0; i < 3; ++i) {
        EXPECT_EQ(top3[i].document_id, "doc-7");
        EXPECT_EQ(top3[i].chunk_index, i);
    }
}

TEST(RetrievalEngineDenseSearch, DataPersistsAfterReopeningDatabase) {
    const std::string db_path = "dense_retrieval_test_persistence.sqlite3";
    std::remove(db_path.c_str());

    const std::vector<retrieval_engine::DocumentInput> documents = MakeSyntheticCorpus();

    {
        retrieval_engine::RetrievalEngine writer(db_path, kDim);
        writer.add_documents(documents);
        ASSERT_EQ(writer.chunk_count(), kNumDocuments * kChunksPerDocument);
    }  // writer destructed -- nothing but the SQLite file on disk survives

    // A fresh instance over the same db_path, with add_documents() never
    // called on it, must (a) see the same chunk row count by reading SQLite,
    // and (b) answer searches correctly by rebuilding the usearch index from
    // those persisted rows -- not by starting from an empty in-memory index.
    retrieval_engine::RetrievalEngine reopened(db_path, kDim);
    EXPECT_EQ(reopened.chunk_count(), kNumDocuments * kChunksPerDocument);

    const std::vector<retrieval_engine::ChunkSearchResult> top1 = reopened.search_dense(QueryVectorForDocument(2), 1);
    ASSERT_EQ(top1.size(), 1u);
    EXPECT_EQ(top1[0].document_id, "doc-2");
    EXPECT_EQ(top1[0].chunk_index, 0u);
}

TEST(RetrievalEngineDenseSearch, AddDocumentsRejectsMismatchedEmbeddingDimension) {
    const std::string db_path = "dense_retrieval_test_dim_mismatch.sqlite3";
    std::remove(db_path.c_str());

    retrieval_engine::RetrievalEngine engine(db_path, kDim);

    retrieval_engine::DocumentInput document;
    document.document_id = "doc-x";
    document.metadata = "m";
    document.chunks.push_back(
        retrieval_engine::DocumentChunkInput{"hello", {1.0f, 0.0f}, /*start_token=*/0, /*end_token=*/1});  // wrong size

    EXPECT_THROW(engine.add_documents({document}), std::invalid_argument);
    EXPECT_EQ(engine.chunk_count(), 0u);
}

// Regression test for a bug caught in Phase 3's independent review: usearch
// has no transaction concept, so a naive implementation that called
// chunk_index.add() *inside* the same loop as the SQLite writes could leave
// vectors permanently live in the index after a later document in the same
// batch failed and rolled SQLite back -- a usearch entry with no backing
// row, which the "SQLite authoritative, usearch rebuildable" architecture
// cannot recover from. add_documents() must defer
// every index mutation until after SQLite's transaction commits.
TEST(RetrievalEngineDenseSearch, FailedBatchLeavesNoPhantomEntriesInChunkIndex) {
    const std::string db_path = "dense_retrieval_test_rollback.sqlite3";
    std::remove(db_path.c_str());

    retrieval_engine::RetrievalEngine engine(db_path, kDim);

    retrieval_engine::DocumentInput document;
    document.document_id = "dup";
    document.metadata = "m";
    document.chunks.push_back(retrieval_engine::DocumentChunkInput{"hello world",
                                                                   {1.0f, 0.0f, 0.0f, 0.0f},
                                                                   /*start_token=*/0,
                                                                   /*end_token=*/2});

    // Two documents sharing the same document_id in one batch: the second
    // INSERT INTO documents violates the PRIMARY KEY constraint, so the
    // whole batch -- including the first document's already-inserted chunk
    // -- must roll back.
    const std::vector<retrieval_engine::DocumentInput> conflicting_batch = {document, document};
    EXPECT_THROW(engine.add_documents(conflicting_batch), std::runtime_error);
    EXPECT_EQ(engine.chunk_count(), 0u);

    // If the bug were present, the rolled-back chunk would still be live in
    // the usearch index with no backing SQLite row. A clean rollback means
    // the index is empty too, so this returns nothing (not a phantom hit,
    // and not a thrown "index and store have desynced" error).
    EXPECT_TRUE(engine.search_dense(document.chunks[0].embedding, 1).empty());

    // The engine must still work normally afterwards.
    engine.add_documents({document});
    ASSERT_EQ(engine.chunk_count(), 1u);
    const std::vector<retrieval_engine::ChunkSearchResult> results =
        engine.search_dense(document.chunks[0].embedding, 1);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].document_id, "dup");
}
