// USearch sidecar persistence: a reopened engine loads "<db_path>.usearch"
// instead of rebuilding the HNSW graph from SQLite, SQLite stays
// authoritative, and a missing / stale / unreadable sidecar falls back to
// a rebuild that produces identical results.
#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "retrieval_engine/retrieval_engine.hpp"

namespace {

constexpr std::size_t kDim = 4;

std::string SidecarPath(const std::string& db_path) { return db_path + ".usearch"; }

void RemoveArtifacts(const std::string& db_path) {
    std::error_code ec;
    std::filesystem::remove(db_path, ec);
    std::filesystem::remove(SidecarPath(db_path), ec);
    std::filesystem::remove(SidecarPath(db_path) + ".tmp", ec);
}

// N single-chunk documents with distinct one-hot-ish embeddings, so every
// query has an unambiguous nearest neighbour.
std::vector<retrieval_engine::DocumentInput> MakeDocuments(std::size_t count) {
    std::vector<retrieval_engine::DocumentInput> documents;
    for (std::size_t i = 0; i < count; ++i) {
        retrieval_engine::DocumentChunkInput chunk;
        chunk.text = "document number " + std::to_string(i);
        chunk.embedding = std::vector<float>(kDim, 0.1f);
        chunk.embedding[i % kDim] = 1.0f + static_cast<float>(i);
        chunk.start_token = 0;
        chunk.end_token = 3;

        retrieval_engine::DocumentInput document;
        document.document_id = "doc-" + std::to_string(i);
        document.chunks.push_back(std::move(chunk));
        documents.push_back(std::move(document));
    }
    return documents;
}

std::vector<std::string> DenseHitIds(const retrieval_engine::RetrievalEngine& engine, const std::vector<float>& query) {
    std::vector<std::string> ids;
    for (const auto& hit : engine.search_dense(query, 5)) ids.push_back(hit.document_id);
    return ids;
}

std::vector<std::string> HybridHitIds(retrieval_engine::RetrievalEngine& engine, const std::string& text,
                                      const std::vector<float>& query) {
    std::vector<std::string> ids;
    for (const auto& hit : engine.search_hybrid(text, query, 5)) ids.push_back(hit.document_id);
    return ids;
}

}  // namespace

TEST(SidecarPersistence, FirstOpenWritesTheSidecarFile) {
    const std::string db_path = "persistence_test_first_open.sqlite3";
    RemoveArtifacts(db_path);

    {
        retrieval_engine::RetrievalEngine engine(db_path, kDim);
        engine.add_documents(MakeDocuments(6));
    }

    EXPECT_TRUE(std::filesystem::exists(SidecarPath(db_path)));
    RemoveArtifacts(db_path);
}

TEST(SidecarPersistence, ReopeningLoadsSidecarWithoutRebuilding) {
    const std::string db_path = "persistence_test_reopen.sqlite3";
    RemoveArtifacts(db_path);

    {
        retrieval_engine::RetrievalEngine engine(db_path, kDim);
        engine.add_documents(MakeDocuments(10));
        EXPECT_FALSE(engine.loaded_index_from_sidecar());  // fresh build, no sidecar yet
    }

    retrieval_engine::RetrievalEngine reopened(db_path, kDim);
    EXPECT_TRUE(reopened.loaded_index_from_sidecar());
    EXPECT_EQ(reopened.chunk_count(), 10u);

    RemoveArtifacts(db_path);
}

TEST(SidecarPersistence, IngestingAfterASidecarLoadExtendsTheIndex) {
    const std::string db_path = "persistence_test_incremental.sqlite3";
    RemoveArtifacts(db_path);

    {
        retrieval_engine::RetrievalEngine engine(db_path, kDim);
        engine.add_documents(MakeDocuments(5));
    }

    retrieval_engine::RetrievalEngine reopened(db_path, kDim);
    ASSERT_TRUE(reopened.loaded_index_from_sidecar());

    auto more = MakeDocuments(9);
    more.erase(more.begin(), more.begin() + 5);  // docs 5..8
    reopened.add_documents(more);
    EXPECT_EQ(reopened.chunk_count(), 9u);
    EXPECT_EQ(reopened.search_dense({0.1f, 0.1f, 0.1f, 1.0f}, 20).size(), 9u);

    // The refreshed sidecar carries all 9 on the next open.
    retrieval_engine::RetrievalEngine reopened_again(db_path, kDim);
    EXPECT_TRUE(reopened_again.loaded_index_from_sidecar());
    EXPECT_EQ(reopened_again.search_dense({0.1f, 0.1f, 0.1f, 1.0f}, 20).size(), 9u);

    RemoveArtifacts(db_path);
}

TEST(SidecarPersistence, LoadedIndexAndRebuiltIndexReturnIdenticalResults) {
    const std::string db_path = "persistence_test_equivalence.sqlite3";
    RemoveArtifacts(db_path);

    const auto documents = MakeDocuments(12);
    {
        retrieval_engine::RetrievalEngine engine(db_path, kDim);
        engine.add_documents(documents);
    }

    retrieval_engine::RetrievalEngine loaded(db_path, kDim);
    ASSERT_TRUE(loaded.loaded_index_from_sidecar());

    std::filesystem::remove(SidecarPath(db_path));
    retrieval_engine::RetrievalEngine rebuilt(db_path, kDim);
    ASSERT_FALSE(rebuilt.loaded_index_from_sidecar());

    const std::vector<std::vector<float>> queries = {
        {1.0f, 0.1f, 0.1f, 0.1f}, {0.1f, 1.0f, 0.1f, 0.1f}, {0.1f, 0.1f, 1.0f, 0.1f}, {0.5f, 0.5f, 0.1f, 0.1f}};

    for (const auto& query : queries) {
        EXPECT_EQ(DenseHitIds(loaded, query), DenseHitIds(rebuilt, query));
        EXPECT_EQ(HybridHitIds(loaded, "document number", query), HybridHitIds(rebuilt, "document number", query));
    }

    RemoveArtifacts(db_path);
}

TEST(SidecarPersistence, UnreadableSidecarFallsBackToRebuild) {
    const std::string db_path = "persistence_test_corrupt.sqlite3";
    RemoveArtifacts(db_path);

    {
        retrieval_engine::RetrievalEngine engine(db_path, kDim);
        engine.add_documents(MakeDocuments(8));
    }

    // A too-short sidecar and a header-sized blob of garbage must both be
    // rejected, and the engine must silently rebuild from SQLite rather
    // than throw.
    for (const std::string& garbage : {std::string("nope"), std::string(256, '\xa5')}) {
        {
            std::ofstream corrupt(SidecarPath(db_path), std::ios::binary | std::ios::trunc);
            corrupt << garbage;
        }
        retrieval_engine::RetrievalEngine reopened(db_path, kDim);
        EXPECT_FALSE(reopened.loaded_index_from_sidecar());
        EXPECT_EQ(reopened.chunk_count(), 8u);
        EXPECT_EQ(reopened.search_dense({1.0f, 0.1f, 0.1f, 0.1f}, 1).size(), 1u);
    }

    RemoveArtifacts(db_path);
}

TEST(SidecarPersistence, SidecarOutOfSyncWithChunkTableIsDiscarded) {
    const std::string db_path = "persistence_test_stale.sqlite3";
    RemoveArtifacts(db_path);

    retrieval_engine::RetrievalEngine engine(db_path, kDim);
    engine.add_documents(MakeDocuments(4));

    // Snapshot the 4-vector sidecar, add more, then restore the snapshot so
    // the sidecar lags SQLite (the shape of a crash between the SQLite
    // commit and the sidecar write).
    const std::string snapshot = SidecarPath(db_path) + ".snapshot";
    std::filesystem::copy_file(SidecarPath(db_path), snapshot, std::filesystem::copy_options::overwrite_existing);

    auto more = MakeDocuments(7);
    more.erase(more.begin(), more.begin() + 4);  // docs 4,5,6
    engine.add_documents(more);
    ASSERT_EQ(engine.chunk_count(), 7u);

    std::filesystem::copy_file(snapshot, SidecarPath(db_path), std::filesystem::copy_options::overwrite_existing);

    retrieval_engine::RetrievalEngine reopened(db_path, kDim);
    EXPECT_FALSE(reopened.loaded_index_from_sidecar());  // count mismatch -> discard + rebuild
    EXPECT_EQ(reopened.chunk_count(), 7u);
    // All 7 vectors are searchable: a rebuild picked up the 3 docs added
    // after the snapshot. A trusted 4-vector sidecar would return only 4.
    EXPECT_EQ(reopened.search_dense({0.1f, 0.1f, 1.0f, 0.1f}, 20).size(), 7u);

    std::filesystem::remove(snapshot);
    RemoveArtifacts(db_path);
}
