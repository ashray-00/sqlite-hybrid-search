#include "chunk_store.hpp"

#include "rrf_fusion.hpp"

#include <stdexcept>
#include <utility>

namespace retrieval_engine::detail {

ChunkStore::ChunkStore(sqlite3* db, std::size_t dim) : dimensions_(dim), repository_(db, dim), dense_index_(dim) {
    // Rebuild the dense sidecar from whatever ChunkRepository already has
    // persisted -- the "rebuild the index from SQLite" recovery path from
    // BUILD_PLAN.md section 5. Run unconditionally (not just after
    // corruption), since DenseIndex doesn't persist itself, only SQLite
    // does. A no-op on a fresh/empty database.
    repository_.ForEachChunk([this](std::uint64_t chunk_id, const std::vector<float>& embedding) {
        dense_index_.Add(chunk_id, embedding);
    });
}

void ChunkStore::add_documents(const std::vector<DocumentInput>& documents) {
    // Validate everything before writing anything, so a bad chunk deep in
    // the batch doesn't leave earlier documents in this same call partially
    // ingested (ChunkRepository's transaction protects against a mid-write
    // SQLite failure; this protects against a caller mistake up front).
    for (const auto& document : documents) {
        for (const auto& chunk : document.chunks) {
            if (chunk.embedding.size() != dimensions_) {
                throw std::invalid_argument(
                    "ChunkStore::add_documents: chunk embedding size does not match index dimensionality");
            }
        }
    }

    // ChunkRepository::add_documents() durably commits documents/chunks/
    // chunks_fts as one SQLite transaction (or throws having committed
    // nothing) before this returns. Only once that has succeeded do we
    // touch DenseIndex: it has no transaction concept of its own, so
    // populating it *before* ChunkRepository's commit could leave a vector
    // live in the index with no backing SQLite row if a later document in
    // the batch failed and rolled everything else back -- the exact
    // inconsistency the "SQLite is authoritative, usearch is a rebuildable
    // sidecar" architecture (BUILD_PLAN.md section 5) doesn't support
    // recovering from. A DenseIndex failure here can only ever leave it
    // lagging what's already durably in SQLite -- recoverable by rebuilding
    // it, the opposite (and architecturally sanctioned) direction of drift.
    const auto pending = repository_.add_documents(documents);
    for (const auto& [chunk_id, embedding] : pending) dense_index_.Add(chunk_id, *embedding);
}

std::size_t ChunkStore::chunk_count() const { return repository_.chunk_count(); }

std::vector<ChunkSearchResult> ChunkStore::search_dense(const std::vector<float>& query, std::size_t k) const {
    const auto hits = dense_index_.Search(query, k);

    std::vector<ChunkSearchResult> results;
    results.reserve(hits.size());
    for (const auto& [chunk_id, distance] : hits) {
        ChunkSearchResult result = repository_.Resolve(chunk_id);
        result.score = distance;
        results.push_back(std::move(result));
    }
    return results;
}

std::vector<ChunkSearchResult> ChunkStore::search_sparse(const std::string& query_text, std::size_t k) const {
    return repository_.SearchSparse(query_text, k);
}

std::vector<ChunkSearchResult> ChunkStore::search_hybrid(const std::string& query_text,
                                                          const std::vector<float>& query_vec, std::size_t k) const {
    const std::vector<FusionEntry> fused = RrfFuse(search_dense(query_vec, k), search_sparse(query_text, k), k);

    std::vector<ChunkSearchResult> results;
    results.reserve(fused.size());
    for (const FusionEntry& entry : fused) {
        results.push_back(ChunkSearchResult{entry.document_id, entry.chunk_index, entry.text, entry.fused_score});
    }
    return results;
}

std::vector<SearchExplanation> ChunkStore::search_explained(const std::string& query_text,
                                                             const std::vector<float>& query_vec,
                                                             std::size_t k) const {
    const std::vector<FusionEntry> fused = RrfFuse(search_dense(query_vec, k), search_sparse(query_text, k), k);

    std::vector<SearchExplanation> explanations;
    explanations.reserve(fused.size());
    for (std::size_t i = 0; i < fused.size(); ++i) {
        const FusionEntry& entry = fused[i];
        SearchExplanation explanation;
        explanation.document_id = entry.document_id;
        explanation.chunk_index = entry.chunk_index;
        explanation.text = entry.text;
        explanation.dense_present = entry.dense_present;
        explanation.dense_distance = entry.dense_distance;
        explanation.dense_rank = entry.dense_rank;
        explanation.sparse_present = entry.sparse_present;
        explanation.sparse_bm25_score = entry.sparse_bm25_score;
        explanation.sparse_rank = entry.sparse_rank;
        explanation.fused_score = entry.fused_score;
        explanation.final_rank = i + 1;  // 1-based
        explanations.push_back(std::move(explanation));
    }
    return explanations;
}

}  // namespace retrieval_engine::detail
