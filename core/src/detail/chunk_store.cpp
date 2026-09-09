#include "chunk_store.hpp"

#include "recency_decay.hpp"
#include "rrf_fusion.hpp"
#include "time_util.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace retrieval_engine::detail {

ChunkStore::ChunkStore(sqlite3* db, std::size_t dim) : dimensions_(dim), repository_(db, dim), dense_index_(dim) {
    // Rebuild the dense sidecar from whatever ChunkRepository already has
    // persisted -- the "rebuild the index from SQLite" recovery path. Run
    // unconditionally (not just after corruption), since DenseIndex does
    // not persist itself, only SQLite does. A no-op on a fresh/empty
    // database.
    repository_.ForEachChunk(
        [this](std::uint64_t chunk_id, const std::vector<float>& embedding) { dense_index_.Add(chunk_id, embedding); });
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
    // sidecar" architecture does not support recovering from. A DenseIndex
    // failure here can only ever leave it
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

std::vector<FusionEntry> ChunkStore::Fuse(const std::string& query_text, const std::vector<float>& query_vec,
                                          std::size_t k) const {
    return RrfFuse(search_dense(query_vec, k), search_sparse(query_text, k), k);
}

SearchExplanation ChunkStore::ExplanationFromFusionEntry(const FusionEntry& entry, std::size_t rank) {
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
    explanation.final_rank = rank;
    // created_at_unix_seconds/age_seconds/recency_factor/decayed_score are
    // left at SearchExplanation's own defaults (see retrieval_engine.hpp) --
    // meaningful only for search_memory_explained(), which fills them in
    // itself after calling this.
    return explanation;
}

std::vector<ChunkSearchResult> ChunkStore::search_hybrid(const std::string& query_text,
                                                         const std::vector<float>& query_vec, std::size_t k) const {
    const std::vector<FusionEntry> fused = Fuse(query_text, query_vec, k);

    std::vector<ChunkSearchResult> results;
    results.reserve(fused.size());
    for (const FusionEntry& entry : fused) {
        results.push_back(ChunkSearchResult{entry.document_id, entry.chunk_index, entry.text, entry.fused_score});
    }
    return results;
}

std::vector<SearchExplanation> ChunkStore::search_explained(const std::string& query_text,
                                                            const std::vector<float>& query_vec, std::size_t k) const {
    const std::vector<FusionEntry> fused = Fuse(query_text, query_vec, k);

    std::vector<SearchExplanation> explanations;
    explanations.reserve(fused.size());
    for (std::size_t i = 0; i < fused.size(); ++i) {
        explanations.push_back(ExplanationFromFusionEntry(fused[i], /*rank=*/i + 1));  // 1-based
    }
    return explanations;
}

std::vector<ChunkStore::DecayedEntry> ChunkStore::FuseRankAndDecay(const std::string& query_text,
                                                                   const std::vector<float>& query_vec, std::size_t k,
                                                                   float decay_lambda) const {
    std::vector<FusionEntry> fused = Fuse(query_text, query_vec, k);
    const std::int64_t now = CurrentUnixTimeSeconds();

    std::vector<DecayedEntry> decayed_entries;
    decayed_entries.reserve(fused.size());
    for (FusionEntry& entry : fused) {
        const std::int64_t created_at = repository_.GetCreatedAt(entry.document_id, entry.chunk_index);
        const double age_seconds = static_cast<double>(now - created_at);
        const double recency_factor = ComputeRecencyFactor(age_seconds, static_cast<double>(decay_lambda));
        const float decayed_score = entry.fused_score * static_cast<float>(recency_factor);
        decayed_entries.push_back(
            DecayedEntry{std::move(entry), created_at, age_seconds, recency_factor, decayed_score});
    }

    // Re-rank by decayed_score: this is the "temporal reranker" -- decay
    // isn't just a cosmetic score adjustment, it can (and is meant to)
    // change which chunks land in the final top-k and in what order. Same
    // deterministic tie-break as RrfFuse (by document_id, then
    // chunk_index), for the same explainability reason.
    std::sort(decayed_entries.begin(), decayed_entries.end(), [](const DecayedEntry& a, const DecayedEntry& b) {
        if (a.decayed_score != b.decayed_score) return a.decayed_score > b.decayed_score;
        if (a.entry.document_id != b.entry.document_id) return a.entry.document_id < b.entry.document_id;
        return a.entry.chunk_index < b.entry.chunk_index;
    });

    return decayed_entries;
}

std::vector<ChunkSearchResult> ChunkStore::search_memory(const std::string& query_text,
                                                         const std::vector<float>& query_vec, std::size_t k,
                                                         float decay_lambda) const {
    const std::vector<DecayedEntry> decayed = FuseRankAndDecay(query_text, query_vec, k, decay_lambda);

    std::vector<ChunkSearchResult> results;
    results.reserve(decayed.size());
    for (const DecayedEntry& decayed_entry : decayed) {
        results.push_back(ChunkSearchResult{decayed_entry.entry.document_id, decayed_entry.entry.chunk_index,
                                            decayed_entry.entry.text, decayed_entry.decayed_score});
    }
    return results;
}

std::vector<SearchExplanation> ChunkStore::search_memory_explained(const std::string& query_text,
                                                                   const std::vector<float>& query_vec, std::size_t k,
                                                                   float decay_lambda) const {
    const std::vector<DecayedEntry> decayed = FuseRankAndDecay(query_text, query_vec, k, decay_lambda);

    std::vector<SearchExplanation> explanations;
    explanations.reserve(decayed.size());
    for (std::size_t i = 0; i < decayed.size(); ++i) {
        const DecayedEntry& decayed_entry = decayed[i];
        // rank is 1-based, over the *decayed* order (not the raw fused order).
        SearchExplanation explanation = ExplanationFromFusionEntry(decayed_entry.entry, /*rank=*/i + 1);
        explanation.created_at_unix_seconds = decayed_entry.created_at_unix_seconds;
        explanation.age_seconds = decayed_entry.age_seconds;
        explanation.recency_factor = decayed_entry.recency_factor;
        explanation.decayed_score = decayed_entry.decayed_score;
        explanations.push_back(std::move(explanation));
    }
    return explanations;
}

}  // namespace retrieval_engine::detail
