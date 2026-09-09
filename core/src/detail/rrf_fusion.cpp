#include "rrf_fusion.hpp"

#include <algorithm>
#include <map>
#include <utility>

namespace retrieval_engine::detail {

namespace {
constexpr double kRrfK = 60.0;
}  // namespace

std::vector<FusionEntry> RrfFuse(const std::vector<ChunkSearchResult>& dense_results,
                                 const std::vector<ChunkSearchResult>& sparse_results, std::size_t k) {
    // Keyed by (document_id, chunk_index) -- unique within the chunks
    // schema (chunk_index is 0-based position within its document) -- so a
    // chunk hit by both rankings merges into one entry instead of two.
    std::map<std::pair<std::string, std::size_t>, FusionEntry> entries;

    for (std::size_t i = 0; i < dense_results.size(); ++i) {
        const ChunkSearchResult& r = dense_results[i];
        FusionEntry& entry = entries[{r.document_id, r.chunk_index}];
        entry.document_id = r.document_id;
        entry.chunk_index = r.chunk_index;
        entry.text = r.text;
        entry.dense_present = true;
        entry.dense_distance = r.score;
        entry.dense_rank = i + 1;  // 1-based
    }

    for (std::size_t i = 0; i < sparse_results.size(); ++i) {
        const ChunkSearchResult& r = sparse_results[i];
        FusionEntry& entry = entries[{r.document_id, r.chunk_index}];
        entry.document_id = r.document_id;
        entry.chunk_index = r.chunk_index;
        entry.text = r.text;
        entry.sparse_present = true;
        entry.sparse_bm25_score = r.score;
        entry.sparse_rank = i + 1;  // 1-based
    }

    std::vector<FusionEntry> sorted_entries;
    sorted_entries.reserve(entries.size());
    for (auto& [key, entry] : entries) {
        entry.fused_score = 0.0f;
        if (entry.dense_present)
            entry.fused_score += static_cast<float>(1.0 / (kRrfK + static_cast<double>(entry.dense_rank)));
        if (entry.sparse_present)
            entry.fused_score += static_cast<float>(1.0 / (kRrfK + static_cast<double>(entry.sparse_rank)));
        sorted_entries.push_back(std::move(entry));
    }

    // Break ties on fused_score deterministically (by document_id then
    // chunk_index) rather than leaving them to std::sort's unspecified
    // handling of equal elements -- an explainability feature should give
    // the same, reproducible ordering for the same input every time.
    std::sort(sorted_entries.begin(), sorted_entries.end(), [](const FusionEntry& a, const FusionEntry& b) {
        if (a.fused_score != b.fused_score) return a.fused_score > b.fused_score;
        if (a.document_id != b.document_id) return a.document_id < b.document_id;
        return a.chunk_index < b.chunk_index;
    });
    if (sorted_entries.size() > k) sorted_entries.resize(k);

    return sorted_entries;
}

}  // namespace retrieval_engine::detail
