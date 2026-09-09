#pragma once

#include "retrieval_engine/retrieval_engine.hpp"  // ChunkSearchResult

#include <cstddef>
#include <string>
#include <vector>

// Pure Reciprocal Rank Fusion math: no SQLite or usearch dependency --
// operates only on already-computed, already-ordered result lists -- so
// it's independently unit-testable with hand-fed data.
// Internal implementation detail, not part of the public API.
namespace retrieval_engine::detail {

// One chunk's fusion state: which ranking(s) it appeared in, its rank
// (1-based) in each, and the resulting fused score. Returned by RrfFuse()
// for search_hybrid()/search_explained() to format differently.
struct FusionEntry {
    std::string document_id;
    std::size_t chunk_index = 0;
    std::string text;

    bool dense_present = false;
    float dense_distance = 0.0f;
    std::size_t dense_rank = 0;  // 1-based; 0 means "not present"

    bool sparse_present = false;
    float sparse_bm25_score = 0.0f;
    std::size_t sparse_rank = 0;  // 1-based; 0 means "not present"

    float fused_score = 0.0f;
};

// Combines `dense_results` and `sparse_results` -- each already ordered
// best-first, as search_dense()/search_sparse() return them -- via
// Reciprocal Rank Fusion (k=60): RRF_Score(d) = sum, over whichever
// ranking(s) chunk d appears in, of 1/(60+rank-in-that-ranking) (0 for a
// ranking it's absent from). A chunk present in both merges into one entry,
// matched by (document_id, chunk_index) -- unique within the chunks schema.
// Ties on fused_score are broken by (document_id, chunk_index) for
// deterministic, reproducible output. Returns the top `k` entries ordered
// highest-fused-score-first.
std::vector<FusionEntry> RrfFuse(const std::vector<ChunkSearchResult>& dense_results,
                                  const std::vector<ChunkSearchResult>& sparse_results, std::size_t k);

}  // namespace retrieval_engine::detail
