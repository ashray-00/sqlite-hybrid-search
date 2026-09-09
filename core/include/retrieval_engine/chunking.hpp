#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace retrieval_engine {

// A single chunk produced by chunk_text(): its text, and its position (as a
// half-open token range [start_token, end_token)) within the source
// document's token sequence, so callers can trace a chunk back to where it
// came from.
struct Chunk {
    std::string text;
    std::size_t start_token;
    std::size_t end_token;
};

// Splits `text` into overlapping token windows (BUILD_PLAN.md Stage 1:
// "token-window with overlap to start"). A "token" is a maximal run of
// non-whitespace characters -- Stage 1 has no real tokenizer yet, callers
// supply embeddings externally regardless of how text is tokenized.
//
// `window_tokens` is the number of tokens per chunk; `overlap_tokens` is how
// many trailing tokens of one chunk are repeated at the start of the next.
// Consecutive chunks advance by `window_tokens - overlap_tokens` tokens, and
// the final chunk is truncated to whatever tokens remain rather than padded.
//
// Returns an empty vector for empty (or all-whitespace) `text`.
// Throws std::invalid_argument if `window_tokens == 0` or
// `overlap_tokens >= window_tokens` (a non-advancing or backwards window
// would never terminate).
std::vector<Chunk> chunk_text(const std::string& text, std::size_t window_tokens, std::size_t overlap_tokens);

}  // namespace retrieval_engine
