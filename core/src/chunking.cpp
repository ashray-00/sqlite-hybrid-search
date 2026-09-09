#include "retrieval_engine/chunking.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace retrieval_engine {

std::vector<Chunk> chunk_text(const std::string& text, std::size_t window_tokens, std::size_t overlap_tokens) {
    if (window_tokens == 0) throw std::invalid_argument("chunk_text: window_tokens must be greater than zero");
    if (overlap_tokens >= window_tokens)
        throw std::invalid_argument("chunk_text: overlap_tokens must be less than window_tokens");

    // Tokenize on whitespace: `operator>>` on an istringstream splits on any
    // run of whitespace and skips leading/trailing whitespace, which is
    // exactly "a token is a maximal run of non-whitespace characters".
    std::vector<std::string> tokens;
    std::istringstream token_stream(text);
    std::string token;
    while (token_stream >> token) tokens.push_back(token);

    std::vector<Chunk> chunks;
    if (tokens.empty()) return chunks;

    const std::size_t step = window_tokens - overlap_tokens;
    std::size_t start = 0;
    while (start < tokens.size()) {
        const std::size_t end = std::min(start + window_tokens, tokens.size());

        std::string chunk_text_value;
        for (std::size_t i = start; i < end; ++i) {
            if (i != start) chunk_text_value += ' ';
            chunk_text_value += tokens[i];
        }
        chunks.push_back(Chunk{std::move(chunk_text_value), start, end});

        if (end == tokens.size()) break;  // last (possibly short) window emitted
        start += step;
    }

    return chunks;
}

}  // namespace retrieval_engine
