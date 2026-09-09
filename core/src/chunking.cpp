#include "retrieval_engine/chunking.hpp"

#include <algorithm>
#include <stdexcept>
#include <string_view>

namespace retrieval_engine {

namespace {

constexpr std::string_view kWhitespace = " \t\n\r\f\v";

}  // namespace

std::vector<Chunk> chunk_text(const std::string& text, std::size_t window_tokens, std::size_t overlap_tokens) {
    if (window_tokens == 0) throw std::invalid_argument("chunk_text: window_tokens must be greater than zero");
    if (overlap_tokens >= window_tokens)
        throw std::invalid_argument("chunk_text: overlap_tokens must be less than window_tokens");

    // Tokenize by scanning for whitespace-delimited runs directly, rather
    // than via std::istringstream. A "token" here is a maximal run of plain
    // ASCII non-whitespace characters (see chunking.hpp) -- nothing
    // locale-sensitive -- so there's no need to pay istringstream's cost for
    // generality we don't use: a global-locale lookup on construction and a
    // virtual call through its streambuf for every character extracted.
    // This runs once per document ingested, and BUILD_PLAN.md's own Stage 1
    // scale target ("ingestion of 10k chunks") makes it worth avoiding.
    // Tokens are non-owning string_views into `text` -- no per-token
    // allocation until a chunk's text is assembled below.
    std::vector<std::string_view> tokens;
    {
        const std::string_view remaining(text);
        std::size_t pos = 0;
        while (pos < remaining.size()) {
            pos = remaining.find_first_not_of(kWhitespace, pos);
            if (pos == std::string_view::npos) break;

            const std::size_t token_end = remaining.find_first_of(kWhitespace, pos);
            const std::size_t token_len = (token_end == std::string_view::npos) ? remaining.size() - pos : token_end - pos;
            tokens.push_back(remaining.substr(pos, token_len));
            pos += token_len;
        }
    }

    std::vector<Chunk> chunks;
    if (tokens.empty()) return chunks;

    const std::size_t step = window_tokens - overlap_tokens;
    std::size_t start = 0;
    while (start < tokens.size()) {
        const std::size_t end = std::min(start + window_tokens, tokens.size());

        // Reserve once instead of letting repeated += grow (and reallocate/
        // copy) the string incrementally -- an exact upper bound, since it
        // counts one separator per token including the last, which never
        // gets written.
        std::size_t reserve_size = 0;
        for (std::size_t i = start; i < end; ++i) reserve_size += tokens[i].size() + 1;

        std::string chunk_text_value;
        chunk_text_value.reserve(reserve_size);
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
