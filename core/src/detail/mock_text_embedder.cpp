#include "mock_text_embedder.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace retrieval_engine::detail {

namespace {

// FNV-1a, 64-bit. Chosen for being tiny, dependency-free and stable across
// platforms/runs -- the mock embedder's whole value is that the same text
// always maps to the same vector.
std::uint64_t Fnv1a64(const std::string& token) {
    constexpr std::uint64_t kOffsetBasis = 1469598103934665603ULL;
    constexpr std::uint64_t kPrime = 1099511628211ULL;
    std::uint64_t hash = kOffsetBasis;
    for (const unsigned char byte : token) {
        hash ^= static_cast<std::uint64_t>(byte);
        hash *= kPrime;
    }
    return hash;
}

bool IsAsciiAlphaNumeric(unsigned char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

unsigned char ToAsciiLower(unsigned char c) {
    return (c >= 'A' && c <= 'Z') ? static_cast<unsigned char>(c - 'A' + 'a') : c;
}

}  // namespace

MockTextEmbedder::MockTextEmbedder(std::size_t dimension) : dimension_(dimension) {
    if (dimension_ == 0) {
        throw std::invalid_argument("MockTextEmbedder: dimension must be greater than zero");
    }
}

std::vector<float> MockTextEmbedder::embed(const std::string& text) const {
    std::vector<double> accumulator(dimension_, 0.0);

    // Single pass: carve `text` into maximal runs of ASCII alphanumerics,
    // lowercasing as we go, and vote each token into its hashed bucket.
    std::string token;
    const auto flush_token = [&]() {
        if (token.empty()) return;
        accumulator[Fnv1a64(token) % dimension_] += 1.0;
        token.clear();
    };
    for (const char raw : text) {
        const unsigned char c = static_cast<unsigned char>(raw);
        if (IsAsciiAlphaNumeric(c)) {
            token.push_back(static_cast<char>(ToAsciiLower(c)));
        } else {
            flush_token();
        }
    }
    flush_token();

    // L2-normalize so results are directly dot-product-comparable, matching
    // the DenseIndex's cosine metric. Text with no alphanumeric tokens has
    // a zero-norm accumulator and stays the all-zeros vector.
    double sum_of_squares = 0.0;
    for (const double value : accumulator) {
        sum_of_squares += value * value;
    }

    std::vector<float> embedding(dimension_, 0.0f);
    if (sum_of_squares > 0.0) {
        const double inverse_norm = 1.0 / std::sqrt(sum_of_squares);
        for (std::size_t i = 0; i < dimension_; ++i) {
            embedding[i] = static_cast<float>(accumulator[i] * inverse_norm);
        }
    }
    return embedding;
}

}  // namespace retrieval_engine::detail
