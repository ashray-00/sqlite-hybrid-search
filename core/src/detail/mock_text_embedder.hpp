#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "text_embedder.hpp"

namespace retrieval_engine::detail {

// A deterministic, dependency-free TextEmbedder used by the tests and by
// the zero-setup path when no real model is available. NOT a semantic
// model: it is the "hashing trick" -- lowercase the text, split it into
// alphanumeric tokens, add 1.0 to the vector bucket each token hashes to
// (FNV-1a, mod dimension), then L2-normalize. Two texts that share words
// therefore get a positive cosine similarity roughly proportional to their
// vocabulary overlap -- enough to exercise embed()/add_text()/search_text()
// end to end, but no more than that.
//
// Activated by load_embedding_model() when the model file's first line is
// the mock-format magic string (see embedding_model_loader.cpp).
//
// Immutable after construction (only `dimension_` is held), so it trivially
// satisfies TextEmbedder's concurrent-const-use contract.
class MockTextEmbedder final : public TextEmbedder {
public:
    explicit MockTextEmbedder(std::size_t dimension);

    std::size_t dimension() const override { return dimension_; }
    std::vector<float> embed(const std::string& text) const override;

private:
    std::size_t dimension_;
};

}  // namespace retrieval_engine::detail
