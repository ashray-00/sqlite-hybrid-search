#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace retrieval_engine::detail {

// Abstract text-to-vector embedder: the seam between RetrievalEngine's
// raw-text API (embed()/add_text()/search_text()) and whatever local model
// actually produces vectors. One concrete implementation per backend --
// MockTextEmbedder (deterministic, dependency-free, for tests and the
// zero-setup path) today, an ONNX Runtime / GGUF backend behind this same
// interface later -- so the engine never depends on a specific runtime
// (Dependency Inversion).
//
// Thread-safety contract: dimension() and embed() are const and MUST be
// safe to call concurrently from multiple threads on the same object,
// without external locking, once the object is fully constructed.
// RetrievalEngine relies on this: it loads an embedder once
// (load_embedding_model()) and thereafter only reads it. Implementations
// therefore either hold immutable state or synchronize internally.
class TextEmbedder {
public:
    virtual ~TextEmbedder() = default;

    // Output vector width. Fixed for the lifetime of the object.
    virtual std::size_t dimension() const = 0;

    // Embeds `text` into exactly dimension() finite floats. The result is
    // L2-normalized (unit length) so callers can compare vectors by dot
    // product; text that contains no usable tokens yields the all-zeros
    // vector (the one input for which a unit-length result is impossible).
    virtual std::vector<float> embed(const std::string& text) const = 0;

    TextEmbedder(const TextEmbedder&) = delete;
    TextEmbedder& operator=(const TextEmbedder&) = delete;

protected:
    TextEmbedder() = default;
};

}  // namespace retrieval_engine::detail
