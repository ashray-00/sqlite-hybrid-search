#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "text_embedder.hpp"

namespace retrieval_engine::detail {

// Opens the model file at `model_path`, identifies its format from the
// contents, and returns a matching TextEmbedder ready for use. This is the
// one place that knows about concrete model formats -- RetrievalEngine and
// everything above it deal only with the TextEmbedder interface.
//
// Recognized today:
//   * the mock format -- a text file whose first line is exactly
//     "RETRIEVAL_ENGINE_MOCK_EMBEDDING_MODEL v1" followed by a "dim=<n>"
//     line -- which yields a MockTextEmbedder.
// A real ONNX/GGUF backend will be added here behind the same return type.
//
// Throws:
//   * std::runtime_error  -- file missing, unreadable, or its format is not
//                            recognized / is malformed.
//   * std::invalid_argument -- format recognized but the model's output
//                            dimension does not equal `expected_dim` (the
//                            dimension the engine was constructed with).
std::unique_ptr<TextEmbedder> LoadTextEmbedder(const std::string& model_path, std::size_t expected_dim);

}  // namespace retrieval_engine::detail
