#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "text_embedder.hpp"
#include "wordpiece_tokenizer.hpp"

namespace retrieval_engine::detail {

// A TextEmbedder backed by ONNX Runtime running a BERT-family sentence
// embedding model (default: all-MiniLM-L6-v2). Pipeline per embed():
// WordPiece tokenize -> run the model -> attention-mask-weighted mean pool
// over the token dimension -> L2-normalize. This matches
// sentence-transformers' own pooling for these models, so the vectors are
// directly comparable to a reference Python implementation.
//
// The model weights and the ONNX Runtime session are created once, in the
// constructor, and reused for every embed() call -- never reloaded
// per query.
//
// Thread-safety: embed() is const and safe to call concurrently from
// multiple threads. ONNX Runtime explicitly supports concurrent Run() on a
// single session; the tokenizer and every cached member below are
// immutable after construction. `session_` is `mutable` only because
// Ort::Session::Run is a non-const method, not because any caller-visible
// state changes.
class OnnxTextEmbedder final : public TextEmbedder {
public:
    // Loads the ONNX model at `model_path` and the WordPiece vocabulary at
    // `vocab_path`. Throws std::runtime_error if either file is missing or
    // invalid, or if the model's input/output signature is not the
    // expected (input_ids, attention_mask, token_type_ids) -> hidden-state
    // shape.
    OnnxTextEmbedder(const std::string& model_path, const std::string& vocab_path);

    std::size_t dimension() const override { return dimension_; }
    std::vector<float> embed(const std::string& text) const override;

private:
    // Which tokenized sequence feeds a given model input position.
    enum class InputSlot { kInputIds, kAttentionMask, kTokenTypeIds };
    static InputSlot ClassifyInput(const std::string& input_name);

    // Reads and validates the model's input/output signature, populating
    // input_slot_, the cached name pointers, and dimension_.
    void ReadModelSignature(Ort::AllocatorWithDefaultOptions& allocator, const std::string& model_path);

    Ort::Env env_;
    mutable Ort::Session session_;
    WordPieceTokenizer tokenizer_;
    Ort::MemoryInfo memory_info_;  // CPU allocator info, reused for every input tensor
    std::size_t dimension_ = 0;

    // Input names in the model's own declared order, with stable c_str
    // views into them for Ort::Session::Run, plus which tokenized sequence
    // feeds each position -- so we never rely on the exporter having used a
    // particular input order.
    std::vector<std::string> input_names_;
    std::vector<const char*> input_name_ptrs_;
    std::vector<InputSlot> input_slot_;

    std::string output_name_;
    std::vector<const char*> output_name_ptrs_;
};

}  // namespace retrieval_engine::detail
