#include "onnx_text_embedder.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace retrieval_engine::detail {

OnnxTextEmbedder::InputSlot OnnxTextEmbedder::ClassifyInput(const std::string& input_name) {
    if (input_name == "input_ids") return InputSlot::kInputIds;
    if (input_name == "attention_mask") return InputSlot::kAttentionMask;
    if (input_name == "token_type_ids") return InputSlot::kTokenTypeIds;
    throw std::runtime_error("OnnxTextEmbedder: model has an unexpected input '" + input_name +
                             "' (expected input_ids / attention_mask / token_type_ids)");
}

OnnxTextEmbedder::OnnxTextEmbedder(const std::string& model_path, const std::string& vocab_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "retrieval_engine"),
      session_([&] {
          try {
              return Ort::Session(env_, model_path.c_str(), Ort::SessionOptions{});
          } catch (const Ort::Exception& e) {
              throw std::runtime_error("OnnxTextEmbedder: failed to load ONNX model '" + model_path + "': " + e.what());
          }
      }()),
      tokenizer_(vocab_path),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)) {
    Ort::AllocatorWithDefaultOptions allocator;
    ReadModelSignature(allocator, model_path);
}

void OnnxTextEmbedder::ReadModelSignature(Ort::AllocatorWithDefaultOptions& allocator, const std::string& model_path) {
    const std::size_t input_count = session_.GetInputCount();
    if (input_count < 2 || input_count > 3) {
        throw std::runtime_error("OnnxTextEmbedder: model '" + model_path + "' has " + std::to_string(input_count) +
                                 " inputs; expected 2 or 3 (input_ids, attention_mask[, token_type_ids])");
    }

    input_names_.reserve(input_count);
    input_slot_.reserve(input_count);
    for (std::size_t i = 0; i < input_count; ++i) {
        std::string name = session_.GetInputNameAllocated(i, allocator).get();
        input_slot_.push_back(ClassifyInput(name));
        input_names_.push_back(std::move(name));
    }

    if (session_.GetOutputCount() < 1) {
        throw std::runtime_error("OnnxTextEmbedder: model '" + model_path + "' has no outputs");
    }
    output_name_ = session_.GetOutputNameAllocated(0, allocator).get();

    const std::vector<std::int64_t> output_shape = session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (output_shape.empty() || output_shape.back() <= 0) {
        throw std::runtime_error("OnnxTextEmbedder: cannot determine embedding dimension from model '" + model_path +
                                 "' output shape");
    }
    dimension_ = static_cast<std::size_t>(output_shape.back());

    // Build the c_str views only now that both name containers are final
    // (no further push_back, so no reallocation can invalidate them).
    input_name_ptrs_.reserve(input_names_.size());
    for (const std::string& name : input_names_) input_name_ptrs_.push_back(name.c_str());
    output_name_ptrs_.push_back(output_name_.c_str());
}

std::vector<float> OnnxTextEmbedder::embed(const std::string& text) const {
    // `tokens` is non-const and outlives Run(): ONNX Runtime borrows these
    // buffers rather than copying them, so we hand it tokenizer output
    // directly instead of copying into scratch vectors.
    TokenizedText tokens = tokenizer_.encode(text);
    const auto seq_len = static_cast<std::int64_t>(tokens.input_ids.size());
    const std::array<std::int64_t, 2> tensor_shape{1, seq_len};

    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(input_slot_.size());
    for (const InputSlot slot : input_slot_) {
        std::vector<std::int64_t>* source = nullptr;
        switch (slot) {
            case InputSlot::kInputIds:
                source = &tokens.input_ids;
                break;
            case InputSlot::kAttentionMask:
                source = &tokens.attention_mask;
                break;
            case InputSlot::kTokenTypeIds:
                source = &tokens.token_type_ids;
                break;
        }
        input_tensors.push_back(Ort::Value::CreateTensor<std::int64_t>(memory_info_, source->data(), source->size(),
                                                                       tensor_shape.data(), tensor_shape.size()));
    }

    std::vector<Ort::Value> outputs =
        session_.Run(Ort::RunOptions{nullptr}, input_name_ptrs_.data(), input_tensors.data(), input_tensors.size(),
                     output_name_ptrs_.data(), output_name_ptrs_.size());

    // last_hidden_state: [1, seq_len, dimension_], row-major.
    const float* hidden_state = outputs.front().GetTensorData<float>();

    // Attention-mask-weighted mean pooling over the token axis, matching
    // sentence-transformers' default pooling for these models. Accumulate
    // in double for numerical stability, emit float.
    std::vector<double> pooled(dimension_, 0.0);
    std::int64_t valid_tokens = 0;
    for (std::int64_t t = 0; t < seq_len; ++t) {
        if (tokens.attention_mask[static_cast<std::size_t>(t)] == 0) continue;
        ++valid_tokens;
        const float* row = hidden_state + static_cast<std::size_t>(t) * dimension_;
        for (std::size_t d = 0; d < dimension_; ++d) pooled[d] += static_cast<double>(row[d]);
    }
    const double inverse_count = valid_tokens > 0 ? 1.0 / static_cast<double>(valid_tokens) : 0.0;

    double sum_of_squares = 0.0;
    for (std::size_t d = 0; d < dimension_; ++d) {
        pooled[d] *= inverse_count;
        sum_of_squares += pooled[d] * pooled[d];
    }

    std::vector<float> embedding(dimension_, 0.0f);
    if (sum_of_squares > 0.0) {
        const double inverse_norm = 1.0 / std::sqrt(sum_of_squares);
        for (std::size_t d = 0; d < dimension_; ++d) {
            embedding[d] = static_cast<float>(pooled[d] * inverse_norm);
        }
    }
    return embedding;
}

}  // namespace retrieval_engine::detail
