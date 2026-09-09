#include "onnx_text_embedder.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace retrieval_engine::detail {

namespace {

OnnxTextEmbedder::InputSlot ClassifyInput(const std::string& name) {
    if (name == "input_ids") return OnnxTextEmbedder::InputSlot::kInputIds;
    if (name == "attention_mask") return OnnxTextEmbedder::InputSlot::kAttentionMask;
    if (name == "token_type_ids") return OnnxTextEmbedder::InputSlot::kTokenTypeIds;
    throw std::runtime_error("OnnxTextEmbedder: model has an unexpected input '" + name +
                             "' (expected input_ids / attention_mask / token_type_ids)");
}

}  // namespace

OnnxTextEmbedder::OnnxTextEmbedder(const std::string& model_path, const std::string& vocab_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "retrieval_engine"),
      session_options_(),
      session_([&] {
          try {
              return Ort::Session(env_, model_path.c_str(), session_options_);
          } catch (const Ort::Exception& e) {
              throw std::runtime_error("OnnxTextEmbedder: failed to load ONNX model '" + model_path +
                                       "': " + e.what());
          }
      }()),
      tokenizer_(vocab_path) {
    Ort::AllocatorWithDefaultOptions allocator;

    const std::size_t input_count = session_.GetInputCount();
    if (input_count < 2 || input_count > 3) {
        throw std::runtime_error("OnnxTextEmbedder: model '" + model_path + "' has " +
                                 std::to_string(input_count) +
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

    const std::vector<std::int64_t> output_shape =
        session_.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (output_shape.empty() || output_shape.back() <= 0) {
        throw std::runtime_error("OnnxTextEmbedder: cannot determine embedding dimension from model '" +
                                 model_path + "' output shape");
    }
    dimension_ = static_cast<std::size_t>(output_shape.back());

    // Build the c_str views only now that both name containers are final
    // (no further push_back, so no reallocation can invalidate them).
    input_name_ptrs_.reserve(input_names_.size());
    for (const std::string& name : input_names_) input_name_ptrs_.push_back(name.c_str());
    output_name_ptrs_.push_back(output_name_.c_str());
}

std::vector<float> OnnxTextEmbedder::embed(const std::string& text) const {
    const TokenizedText tokens = tokenizer_.encode(text);
    const auto seq_len = static_cast<std::int64_t>(tokens.input_ids.size());
    const std::array<std::int64_t, 2> tensor_shape{1, seq_len};

    const Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    // ONNX Runtime borrows (does not copy) these buffers for the Run()
    // call, so they must outlive it -- they are locals that do.
    std::vector<std::int64_t> input_ids = tokens.input_ids;
    std::vector<std::int64_t> attention_mask = tokens.attention_mask;
    std::vector<std::int64_t> token_type_ids = tokens.token_type_ids;

    std::vector<Ort::Value> input_tensors;
    input_tensors.reserve(input_slot_.size());
    for (const InputSlot slot : input_slot_) {
        std::vector<std::int64_t>* source = nullptr;
        switch (slot) {
            case InputSlot::kInputIds:
                source = &input_ids;
                break;
            case InputSlot::kAttentionMask:
                source = &attention_mask;
                break;
            case InputSlot::kTokenTypeIds:
                source = &token_type_ids;
                break;
        }
        input_tensors.push_back(Ort::Value::CreateTensor<std::int64_t>(
            memory_info, source->data(), source->size(), tensor_shape.data(), tensor_shape.size()));
    }

    std::vector<Ort::Value> outputs =
        session_.Run(Ort::RunOptions{nullptr}, input_name_ptrs_.data(), input_tensors.data(),
                     input_tensors.size(), output_name_ptrs_.data(), output_name_ptrs_.size());

    // last_hidden_state: [1, seq_len, dimension_], row-major.
    const float* hidden_state = outputs.front().GetTensorData<float>();

    // Attention-mask-weighted mean pooling over the token axis, matching
    // sentence-transformers' default pooling for these models.
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
