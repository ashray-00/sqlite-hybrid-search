#include "embedding_model_loader.hpp"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "mock_text_embedder.hpp"
#ifdef RETRIEVAL_ENGINE_WITH_ONNX
#include "onnx_text_embedder.hpp"
#endif

namespace retrieval_engine::detail {

namespace {

constexpr const char* kMockModelMagic = "RETRIEVAL_ENGINE_MOCK_EMBEDDING_MODEL v1";

// Trims ASCII whitespace (including a trailing CR from a CRLF file) off
// both ends -- model files are hand-written/generated text and shouldn't be
// rejected over line-ending or trailing-space differences.
std::string Trim(const std::string& s) {
    std::size_t begin = 0;
    std::size_t end = s.size();
    while (begin < end && std::isspace(static_cast<unsigned char>(s[begin]))) ++begin;
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
    return s.substr(begin, end - begin);
}

bool HasSuffix(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() && value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void ThrowIfDimensionMismatch(const std::string& model_path, std::size_t model_dim, std::size_t expected_dim) {
    if (model_dim != expected_dim) {
        throw std::invalid_argument(
            "load_embedding_model: model '" + model_path + "' produces " + std::to_string(model_dim) +
            "-dimensional vectors, but the engine was constructed for " + std::to_string(expected_dim) + " dimensions");
    }
}

// Parses the "dim=<n>" line of a mock model file into a positive size_t,
// throwing std::runtime_error (with `model_path` for context) on anything
// that isn't exactly that.
std::size_t ParseMockDimLine(const std::string& raw_line, const std::string& model_path) {
    const std::string line = Trim(raw_line);
    constexpr const char* kPrefix = "dim=";
    if (line.rfind(kPrefix, 0) != 0) {
        throw std::runtime_error("load_embedding_model: malformed mock model file '" + model_path +
                                 "': expected a 'dim=<n>' line, got '" + line + "'");
    }

    const std::string value = line.substr(std::string(kPrefix).size());
    std::size_t consumed = 0;
    long long parsed = 0;
    try {
        parsed = std::stoll(value, &consumed);
    } catch (const std::exception&) {
        consumed = 0;  // fall through to the shared error below
    }
    if (consumed != value.size() || parsed <= 0) {
        throw std::runtime_error("load_embedding_model: malformed mock model file '" + model_path + "': dim value '" +
                                 value + "' is not a positive integer");
    }
    return static_cast<std::size_t>(parsed);
}

std::unique_ptr<TextEmbedder> LoadOnnxEmbedder(const std::string& model_path, std::size_t expected_dim) {
#ifdef RETRIEVAL_ENGINE_WITH_ONNX
    // The BERT WordPiece vocabulary lives beside the .onnx file, as it does
    // in a HuggingFace model directory.
    const std::filesystem::path vocab_path = std::filesystem::path(model_path).replace_filename("vocab.txt");
    if (!std::filesystem::exists(vocab_path)) {
        throw std::runtime_error("load_embedding_model: ONNX model '" + model_path +
                                 "' needs a 'vocab.txt' beside it (looked for '" + vocab_path.string() + "')");
    }

    auto embedder = std::make_unique<OnnxTextEmbedder>(model_path, vocab_path.string());
    ThrowIfDimensionMismatch(model_path, embedder->dimension(), expected_dim);
    return embedder;
#else
    (void)expected_dim;
    throw std::runtime_error("load_embedding_model: '" + model_path +
                             "' is an ONNX model, but this build has no ONNX Runtime backend "
                             "(configure with -DRETRIEVAL_ENGINE_WITH_ONNX=ON)");
#endif
}

std::unique_ptr<TextEmbedder> LoadMockEmbedder(std::ifstream& in, const std::string& model_path,
                                               std::size_t expected_dim) {
    std::string dim_line;
    std::getline(in, dim_line);
    const std::size_t model_dim = ParseMockDimLine(dim_line, model_path);
    ThrowIfDimensionMismatch(model_path, model_dim, expected_dim);
    return std::make_unique<MockTextEmbedder>(model_dim);
}

}  // namespace

std::unique_ptr<TextEmbedder> LoadTextEmbedder(const std::string& model_path, std::size_t expected_dim) {
    std::ifstream in(model_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("load_embedding_model: cannot open model file '" + model_path +
                                 "' (no such file or not readable)");
    }

    // Dispatch by extension first for the binary ONNX format, so we never
    // read a 90 MB protobuf into a string looking for a text header.
    if (HasSuffix(model_path, ".onnx")) {
        return LoadOnnxEmbedder(model_path, expected_dim);
    }

    std::string first_line;
    std::getline(in, first_line);
    if (Trim(first_line) == kMockModelMagic) {
        return LoadMockEmbedder(in, model_path, expected_dim);
    }

    throw std::runtime_error("load_embedding_model: unrecognized embedding-model format in '" + model_path +
                             "' (first line: '" + Trim(first_line) +
                             "'). Expected a '.onnx' model or the mock-model header.");
}

}  // namespace retrieval_engine::detail
