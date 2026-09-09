#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace retrieval_engine::detail {

// The three parallel integer sequences a BERT-family ONNX model expects as
// input, for a single (unbatched) piece of text.
struct TokenizedText {
    std::vector<std::int64_t> input_ids;
    std::vector<std::int64_t> attention_mask;   // 1 for real tokens (all 1s here -- no padding)
    std::vector<std::int64_t> token_type_ids;   // all 0s for single-sentence input
};

// A compact, dependency-free BERT WordPiece tokenizer -- enough to feed
// all-MiniLM-L6-v2 / any bert-base-uncased vocabulary. Reproduces
// HuggingFace's uncased pipeline: basic tokenization (whitespace + ASCII
// punctuation splitting, CJK codepoints isolated, lowercased) followed by
// greedy longest-match WordPiece with "##" continuation, wrapped in
// [CLS] ... [SEP] and truncated to kMaxTokens.
//
// Not reproduced (documented limitation -- fine for English, the engine's
// stated v1 scope): Unicode accent stripping and Unicode-category
// punctuation. `strip_accents` is null for all-MiniLM-L6-v2 anyway, and
// non-ASCII punctuation simply stays attached to its word (usually
// resolving to [UNK] or subwords).
//
// Immutable after construction (only const lookup tables), so encode() is
// safe to call concurrently from multiple threads -- satisfying
// TextEmbedder's contract for the OnnxTextEmbedder built on top of it.
class WordPieceTokenizer {
public:
    // Loads `vocab_path` (one token per line, line number = id -- the
    // standard vocab.txt format). Throws std::runtime_error if the file
    // cannot be read or is missing any of [CLS] [SEP] [UNK] [PAD].
    explicit WordPieceTokenizer(const std::string& vocab_path);

    // Tokenizes `text` into model-ready ids/masks. Never throws for
    // ordinary text; unknown words become [UNK].
    TokenizedText encode(const std::string& text) const;

    std::size_t vocabulary_size() const { return vocab_.size(); }

private:
    // sentence-transformers/all-MiniLM-L6-v2 truncates at 256; matching it
    // keeps our vectors comparable to the reference implementation and
    // bounds inference cost.
    static constexpr std::size_t kMaxTokens = 256;
    static constexpr std::size_t kMaxCharsPerWord = 100;

    std::int64_t TokenId(const std::string& token) const;

    std::unordered_map<std::string, std::int64_t> vocab_;
    std::int64_t cls_id_ = 0;
    std::int64_t sep_id_ = 0;
    std::int64_t unk_id_ = 0;
    std::int64_t pad_id_ = 0;
};

}  // namespace retrieval_engine::detail
