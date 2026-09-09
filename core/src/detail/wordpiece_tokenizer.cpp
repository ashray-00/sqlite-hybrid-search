#include "wordpiece_tokenizer.hpp"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace retrieval_engine::detail {

namespace {

bool IsAsciiWhitespace(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

// BERT's basic tokenizer treats these ASCII characters -- plus anything in
// Unicode's punctuation categories, which we don't decode -- as standalone
// tokens.
bool IsAsciiPunctuation(unsigned char c) {
    return (c >= '!' && c <= '/') || (c >= ':' && c <= '@') || (c >= '[' && c <= '`') || (c >= '{' && c <= '~');
}

unsigned char AsciiLower(unsigned char c) {
    return (c >= 'A' && c <= 'Z') ? static_cast<unsigned char>(c - 'A' + 'a') : c;
}

// Decodes the UTF-8 codepoint starting at bytes[i], advancing i past it.
// Returns 0xFFFD (and advances one byte) on malformed input -- good enough
// for the one thing we need codepoints for: isolating CJK characters.
std::uint32_t NextCodepoint(const std::string& bytes, std::size_t& i) {
    const auto b0 = static_cast<unsigned char>(bytes[i]);
    std::size_t extra = 0;
    std::uint32_t cp = 0;
    if (b0 < 0x80) {
        cp = b0;
    } else if ((b0 & 0xE0) == 0xC0) {
        cp = b0 & 0x1F;
        extra = 1;
    } else if ((b0 & 0xF0) == 0xE0) {
        cp = b0 & 0x0F;
        extra = 2;
    } else if ((b0 & 0xF8) == 0xF0) {
        cp = b0 & 0x07;
        extra = 3;
    } else {
        ++i;
        return 0xFFFD;
    }
    if (i + extra >= bytes.size()) {
        ++i;
        return 0xFFFD;
    }
    for (std::size_t k = 1; k <= extra; ++k) {
        const auto bk = static_cast<unsigned char>(bytes[i + k]);
        if ((bk & 0xC0) != 0x80) {
            ++i;
            return 0xFFFD;
        }
        cp = (cp << 6) | (bk & 0x3F);
    }
    i += extra + 1;
    return cp;
}

bool IsCjkCodepoint(std::uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x20000 && cp <= 0x2A6DF) ||
           (cp >= 0x2A700 && cp <= 0x2B73F) || (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}

// Basic tokenization: lowercase ASCII, put spaces around CJK codepoints and
// ASCII punctuation, then split on whitespace. Non-ASCII, non-CJK bytes
// pass through untouched and stay attached to their surrounding word.
std::vector<std::string> BasicTokenize(const std::string& text) {
    std::string spaced;
    spaced.reserve(text.size() + text.size() / 4);

    for (std::size_t i = 0; i < text.size();) {
        const auto c = static_cast<unsigned char>(text[i]);
        if (c < 0x80) {
            if (IsAsciiWhitespace(c)) {
                spaced.push_back(' ');
            } else if (IsAsciiPunctuation(c)) {
                spaced.push_back(' ');
                spaced.push_back(static_cast<char>(c));
                spaced.push_back(' ');
            } else {
                spaced.push_back(static_cast<char>(AsciiLower(c)));
            }
            ++i;
        } else {
            const std::size_t start = i;
            const std::uint32_t cp = NextCodepoint(text, i);
            if (IsCjkCodepoint(cp)) {
                spaced.push_back(' ');
                spaced.append(text, start, i - start);
                spaced.push_back(' ');
            } else {
                spaced.append(text, start, i - start);
            }
        }
    }

    std::vector<std::string> tokens;
    std::string current;
    for (const char ch : spaced) {
        if (ch == ' ') {
            if (!current.empty()) {
                tokens.push_back(std::move(current));
                current.clear();
            }
        } else {
            current.push_back(ch);
        }
    }
    if (!current.empty()) tokens.push_back(std::move(current));
    return tokens;
}

}  // namespace

WordPieceTokenizer::WordPieceTokenizer(const std::string& vocab_path) {
    std::ifstream in(vocab_path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("WordPieceTokenizer: cannot open vocab file '" + vocab_path +
                                 "' (no such file or not readable)");
    }

    std::string line;
    std::int64_t next_id = 0;
    while (std::getline(in, line)) {
        // vocab.txt is LF; tolerate a stray CR from a CRLF checkout.
        if (!line.empty() && line.back() == '\r') line.pop_back();
        vocab_.emplace(line, next_id++);
    }
    if (vocab_.empty()) {
        throw std::runtime_error("WordPieceTokenizer: vocab file '" + vocab_path + "' is empty");
    }

    const auto require = [&](const char* token) -> std::int64_t {
        const auto it = vocab_.find(token);
        if (it == vocab_.end()) {
            throw std::runtime_error("WordPieceTokenizer: vocab file '" + vocab_path +
                                     "' is missing the required special token '" + token + "'");
        }
        return it->second;
    };
    cls_id_ = require("[CLS]");
    sep_id_ = require("[SEP]");
    unk_id_ = require("[UNK]");
    require("[PAD]");  // validated present (a real BERT vocab has it); not emitted -- input is unbatched
}

TokenizedText WordPieceTokenizer::encode(const std::string& text) const {
    TokenizedText out;
    out.input_ids.push_back(cls_id_);

    // Scratch buffers reused across every word, so the per-word WordPiece
    // matching allocates nothing after the first few iterations.
    std::vector<std::int64_t> pieces;
    std::string candidate;

    for (const std::string& word : BasicTokenize(text)) {
        if (out.input_ids.size() >= kMaxTokens - 1) break;  // leave room for [SEP]

        if (word.size() > kMaxCharsPerWord) {
            out.input_ids.push_back(unk_id_);
            continue;
        }

        // Greedy longest-match-first WordPiece over `word`'s bytes.
        pieces.clear();
        std::size_t start = 0;
        bool ok = true;
        while (start < word.size()) {
            std::size_t end = word.size();
            std::int64_t matched = -1;
            while (start < end) {
                candidate.clear();
                if (start > 0) candidate.append("##");
                candidate.append(word, start, end - start);
                const auto it = vocab_.find(candidate);
                if (it != vocab_.end()) {
                    matched = it->second;
                    break;
                }
                --end;
            }
            if (matched < 0) {
                ok = false;
                break;
            }
            pieces.push_back(matched);
            start = end;
        }

        if (ok) {
            for (const std::int64_t piece_id : pieces) {
                if (out.input_ids.size() >= kMaxTokens - 1) break;
                out.input_ids.push_back(piece_id);
            }
        } else {
            out.input_ids.push_back(unk_id_);
        }
    }

    out.input_ids.push_back(sep_id_);

    out.attention_mask.assign(out.input_ids.size(), 1);
    out.token_type_ids.assign(out.input_ids.size(), 0);
    return out;
}

}  // namespace retrieval_engine::detail
