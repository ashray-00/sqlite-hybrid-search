#pragma once

#include <string>

// Pure, storage-agnostic FTS5 query-string handling: no SQLite dependency,
// independently unit-testable. Internal implementation detail, not part of
// the public API.
namespace retrieval_engine::detail {

// FTS5's MATCH operand isn't a literal string -- it's parsed by FTS5's own
// query grammar (AND/OR/NOT, quoted phrases, a leading '-' meaning NOT,
// '*' prefix queries, "column:" filters, NEAR()). Passing raw, untrusted
// search text straight through means an ordinary hyphenated word or an
// unbalanced quote -- both entirely normal in real search input -- throws
// a SQLite error instead of matching literal text (verified against our
// exact linked SQLite3: `"ZXQ7742 -stock"` raises "no such column: stock",
// and `hello "world` raises "unterminated string").
//
// Splits `query_text` on whitespace and wraps each token in its own
// double-quoted phrase (escaping embedded '"' by doubling, FTS5's string
// escape -- confirmed against our linked build), OR'd together. Wrapping in
// quotes takes every character out of FTS5's operator grammar entirely: a
// quoted phrase is tokenized like ordinary document text, not parsed for
// operators, so nothing the caller types can be interpreted as query syntax
// -- it can only ever mean "search for this literal text". OR (rather than
// implicit AND) matches typical keyword-search UX: find chunks containing
// any of the given terms, ranked by BM25, not requiring every term present.
//
// Returns an empty string if `query_text` has no non-whitespace content.
std::string BuildSafeFts5MatchQuery(const std::string& query_text);

}  // namespace retrieval_engine::detail
