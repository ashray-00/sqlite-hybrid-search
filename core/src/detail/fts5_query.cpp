#include "fts5_query.hpp"

#include <string_view>
#include <vector>

namespace retrieval_engine::detail {

std::string BuildSafeFts5MatchQuery(const std::string& query_text) {
    std::vector<std::string> quoted_terms;

    const std::string_view remaining(query_text);
    std::size_t pos = 0;
    while (pos < remaining.size()) {
        pos = remaining.find_first_not_of(" \t\n\r\f\v", pos);
        if (pos == std::string_view::npos) break;

        std::size_t end = remaining.find_first_of(" \t\n\r\f\v", pos);
        if (end == std::string_view::npos) end = remaining.size();

        std::string escaped_term;
        escaped_term.reserve(end - pos);
        for (std::size_t i = pos; i < end; ++i) {
            escaped_term += remaining[i];
            if (remaining[i] == '"') escaped_term += '"';  // "" is FTS5's escape for a literal quote
        }
        quoted_terms.push_back("\"" + escaped_term + "\"");

        pos = end;
    }

    if (quoted_terms.empty()) return {};

    std::string match_query = quoted_terms[0];
    for (std::size_t i = 1; i < quoted_terms.size(); ++i) match_query += " OR " + quoted_terms[i];
    return match_query;
}

}  // namespace retrieval_engine::detail
