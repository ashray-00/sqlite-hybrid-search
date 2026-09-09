#include "dense_index.hpp"

#include "usearch_util.hpp"

#include <stdexcept>

namespace retrieval_engine::detail {

using unum::usearch::index_dense_t;
using unum::usearch::metric_kind_t;
using unum::usearch::metric_punned_t;

DenseIndex::DenseIndex(std::size_t dim)
    : dimensions_(dim), index_(index_dense_t::make(metric_punned_t(dim, metric_kind_t::cos_k))) {}

void DenseIndex::Add(std::uint64_t key, const std::vector<float>& embedding) {
    if (embedding.size() != dimensions_)
        throw std::invalid_argument("DenseIndex::Add: embedding size does not match index dimensionality");

    EnsureCapacity(index_, "DenseIndex::Add");
    const auto add_result = index_.add(static_cast<index_dense_t::vector_key_t>(key), embedding.data());
    if (!add_result) {
        throw std::runtime_error(std::string("DenseIndex::Add: usearch insertion failed: ") +
                                 (add_result.error.what() ? add_result.error.what() : "unknown error"));
    }
}

std::vector<std::pair<std::uint64_t, float>> DenseIndex::Search(const std::vector<float>& query, std::size_t k) const {
    if (query.size() != dimensions_)
        throw std::invalid_argument("DenseIndex::Search: query size does not match index dimensionality");

    const auto search_result = index_.search(query.data(), k);
    if (!search_result) {
        throw std::runtime_error(std::string("DenseIndex::Search: usearch query failed: ") +
                                 (search_result.error.what() ? search_result.error.what() : "unknown error"));
    }

    std::vector<std::uint64_t> keys(search_result.size());
    std::vector<float> distances(search_result.size());
    search_result.dump_to(keys.data(), distances.data());

    std::vector<std::pair<std::uint64_t, float>> hits;
    hits.reserve(keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i) hits.emplace_back(keys[i], distances[i]);
    return hits;
}

}  // namespace retrieval_engine::detail
