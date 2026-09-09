#include "dense_index.hpp"

#include "usearch_util.hpp"

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>

namespace retrieval_engine::detail {

using unum::usearch::index_dense_t;
using unum::usearch::metric_kind_t;
using unum::usearch::metric_punned_t;

namespace {

// Validate a sidecar cheaply before the full Load(). usearch's
// index_dense_metadata_from_path() reads only the ~64-byte header and
// locates the "usearch" magic; on a truncated or garbage file it fails in
// microseconds, whereas index.load() can spin for tens of seconds
// interpreting random bytes as node counts.
bool SidecarHeaderIsValid(const std::string& path) {
    std::error_code ec;
    if (std::filesystem::file_size(path, ec) < 64 || ec) return false;

    auto metadata = unum::usearch::index_dense_metadata_from_path(path.c_str());
    const bool valid = static_cast<bool>(metadata);
    // usearch's error_t throws from its destructor if a set message is
    // dropped outside exception handling -- consume it explicitly.
    if (!valid) metadata.error.release();
    return valid;
}

}  // namespace

index_dense_t DenseIndex::MakeIndex(std::size_t dim) {
    return index_dense_t::make(metric_punned_t(dim, metric_kind_t::cos_k));
}

DenseIndex::DenseIndex(std::size_t dim)
    : dimensions_(dim),
      search_threads_(UsearchSearchThreadCount()),
      index_(MakeIndex(dim)),
      search_slots_(search_threads_) {
    ProvisionSearchThreads(index_, search_threads_, "DenseIndex");
}

void DenseIndex::Add(std::uint64_t key, const std::vector<float>& embedding) {
    if (embedding.size() != dimensions_)
        throw std::invalid_argument("DenseIndex::Add: embedding size does not match index dimensionality");

    EnsureCapacity(index_, search_threads_, "DenseIndex::Add");
    const auto add_result = index_.add(static_cast<index_dense_t::vector_key_t>(key), embedding.data());
    if (!add_result) {
        throw std::runtime_error(std::string("DenseIndex::Add: usearch insertion failed: ") +
                                 (add_result.error.what() ? add_result.error.what() : "unknown error"));
    }
}

std::vector<std::pair<std::uint64_t, float>> DenseIndex::Search(const std::vector<float>& query, std::size_t k) const {
    if (query.size() != dimensions_)
        throw std::invalid_argument("DenseIndex::Search: query size does not match index dimensionality");

    // Hold an explicit search-slot id for the whole call. usearch's
    // default any_thread() returns the slot to its free list the instant
    // search() returns, but search_result still points into that slot's
    // buffers, which dump_to() reads below -- a concurrent Search() reusing
    // the slot would race. The lease also caps concurrent searchers at the
    // pool size (blocking the surplus) instead of exhausting usearch's list.
    const SearchSlotPool::Lease slot = search_slots_.Acquire();
    const auto search_result = index_.search(query.data(), k, slot.id());
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

void DenseIndex::Save(const std::string& path) const {
    const std::string temp_path = path + ".tmp";

    const auto save_result = index_.save(temp_path.c_str());
    if (!save_result) {
        throw std::runtime_error(std::string("DenseIndex::Save: usearch serialisation failed: ") +
                                 (save_result.error.what() ? save_result.error.what() : "unknown error"));
    }

    std::error_code ec;
    std::filesystem::rename(temp_path, path, ec);
    if (ec) {
        std::filesystem::remove(temp_path, ec);
        throw std::runtime_error("DenseIndex::Save: could not move sidecar into place: " + ec.message());
    }
}

void DenseIndex::Load(const std::string& path) {
    if (!SidecarHeaderIsValid(path)) {
        throw std::runtime_error("DenseIndex::Load: '" + path + "' is not a readable usearch sidecar");
    }

    const auto load_result = index_.load(path.c_str());
    if (!load_result) {
        throw std::runtime_error(std::string("DenseIndex::Load: usearch deserialisation failed: ") +
                                 (load_result.error.what() ? load_result.error.what() : "unknown error"));
    }

    if (index_.dimensions() != dimensions_) {
        // The sidecar was written for a different embedding dimensionality
        // than this engine was opened with; the caller (ChunkStore) treats
        // this like any other unusable sidecar and rebuilds from SQLite.
        Clear();
        throw std::invalid_argument("DenseIndex::Load: sidecar dimensionality does not match this index");
    }

    // load() re-derived the per-thread `contexts_` count from the file --
    // re-assert this host's concurrent-search provisioning.
    ProvisionSearchThreads(index_, search_threads_, "DenseIndex::Load");
}

void DenseIndex::Clear() {
    // reset() (not `index_ = MakeIndex(...)`) so usearch runs its own full
    // teardown -- releasing any file mapping a preceding Load() established
    // and freeing the vector tape through the same path that allocated it.
    // Swapping in a fresh index instead corrupts the heap on macOS/ARM64.
    index_.reset();
    // reset() drops the per-thread `contexts_` count back to the default.
    ProvisionSearchThreads(index_, search_threads_, "DenseIndex::Clear");
}

}  // namespace retrieval_engine::detail
