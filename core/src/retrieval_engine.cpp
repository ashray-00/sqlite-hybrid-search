#include "retrieval_engine/retrieval_engine.hpp"

#include "detail/chunk_store.hpp"
#include "detail/sqlite_util.hpp"

#include <stdexcept>

namespace retrieval_engine {

// The private implementation (Pimpl idiom) -- keeps usearch/SQLite types out
// of the public header. A thin composition of the one SQLite connection and
// the ChunkStore built on top of it -- see core/src/detail/chunk_store.hpp
// for its own docs, and docs/DECISIONS.md for the retired DummyVectorStore
// that used to sit alongside it (Stage 0's scaffolding, removed once Stage 1
// shipped the real feature and nothing else depended on it).
//
// Both members are constructed via the initializer list, in declaration
// order (connection, then the store that needs it already open), so
// ordinary exception-during-construction rules keep this leak-safe with no
// manual cleanup: if `chunk_store`'s constructor throws, `connection`
// (already fully constructed) is automatically destroyed.
struct RetrievalEngine::Impl {
    detail::SqliteConnection connection;
    detail::ChunkStore chunk_store;

    Impl(const std::string& db_path, std::size_t dim) : connection(db_path), chunk_store(connection.get(), dim) {}
};

RetrievalEngine::RetrievalEngine(const std::string& db_path, std::size_t dim) {
    if (dim == 0) throw std::invalid_argument("RetrievalEngine: dim must be greater than zero");
    impl_ = std::make_unique<Impl>(db_path, dim);
}

RetrievalEngine::~RetrievalEngine() = default;
RetrievalEngine::RetrievalEngine(RetrievalEngine&&) noexcept = default;
RetrievalEngine& RetrievalEngine::operator=(RetrievalEngine&&) noexcept = default;

void RetrievalEngine::add_documents(const std::vector<DocumentInput>& documents) {
    impl_->chunk_store.add_documents(documents);
}

std::vector<ChunkSearchResult> RetrievalEngine::search_chunks(const std::vector<float>& query, std::size_t k) const {
    return impl_->chunk_store.search_chunks(query, k);
}

std::size_t RetrievalEngine::chunk_count() const { return impl_->chunk_store.chunk_count(); }

}  // namespace retrieval_engine
