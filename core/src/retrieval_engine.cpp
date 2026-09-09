#include "retrieval_engine/retrieval_engine.hpp"

#include "chunk_store.hpp"
#include "dummy_vector_store.hpp"
#include "sqlite_util.hpp"

#include <stdexcept>

namespace retrieval_engine {

// The private implementation (Pimpl idiom) -- keeps usearch/SQLite types out
// of the public header. A thin composition of the one SQLite connection and
// the two independent stores built on top of it: DummyVectorStore (Stage 0's
// scaffolding) and ChunkStore (Stage 1's real feature). Each store owns its
// own table(s) and its own usearch index and is independently testable and
// (eventually) deletable -- see docs/DECISIONS.md for why there are two
// stores instead of one, and core/src/*.hpp for each store's own docs.
//
// All three members are constructed via the initializer list, in
// declaration order (connection, then the stores that need it already
// open), so ordinary exception-during-construction rules keep this
// leak-safe with no manual cleanup: if `chunk_store`'s constructor throws,
// `dummy_store` and `connection` (already fully constructed) are
// automatically destroyed in reverse order.
struct RetrievalEngine::Impl {
    detail::SqliteConnection connection;
    detail::DummyVectorStore dummy_store;
    detail::ChunkStore chunk_store;

    Impl(const std::string& db_path, std::size_t dim)
        : connection(db_path), dummy_store(connection.get(), dim), chunk_store(connection.get(), dim) {}
};

RetrievalEngine::RetrievalEngine(const std::string& db_path, std::size_t dim) {
    if (dim == 0) throw std::invalid_argument("RetrievalEngine: dim must be greater than zero");
    impl_ = std::make_unique<Impl>(db_path, dim);
}

RetrievalEngine::~RetrievalEngine() = default;
RetrievalEngine::RetrievalEngine(RetrievalEngine&&) noexcept = default;
RetrievalEngine& RetrievalEngine::operator=(RetrievalEngine&&) noexcept = default;

void RetrievalEngine::add_vector(std::uint64_t id, const std::vector<float>& vector) {
    impl_->dummy_store.add_vector(id, vector);
}

std::vector<std::uint64_t> RetrievalEngine::search(const std::vector<float>& query, std::size_t k) const {
    return impl_->dummy_store.search(query, k);
}

std::size_t RetrievalEngine::dummy_table_row_count() const { return impl_->dummy_store.row_count(); }

void RetrievalEngine::add_documents(const std::vector<DocumentInput>& documents) {
    impl_->chunk_store.add_documents(documents);
}

std::vector<ChunkSearchResult> RetrievalEngine::search_chunks(const std::vector<float>& query, std::size_t k) const {
    return impl_->chunk_store.search_chunks(query, k);
}

std::size_t RetrievalEngine::chunk_count() const { return impl_->chunk_store.chunk_count(); }

}  // namespace retrieval_engine
