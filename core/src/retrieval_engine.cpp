#include "retrieval_engine/retrieval_engine.hpp"

#include "detail/chunk_store.hpp"
#include "detail/embedding_model_loader.hpp"
#include "detail/sqlite_util.hpp"
#include "detail/text_embedder.hpp"

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace retrieval_engine {

namespace {

// The usearch graph is persisted alongside the SQLite file as
// "<db_path>.usearch". In-memory databases (":memory:") and the anonymous
// temp database ("") have no stable path to sit beside, so they get no
// sidecar and rebuild the index from SQLite on every open.
std::string DeriveIndexSidecarPath(const std::string& db_path) {
    if (db_path.empty() || db_path == ":memory:") return {};
    return db_path + ".usearch";
}

// Counts whitespace-delimited tokens, matching chunk_text()'s notion of a
// "token" (a maximal run of non-whitespace) -- used only to fill in a
// single-chunk document's [start_token, end_token) span for add_text().
std::size_t CountWhitespaceTokens(const std::string& text) {
    std::size_t count = 0;
    bool in_token = false;
    for (const unsigned char c : text) {
        const bool is_space = (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v');
        if (!is_space && !in_token) ++count;
        in_token = !is_space;
    }
    return count;
}

}  // namespace

// The private implementation (Pimpl idiom) -- keeps usearch/SQLite types out
// of the public header. A thin composition of the SQLite connection(s) and
// the ChunkStore built on top of it (see core/src/detail/chunk_store.hpp).
//
// Members are constructed in declaration order (connection, then the store
// that needs it already open), so ordinary exception-during-construction
// rules keep this leak-safe with no manual cleanup: if `chunk_store`'s
// constructor throws, `connection` (already fully constructed) is
// automatically destroyed.
//
// Concurrency (ADR-11): `rw_mutex` gives one instance a single-writer /
// concurrent-reader discipline. Every public read method takes a
// std::shared_lock; add_documents()/add_text()/load_embedding_model() take a
// std::unique_lock. The lock is non-recursive, which is safe because no
// public method calls another public method -- each delegates straight to
// `chunk_store` or `require_embedder()`. A future method wanting a sibling's
// behaviour must call the private `chunk_store` path, never the public one.
// The constructor, destructor and move operations take no lock: the caller
// must not have a call in flight when an instance is destroyed or moved
// (standard C++ object lifetime).
struct RetrievalEngine::Impl {
    detail::SqliteConnection connection;
    std::size_t dim;
    detail::ChunkStore chunk_store;
    std::unique_ptr<detail::TextEmbedder> embedder;
    mutable std::shared_mutex rw_mutex;

    Impl(const std::string& db_path, std::size_t dimension)
        : connection(db_path),
          dim(dimension),
          chunk_store(connection.get(), db_path, dimension, DeriveIndexSidecarPath(db_path)) {}

    // Every raw-text method funnels through here so the "no model attached"
    // error message is written once and is identical everywhere.
    const detail::TextEmbedder& require_embedder() const {
        if (!embedder) {
            throw std::logic_error("RetrievalEngine: no embedding model attached -- call load_embedding_model() first");
        }
        return *embedder;
    }
};

RetrievalEngine::RetrievalEngine(const std::string& db_path, std::size_t dim) {
    if (dim == 0) throw std::invalid_argument("RetrievalEngine: dim must be greater than zero");
    impl_ = std::make_unique<Impl>(db_path, dim);
}

RetrievalEngine::~RetrievalEngine() = default;
RetrievalEngine::RetrievalEngine(RetrievalEngine&&) noexcept = default;
RetrievalEngine& RetrievalEngine::operator=(RetrievalEngine&&) noexcept = default;

void RetrievalEngine::add_documents(const std::vector<DocumentInput>& documents) {
    std::unique_lock<std::shared_mutex> lock(impl_->rw_mutex);
    impl_->chunk_store.add_documents(documents);
}

std::size_t RetrievalEngine::chunk_count() const {
    std::shared_lock<std::shared_mutex> lock(impl_->rw_mutex);
    return impl_->chunk_store.chunk_count();
}

bool RetrievalEngine::loaded_index_from_sidecar() const {
    std::shared_lock<std::shared_mutex> lock(impl_->rw_mutex);
    return impl_->chunk_store.loaded_index_from_sidecar();
}

std::vector<ChunkSearchResult> RetrievalEngine::search_dense(const std::vector<float>& query, std::size_t k) const {
    std::shared_lock<std::shared_mutex> lock(impl_->rw_mutex);
    return impl_->chunk_store.search_dense(query, k);
}

std::vector<ChunkSearchResult> RetrievalEngine::search_sparse(const std::string& query_text, std::size_t k) const {
    std::shared_lock<std::shared_mutex> lock(impl_->rw_mutex);
    return impl_->chunk_store.search_sparse(query_text, k);
}

std::vector<ChunkSearchResult> RetrievalEngine::search_hybrid(const std::string& query_text,
                                                              const std::vector<float>& query_vec,
                                                              std::size_t k) const {
    std::shared_lock<std::shared_mutex> lock(impl_->rw_mutex);
    return impl_->chunk_store.search_hybrid(query_text, query_vec, k);
}

std::vector<SearchExplanation> RetrievalEngine::search_explained(const std::string& query_text,
                                                                 const std::vector<float>& query_vec,
                                                                 std::size_t k) const {
    std::shared_lock<std::shared_mutex> lock(impl_->rw_mutex);
    return impl_->chunk_store.search_explained(query_text, query_vec, k);
}

std::vector<ChunkSearchResult> RetrievalEngine::search_memory(const std::string& query_text,
                                                              const std::vector<float>& query_vec, std::size_t k,
                                                              float decay_lambda) const {
    std::shared_lock<std::shared_mutex> lock(impl_->rw_mutex);
    return impl_->chunk_store.search_memory(query_text, query_vec, k, decay_lambda);
}

std::vector<SearchExplanation> RetrievalEngine::search_memory_explained(const std::string& query_text,
                                                                        const std::vector<float>& query_vec,
                                                                        std::size_t k, float decay_lambda) const {
    std::shared_lock<std::shared_mutex> lock(impl_->rw_mutex);
    return impl_->chunk_store.search_memory_explained(query_text, query_vec, k, decay_lambda);
}

// --- Built-in local embedding model --------------------------------------

void RetrievalEngine::load_embedding_model(const std::string& model_path) {
    // Build the new embedder into a local first: if LoadTextEmbedder throws
    // (missing file, bad format, dimension mismatch), the previously
    // attached model -- if any -- stays in place untouched. Only the pointer
    // swap needs the write lock.
    std::unique_ptr<detail::TextEmbedder> loaded = detail::LoadTextEmbedder(model_path, impl_->dim);
    std::unique_lock<std::shared_mutex> lock(impl_->rw_mutex);
    impl_->embedder = std::move(loaded);
}

bool RetrievalEngine::has_embedding_model() const {
    std::shared_lock<std::shared_mutex> lock(impl_->rw_mutex);
    return impl_->embedder != nullptr;
}

std::size_t RetrievalEngine::embedding_dim() const {
    std::shared_lock<std::shared_mutex> lock(impl_->rw_mutex);
    return impl_->require_embedder().dimension();
}

std::vector<float> RetrievalEngine::embed(const std::string& text) const {
    std::shared_lock<std::shared_mutex> lock(impl_->rw_mutex);
    return impl_->require_embedder().embed(text);
}

void RetrievalEngine::add_text(const std::vector<TextDocumentInput>& documents) {
    std::unique_lock<std::shared_mutex> lock(impl_->rw_mutex);
    const detail::TextEmbedder& embedder = impl_->require_embedder();

    std::vector<DocumentInput> native_documents;
    native_documents.reserve(documents.size());
    for (const TextDocumentInput& document : documents) {
        DocumentChunkInput chunk;
        chunk.text = document.text;
        chunk.embedding = embedder.embed(document.text);
        chunk.start_token = 0;
        chunk.end_token = CountWhitespaceTokens(document.text);
        chunk.created_at_unix_seconds = document.created_at_unix_seconds;

        DocumentInput native_document;
        native_document.document_id = document.document_id;
        native_document.metadata = document.metadata;
        native_document.chunks.push_back(std::move(chunk));
        native_documents.push_back(std::move(native_document));
    }

    impl_->chunk_store.add_documents(native_documents);
}

std::vector<ChunkSearchResult> RetrievalEngine::search_text(const std::string& query_text, std::size_t k,
                                                            float decay_lambda) const {
    std::shared_lock<std::shared_mutex> lock(impl_->rw_mutex);
    const std::vector<float> query_vec = impl_->require_embedder().embed(query_text);
    return impl_->chunk_store.search_memory(query_text, query_vec, k, decay_lambda);
}

}  // namespace retrieval_engine
