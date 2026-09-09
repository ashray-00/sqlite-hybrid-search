// Nanobind bindings for retrieval_engine::RetrievalEngine, chunk_text(), and
// their supporting structs. Deliberately kept close to the actual C++ API --
// ergonomic adaptation for Python (dict-shaped documents, plain-dict
// results) lives in the pure-Python Engine wrapper
// (python/retrieval_engine/__init__.py), not here. This module is compiled
// as `_retrieval_engine_ext` and imported only by that package, never
// directly by end users (see its own docstring for why).
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "retrieval_engine/chunking.hpp"
#include "retrieval_engine/retrieval_engine.hpp"

namespace nb = nanobind;
using namespace retrieval_engine;

NB_MODULE(_retrieval_engine_ext, m) {
    m.doc() = "Internal nanobind extension for retrieval_engine -- import retrieval_engine instead.";

    nb::class_<Chunk>(m, "Chunk")
        .def_ro("text", &Chunk::text)
        .def_ro("start_token", &Chunk::start_token)
        .def_ro("end_token", &Chunk::end_token);

    m.def("chunk_text", &chunk_text, nb::arg("text"), nb::arg("window_tokens"), nb::arg("overlap_tokens"),
          nb::call_guard<nb::gil_scoped_release>(),
          "Split `text` into overlapping token windows. See chunking.hpp for the exact contract.");

    nb::class_<DocumentChunkInput>(m, "DocumentChunkInput")
        .def(nb::init<>())
        .def_rw("text", &DocumentChunkInput::text)
        .def_rw("embedding", &DocumentChunkInput::embedding)
        .def_rw("start_token", &DocumentChunkInput::start_token)
        .def_rw("end_token", &DocumentChunkInput::end_token)
        .def_rw("created_at_unix_seconds", &DocumentChunkInput::created_at_unix_seconds);

    nb::class_<DocumentInput>(m, "DocumentInput")
        .def(nb::init<>())
        .def_rw("document_id", &DocumentInput::document_id)
        .def_rw("metadata", &DocumentInput::metadata)
        // `chunks` is a property backed by a get/set pair (nanobind/stl/
        // vector.h converts a fresh Python list on every read), not a live
        // reference into the C++ vector: `doc.chunks.append(x)` mutates a
        // throwaway temporary and is silently lost. Always assign the whole
        // list -- `doc.chunks = [...]` -- as every caller in this codebase
        // already does (see cli.py, __init__.py).
        .def_rw("chunks", &DocumentInput::chunks);

    nb::class_<TextDocumentInput>(m, "TextDocumentInput")
        .def(nb::init<>())
        .def_rw("document_id", &TextDocumentInput::document_id)
        .def_rw("text", &TextDocumentInput::text)
        .def_rw("metadata", &TextDocumentInput::metadata)
        .def_rw("created_at_unix_seconds", &TextDocumentInput::created_at_unix_seconds);

    nb::class_<ChunkSearchResult>(m, "ChunkSearchResult")
        .def_ro("document_id", &ChunkSearchResult::document_id)
        .def_ro("chunk_index", &ChunkSearchResult::chunk_index)
        .def_ro("text", &ChunkSearchResult::text)
        .def_ro("score", &ChunkSearchResult::score);

    nb::class_<SearchExplanation>(m, "SearchExplanation")
        .def_ro("document_id", &SearchExplanation::document_id)
        .def_ro("chunk_index", &SearchExplanation::chunk_index)
        .def_ro("text", &SearchExplanation::text)
        .def_ro("dense_present", &SearchExplanation::dense_present)
        .def_ro("dense_distance", &SearchExplanation::dense_distance)
        .def_ro("dense_rank", &SearchExplanation::dense_rank)
        .def_ro("sparse_present", &SearchExplanation::sparse_present)
        .def_ro("sparse_bm25_score", &SearchExplanation::sparse_bm25_score)
        .def_ro("sparse_rank", &SearchExplanation::sparse_rank)
        .def_ro("fused_score", &SearchExplanation::fused_score)
        .def_ro("final_rank", &SearchExplanation::final_rank)
        .def_ro("created_at_unix_seconds", &SearchExplanation::created_at_unix_seconds)
        .def_ro("age_seconds", &SearchExplanation::age_seconds)
        .def_ro("recency_factor", &SearchExplanation::recency_factor)
        .def_ro("decayed_score", &SearchExplanation::decayed_score);

    // Exposed as "NativeEngine", not "Engine": the public, documented
    // Python-facing class is retrieval_engine.Engine (a pure-Python wrapper
    // over this), with a friendlier ingestion shape -- see
    // python/retrieval_engine/__init__.py and docs/DECISIONS.md.
    nb::class_<RetrievalEngine>(m, "NativeEngine")
        // Opening an existing database rebuilds the entire dense index from
        // every persisted chunk (BUILD_PLAN.md section 5's "rebuild the
        // sidecar" architecture) -- unbounded work for a large corpus, so
        // this needs the same GIL release as add_documents()/search_*()
        // below, not just the obviously "long" methods.
        .def(nb::init<const std::string&, std::size_t>(), nb::arg("db_path"), nb::arg("dim"),
             nb::call_guard<nb::gil_scoped_release>())
        .def("add_documents", &RetrievalEngine::add_documents, nb::arg("documents"),
             nb::call_guard<nb::gil_scoped_release>())
        .def("chunk_count", &RetrievalEngine::chunk_count)
        .def("search_dense", &RetrievalEngine::search_dense, nb::arg("query"), nb::arg("k"),
             nb::call_guard<nb::gil_scoped_release>())
        .def("search_sparse", &RetrievalEngine::search_sparse, nb::arg("query_text"), nb::arg("k"),
             nb::call_guard<nb::gil_scoped_release>())
        .def("search_hybrid", &RetrievalEngine::search_hybrid, nb::arg("query_text"), nb::arg("query_vec"),
             nb::arg("k"), nb::call_guard<nb::gil_scoped_release>())
        .def("search_explained", &RetrievalEngine::search_explained, nb::arg("query_text"), nb::arg("query_vec"),
             nb::arg("k"), nb::call_guard<nb::gil_scoped_release>())
        .def("search_memory", &RetrievalEngine::search_memory, nb::arg("query_text"), nb::arg("query_vec"),
             nb::arg("k"), nb::arg("decay_lambda"), nb::call_guard<nb::gil_scoped_release>())
        .def("search_memory_explained", &RetrievalEngine::search_memory_explained, nb::arg("query_text"),
             nb::arg("query_vec"), nb::arg("k"), nb::arg("decay_lambda"),
             nb::call_guard<nb::gil_scoped_release>())
        // --- Built-in local embedding model ---
        // load_embedding_model() reads a model file off disk and, for a
        // real backend, would load weights into memory -- unbounded work,
        // so it releases the GIL like the other heavy methods. embed()/
        // add_text()/search_text() run inference and release it too.
        .def("load_embedding_model", &RetrievalEngine::load_embedding_model, nb::arg("model_path"),
             nb::call_guard<nb::gil_scoped_release>())
        .def("has_embedding_model", &RetrievalEngine::has_embedding_model)
        .def("embedding_dim", &RetrievalEngine::embedding_dim)
        .def("embed", &RetrievalEngine::embed, nb::arg("text"), nb::call_guard<nb::gil_scoped_release>())
        .def("add_text", &RetrievalEngine::add_text, nb::arg("documents"),
             nb::call_guard<nb::gil_scoped_release>())
        .def("search_text", &RetrievalEngine::search_text, nb::arg("query_text"), nb::arg("k"),
             nb::arg("decay_lambda") = 0.0f, nb::call_guard<nb::gil_scoped_release>());
}
