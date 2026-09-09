# The core engine library.
#
# Layout mirrors include/ vs src/ one level deeper: src/ holds the
# translation units implementing the *public* API (retrieval_engine.cpp,
# chunking.cpp), each internal (non-public) component's header and .cpp live
# together under src/detail/, matching the retrieval_engine::detail
# namespace they're in. One responsibility per component (SOLID/SRP) --
#   detail/chunk_store.*      orchestrates the four below for
#                              add_documents()/search_dense()/search_sparse()/
#                              search_hybrid()/search_explained()
#   detail/chunk_repository.* SQLite persistence: documents/chunks/chunks_fts
#   detail/dense_index.*      storage-agnostic usearch wrapper
#   detail/fts5_query.*       pure FTS5 MATCH-string sanitization
#   detail/rrf_fusion.*       pure Reciprocal Rank Fusion math
#   detail/recency_decay.*    pure exponential recency-decay math
#   detail/text_embedder.hpp   abstract text-to-vector interface (the seam
#                              for the built-in embed()/add_text()/
#                              search_text() path)
#   detail/mock_text_embedder.* deterministic hashing-trick TextEmbedder
#   detail/wordpiece_tokenizer.* BERT WordPiece tokenizer (ONNX build only)
#   detail/onnx_text_embedder.*  ONNX Runtime TextEmbedder (ONNX build only)
#   detail/embedding_model_loader.* sniffs a model file -> a TextEmbedder
#   detail/sqlite_util.*      generic SQLite RAII helpers, shared throughout
#   detail/time_util.hpp      "current time" helper (header-only), likewise
#   detail/usearch_util.hpp   usearch capacity helper (header-only), likewise
add_library(sqlite_hybrid_search_core
    src/chunking.cpp
    src/detail/chunk_repository.cpp
    src/detail/chunk_store.cpp
    src/detail/dense_index.cpp
    src/detail/embedding_model_loader.cpp
    src/detail/fts5_query.cpp
    src/detail/mock_text_embedder.cpp
    src/detail/read_connection_pool.cpp
    src/detail/recency_decay.cpp
    src/detail/rrf_fusion.cpp
    src/detail/sqlite_util.cpp
    src/retrieval_engine.cpp
)
add_library(sqlite_hybrid_search::core ALIAS sqlite_hybrid_search_core)

target_include_directories(sqlite_hybrid_search_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(sqlite_hybrid_search_core PUBLIC usearch::usearch Threads::Threads)

# CMake's FindSQLite3 only defines the SQLite3::SQLite3 imported target when
# it locates the library itself, not just the header (e.g. a machine with
# sqlite3.h but no libsqlite3.so dev symlink). Fall back to the raw result
# variables so the build still works there.
if(TARGET SQLite3::SQLite3)
    target_link_libraries(sqlite_hybrid_search_core PUBLIC SQLite3::SQLite3)
else()
    target_link_libraries(sqlite_hybrid_search_core PUBLIC ${SQLite3_LIBRARIES})
    target_include_directories(sqlite_hybrid_search_core SYSTEM PUBLIC ${SQLite3_INCLUDE_DIRS})
endif()

target_compile_features(sqlite_hybrid_search_core PUBLIC cxx_std_17)

# The ONNX Runtime embedding backend is compiled in only when the library
# was found (see the top-level CMakeLists.txt). RETRIEVAL_ENGINE_WITH_ONNX
# is PUBLIC so both this library's own translation units and the test
# targets can guard ONNX-only code on it.
if(RETRIEVAL_ENGINE_WITH_ONNX)
    target_sources(sqlite_hybrid_search_core PRIVATE
        src/detail/onnx_text_embedder.cpp
        src/detail/wordpiece_tokenizer.cpp
    )
    target_link_libraries(sqlite_hybrid_search_core PRIVATE onnxruntime::onnxruntime)
    target_compile_definitions(sqlite_hybrid_search_core PUBLIC RETRIEVAL_ENGINE_WITH_ONNX)
endif()

if(RETRIEVAL_ENGINE_BUILD_TESTS)
    add_subdirectory(tests)
endif()
