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
#   detail/sqlite_util.*      generic SQLite RAII helpers, shared throughout
#   detail/usearch_util.hpp   usearch capacity helper (header-only), likewise
# (Stage 0's scaffolding, detail/dummy_vector_store.*, was removed once
# Stage 1 shipped the real feature and nothing else depended on it -- see
# docs/DECISIONS.md.)
add_library(retrieval_engine_core
    src/chunking.cpp
    src/detail/chunk_repository.cpp
    src/detail/chunk_store.cpp
    src/detail/dense_index.cpp
    src/detail/fts5_query.cpp
    src/detail/rrf_fusion.cpp
    src/detail/sqlite_util.cpp
    src/retrieval_engine.cpp
)
add_library(retrieval_engine::core ALIAS retrieval_engine_core)

target_include_directories(retrieval_engine_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(retrieval_engine_core PUBLIC
    usearch::usearch
    SQLite3::SQLite3
)

target_compile_features(retrieval_engine_core PUBLIC cxx_std_17)

if(RETRIEVAL_ENGINE_BUILD_TESTS)
    add_subdirectory(tests)
endif()
