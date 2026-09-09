# The core engine library.
#
# Layout mirrors include/ vs src/ one level deeper: src/ holds the
# translation units implementing the *public* API (retrieval_engine.cpp,
# chunking.cpp), each internal (non-public) component's header and .cpp live
# together under src/detail/, matching the retrieval_engine::detail
# namespace they're in --
#   detail/chunk_store.*    Stage 1's real feature, used by retrieval_engine.cpp
#   detail/sqlite_util.*    generic SQLite RAII helpers, shared by detail/ components
#   detail/usearch_util.hpp usearch capacity helper (header-only), shared likewise
# (Stage 0's scaffolding, detail/dummy_vector_store.*, was removed once
# Stage 1 shipped the real feature and nothing else depended on it -- see
# docs/DECISIONS.md.)
add_library(retrieval_engine_core
    src/chunking.cpp
    src/detail/chunk_store.cpp
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
