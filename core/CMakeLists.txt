# The core engine library.
#
# retrieval_engine.cpp is a thin Pimpl facade implementing the public
# interface (include/retrieval_engine/retrieval_engine.hpp) by delegating to
# two internal, independently-testable stores (dummy_vector_store.cpp:
# Stage 0's scaffolding; chunk_store.cpp: Stage 1's real feature), which in
# turn share the generic SQLite RAII helpers in sqlite_util.cpp and the
# usearch capacity helper in usearch_util.hpp (header-only). chunking.cpp
# (Stage 1) implements the token-window chunker, used by callers directly
# rather than by any store.
add_library(retrieval_engine_core
    src/chunk_store.cpp
    src/chunking.cpp
    src/dummy_vector_store.cpp
    src/retrieval_engine.cpp
    src/sqlite_util.cpp
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
