# Stage 0: the core engine library.
#
# Phase 2 (GREEN): retrieval_engine.cpp now implements the RetrievalEngine
# interface declared in include/retrieval_engine/retrieval_engine.hpp, so
# this is a real compiled library rather than a header-only INTERFACE target.
add_library(retrieval_engine_core
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
