cmake_minimum_required(VERSION 3.24)
project(retrieval_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(RETRIEVAL_ENGINE_BUILD_TESTS "Build the C++ unit tests" ON)
# OFF by default so the plain C++ dev/test build (`cmake -B build &&
# cmake --build build`) is completely unaffected; turned ON by
# pyproject.toml's scikit-build-core config when building the Python
# extension (`pip install -e .`).
option(RETRIEVAL_ENGINE_BUILD_PYTHON_BINDINGS "Build the nanobind Python extension module" OFF)

include(FetchContent)

# --- SQLite3 --------------------------------------------------------------
# Homebrew's sqlite formula is keg-only (macOS ships its own, older sqlite),
# so it is NOT linked into /opt/homebrew/include or /opt/homebrew/lib by
# default. Help CMake find the Homebrew ARM64 build explicitly, per
# CLAUDE.md's M1 Architecture Compatibility directive.
find_program(HOMEBREW_EXECUTABLE brew)
if(HOMEBREW_EXECUTABLE)
    execute_process(
        COMMAND ${HOMEBREW_EXECUTABLE} --prefix sqlite
        OUTPUT_VARIABLE HOMEBREW_SQLITE_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(HOMEBREW_SQLITE_PREFIX)
        list(APPEND CMAKE_PREFIX_PATH "${HOMEBREW_SQLITE_PREFIX}")
    endif()
endif()

find_package(SQLite3 REQUIRED)

# --- usearch (header-only ANN index) ---------------------------------------
# Fetched via FetchContent rather than assuming a system/Homebrew install
# (usearch is not packaged in Homebrew).
FetchContent_Declare(
    usearch
    GIT_REPOSITORY https://github.com/unum-cloud/usearch.git
    GIT_TAG v2.9.2
    GIT_SHALLOW TRUE
    GIT_SUBMODULES "" # header-only usage on ARM64 needs no submodules (see BUILD_PLAN notes)
)
FetchContent_GetProperties(usearch)
if(NOT usearch_POPULATED)
    # usearch's own CMakeLists.txt builds C/Python/etc. bindings we don't
    # need -- we only want its header-only C++ API, so populate the sources
    # without add_subdirectory()'ing it. CMP0169=OLD keeps the classic
    # Populate() signature usable for that (superseded by MakeAvailable(),
    # which always add_subdirectory()s when a CMakeLists.txt is present).
    cmake_policy(SET CMP0169 OLD)
    FetchContent_Populate(usearch)

    # v2.9.2 has a genuine upstream bug that fails to compile under
    # AppleClang: see cmake/patch_usearch_candidates_iterator_bug.cmake.
    set(USEARCH_SOURCE_DIR "${usearch_SOURCE_DIR}")
    include(cmake/patch_usearch_candidates_iterator_bug.cmake)
endif()

add_library(usearch INTERFACE)
add_library(usearch::usearch ALIAS usearch)
target_include_directories(usearch SYSTEM INTERFACE "${usearch_SOURCE_DIR}/include")

# --- ONNX Runtime (built-in embedding backend) ---------------------------
# The real embed()/add_text()/search_text() backend runs a local ONNX
# sentence-embedding model (all-MiniLM-L6-v2 by default). ONNX Runtime is
# NOT header-only and NOT fetchable -- it must be installed
# (`brew install onnxruntime` on this macOS/ARM64 setup). The option
# defaults ON but degrades gracefully: if the library isn't found the
# engine still builds with only the dependency-free mock backend, and any
# attempt to load a .onnx model reports that clearly at runtime.
option(RETRIEVAL_ENGINE_WITH_ONNX "Build the ONNX Runtime embedding backend" ON)

if(RETRIEVAL_ENGINE_WITH_ONNX)
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        PATHS /opt/homebrew/include /opt/homebrew/include/onnxruntime /usr/local/include
              /usr/local/include/onnxruntime /usr/include /usr/include/onnxruntime)
    find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime
        PATHS /opt/homebrew/lib /usr/local/lib /usr/lib)

    if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
        add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")
        message(STATUS "ONNX Runtime embedding backend: ON (${ONNXRUNTIME_LIBRARY})")
    else()
        message(WARNING
            "ONNX Runtime not found (looked for onnxruntime_cxx_api.h and libonnxruntime); "
            "building with the mock embedding backend only. Install it "
            "(`brew install onnxruntime`) and reconfigure to enable the real backend.")
        set(RETRIEVAL_ENGINE_WITH_ONNX OFF)
    endif()
endif()

# --- GoogleTest (test framework) -------------------------------------------
if(RETRIEVAL_ENGINE_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.18.0
        GIT_SHALLOW TRUE
    )
    # Match the parent project's runtime library on Windows; irrelevant on
    # macOS but harmless to set.
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    include(GoogleTest)
endif()

add_subdirectory(core)

# Native benchmark harness (benchmarks/). ON alongside the C++ tests by
# default; the Python packaging build turns both OFF (see pyproject.toml).
option(RETRIEVAL_ENGINE_BUILD_BENCHMARKS
    "Build the native retrieval micro-benchmark (benchmarks/run_benchmarks)"
    ${RETRIEVAL_ENGINE_BUILD_TESTS})
if(RETRIEVAL_ENGINE_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(RETRIEVAL_ENGINE_BUILD_PYTHON_BINDINGS)
    add_subdirectory(bindings)
endif()
