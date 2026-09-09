cmake_minimum_required(VERSION 3.24)
project(sqlite_hybrid_search LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# The static core library gets linked into the nanobind Python extension (a
# shared object), so all of its objects must be position-independent. ELF
# linkers reject non-PIC objects in a shared library; Mach-O does not, which
# is why this was invisible on macOS.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(RETRIEVAL_ENGINE_BUILD_TESTS "Build the C++ unit tests" ON)
# OFF by default so the plain C++ dev/test build (`cmake -B build &&
# cmake --build build`) is completely unaffected; turned ON by
# pyproject.toml's scikit-build-core config when building the Python
# extension (`pip install -e .`).
option(RETRIEVAL_ENGINE_BUILD_PYTHON_BINDINGS "Build the nanobind Python extension module" OFF)

# Build with a sanitizer. Empty (default) = no effect on the normal build.
# Values: thread | address | undefined | "address;undefined". GCC/Clang only.
# Used to verify the concurrency model is race-free (see docs/DECISIONS.md
# ADR-11): `cmake -B build-tsan -DRETRIEVAL_ENGINE_SANITIZE=thread`.
set(RETRIEVAL_ENGINE_SANITIZE "" CACHE STRING
    "Sanitizer to build with: thread | address | undefined | address;undefined")

if(RETRIEVAL_ENGINE_SANITIZE)
    set(_sanitize_flags "")
    foreach(_san IN LISTS RETRIEVAL_ENGINE_SANITIZE)
        if(_san STREQUAL "thread")
            list(APPEND _sanitize_flags -fsanitize=thread)
        elseif(_san STREQUAL "address")
            list(APPEND _sanitize_flags -fsanitize=address -fno-omit-frame-pointer)
        elseif(_san STREQUAL "undefined")
            list(APPEND _sanitize_flags -fsanitize=undefined -fno-sanitize-recover=all)
        else()
            message(FATAL_ERROR "RETRIEVAL_ENGINE_SANITIZE: unknown value '${_san}'")
        endif()
    endforeach()
    if("thread" IN_LIST RETRIEVAL_ENGINE_SANITIZE AND "address" IN_LIST RETRIEVAL_ENGINE_SANITIZE)
        message(FATAL_ERROR "RETRIEVAL_ENGINE_SANITIZE: 'thread' and 'address' are mutually exclusive")
    endif()
    add_compile_options(${_sanitize_flags} -g -O1)
    add_link_options(${_sanitize_flags})
    message(STATUS "Sanitizer build: ${RETRIEVAL_ENGINE_SANITIZE}")
endif()

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
    # `brew --prefix sqlite` prints a path even when the formula is not
    # installed, so only trust it if the library is actually there.
    if(HOMEBREW_SQLITE_PREFIX AND EXISTS "${HOMEBREW_SQLITE_PREFIX}/lib")
        list(APPEND CMAKE_PREFIX_PATH "${HOMEBREW_SQLITE_PREFIX}")
    endif()
endif()

find_package(SQLite3 REQUIRED)

# For std::shared_mutex / std::thread in the core library (pthreads on Linux).
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# --- usearch (header-only ANN index) ---------------------------------------
# Fetched via FetchContent rather than assuming a system/Homebrew install
# (usearch is not packaged in Homebrew).
FetchContent_Declare(
    usearch
    GIT_REPOSITORY https://github.com/unum-cloud/usearch.git
    GIT_TAG v2.9.2
    GIT_SHALLOW TRUE
    # Only the `fp16` submodule: usearch's headers include <fp16/fp16.h> on
    # targets without native _Float16 (x86_64 without AVX512 -- ARM and
    # AVX512 use the compiler's own half type). `simsimd`/`stringzilla` are
    # not referenced by the header subset this project uses.
    GIT_SUBMODULES "fp16"
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
if(EXISTS "${usearch_SOURCE_DIR}/fp16/include")
    target_include_directories(usearch SYSTEM INTERFACE "${usearch_SOURCE_DIR}/fp16/include")
endif()

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
            "building with the mock embedding backend only. To enable the real backend, install "
            "it (macOS: `brew install onnxruntime`; Linux: extract an onnxruntime-linux-* release "
            "into /usr/local, or pass -DONNXRUNTIME_INCLUDE_DIR= and -DONNXRUNTIME_LIBRARY=) and "
            "reconfigure. See README.md.")
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
