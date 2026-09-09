# Patches a genuine upstream bug in usearch v2.9.2 (present in every tagged
# release we checked back to at least v2.8.15): index.hpp's nested
# `candidates_iterator_t::key()` dereferences a reference member (`index_gt
# const& index_`) with `->` instead of `.`, which AppleClang rejects as a
# hard compile error ("member reference type ... is not a pointer").
#
# Confirmed against upstream's unreleased `main` branch (commit f91fe5b,
# VERSION 2.26.2) that this is exactly how it was later fixed there --
# `index_.node_at_(slot()).key()`. This script applies that same one-line
# fix to our vendored v2.9.2 copy so we can stay pinned to a citable tagged
# release instead of an unreleased commit. Drop this patch once a usearch
# release ships with the fix.
#
# Expected variable: USEARCH_SOURCE_DIR (the populated FetchContent source
# tree root).

set(header_path "${USEARCH_SOURCE_DIR}/include/usearch/index.hpp")

if(NOT EXISTS "${header_path}")
    message(FATAL_ERROR "patch_usearch_candidates_iterator_bug: expected header not found at ${header_path}")
endif()

file(READ "${header_path}" header_contents)

set(buggy_line "vector_key_t key() const noexcept { return index_->node_at_(slot()).key(); }")
set(fixed_line "vector_key_t key() const noexcept { return index_.node_at_(slot()).key(); }")

string(FIND "${header_contents}" "${buggy_line}" buggy_match_index)
if(NOT buggy_match_index EQUAL -1)
    string(REPLACE "${buggy_line}" "${fixed_line}" patched_contents "${header_contents}")
    file(WRITE "${header_path}" "${patched_contents}")
    message(STATUS "patch_usearch_candidates_iterator_bug: patched candidates_iterator_t::key() in ${header_path}")
    return()
endif()

# `usearch_POPULATED` (the guard around calling this script) is an in-memory
# FetchContent property that does NOT survive across separate `cmake`
# invocations -- e.g. the internal reconfigure `cmake --build` triggers when
# a CMakeLists.txt changes. On such a re-run, FetchContent_Populate() is a
# no-op (the source dir already exists) but our guard still re-enters this
# script against an already-patched file. Treat "already fixed" as success
# so the patch step is idempotent.
string(FIND "${header_contents}" "${fixed_line}" fixed_match_index)
if(NOT fixed_match_index EQUAL -1)
    message(STATUS "patch_usearch_candidates_iterator_bug: already patched, nothing to do")
    return()
endif()

message(FATAL_ERROR
    "patch_usearch_candidates_iterator_bug: neither the buggy nor the fixed line was found in "
    "${header_path} -- usearch may have changed in an unexpected way. Re-check whether this "
    "patch is still needed and update or remove it.")
