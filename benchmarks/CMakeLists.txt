# Native retrieval micro-benchmark (latency + memory), a cross-check on the
# Python-driven benchmarks/run_eval.py. Not a test target: it is built when
# RETRIEVAL_ENGINE_BUILD_BENCHMARKS is ON (default: mirrors
# RETRIEVAL_ENGINE_BUILD_TESTS) and run manually or via run_eval.py.
add_executable(run_benchmarks run_benchmarks.cpp)

target_link_libraries(run_benchmarks PRIVATE sqlite_hybrid_search::core)
target_compile_features(run_benchmarks PRIVATE cxx_std_17)
