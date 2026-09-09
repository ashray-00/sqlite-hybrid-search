add_executable(test_infra_sanity test_infra_sanity.cpp)

target_link_libraries(test_infra_sanity PRIVATE
    retrieval_engine::core
    GTest::gtest_main
)

gtest_discover_tests(test_infra_sanity DISCOVERY_TIMEOUT 60)

add_executable(test_chunking_and_dense_retrieval test_chunking_and_dense_retrieval.cpp)

target_link_libraries(test_chunking_and_dense_retrieval PRIVATE
    retrieval_engine::core
    GTest::gtest_main
)

gtest_discover_tests(test_chunking_and_dense_retrieval DISCOVERY_TIMEOUT 60)

add_executable(test_hybrid_retrieval test_hybrid_retrieval.cpp)

target_link_libraries(test_hybrid_retrieval PRIVATE
    retrieval_engine::core
    GTest::gtest_main
)

gtest_discover_tests(test_hybrid_retrieval DISCOVERY_TIMEOUT 60)

add_executable(test_recency_decay_and_temporal_reranking test_recency_decay_and_temporal_reranking.cpp)

target_link_libraries(test_recency_decay_and_temporal_reranking PRIVATE
    retrieval_engine::core
    GTest::gtest_main
)

gtest_discover_tests(test_recency_decay_and_temporal_reranking DISCOVERY_TIMEOUT 60)

add_executable(test_builtin_embedder test_builtin_embedder.cpp)

target_link_libraries(test_builtin_embedder PRIVATE
    retrieval_engine::core
    GTest::gtest_main
)

gtest_discover_tests(test_builtin_embedder DISCOVERY_TIMEOUT 60)

add_executable(test_onnx_embedder test_onnx_embedder.cpp)

target_link_libraries(test_onnx_embedder PRIVATE
    retrieval_engine::core
    GTest::gtest_main
)

gtest_discover_tests(test_onnx_embedder DISCOVERY_TIMEOUT 60)

add_executable(test_persistence test_persistence.cpp)

target_link_libraries(test_persistence PRIVATE
    retrieval_engine::core
    GTest::gtest_main
)

gtest_discover_tests(test_persistence DISCOVERY_TIMEOUT 60)
