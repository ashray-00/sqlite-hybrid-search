add_executable(test_infra_sanity test_infra_sanity.cpp)

target_link_libraries(test_infra_sanity PRIVATE
    retrieval_engine::core
    GTest::gtest_main
)

gtest_discover_tests(test_infra_sanity)

add_executable(test_chunking_and_dense_retrieval test_chunking_and_dense_retrieval.cpp)

target_link_libraries(test_chunking_and_dense_retrieval PRIVATE
    retrieval_engine::core
    GTest::gtest_main
)

gtest_discover_tests(test_chunking_and_dense_retrieval)

add_executable(test_hybrid_retrieval test_hybrid_retrieval.cpp)

target_link_libraries(test_hybrid_retrieval PRIVATE
    retrieval_engine::core
    GTest::gtest_main
)

gtest_discover_tests(test_hybrid_retrieval)
