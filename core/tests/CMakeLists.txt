add_executable(test_infra_sanity test_infra_sanity.cpp)

target_link_libraries(test_infra_sanity PRIVATE
    retrieval_engine::core
    GTest::gtest_main
)

gtest_discover_tests(test_infra_sanity)

add_executable(test_stage1 test_stage1.cpp)

target_link_libraries(test_stage1 PRIVATE
    retrieval_engine::core
    GTest::gtest_main
)

gtest_discover_tests(test_stage1)

add_executable(test_stage2 test_stage2.cpp)

target_link_libraries(test_stage2 PRIVATE
    retrieval_engine::core
    GTest::gtest_main
)

gtest_discover_tests(test_stage2)
