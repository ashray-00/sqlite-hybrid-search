add_executable(test_stage0 test_stage0.cpp)

target_link_libraries(test_stage0 PRIVATE
    retrieval_engine::core
    GTest::gtest_main
)

gtest_discover_tests(test_stage0)
