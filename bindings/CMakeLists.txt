# Builds the nanobind Python extension module. Only pulled in when
# RETRIEVAL_ENGINE_BUILD_PYTHON_BINDINGS is ON (see the root
# CMakeLists.txt) -- not part of the plain C++ dev/test build, so
# `cmake -B build && cmake --build build` is completely unaffected.

# Standard nanobind CMake discovery: ask the active Python interpreter
# (found via find_package(Python...)) where its installed `nanobind` package
# keeps its CMake config. Under scikit-build-core's isolated build (see
# pyproject.toml's [build-system] requires), that's the isolated build
# environment nanobind was installed into for this build.
find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE nanobind_ROOT
)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_retrieval_engine_ext NB_STATIC python_bindings.cpp)
target_link_libraries(_retrieval_engine_ext PRIVATE retrieval_engine::core)

# Installed alongside the pure-Python package (see pyproject.toml's
# tool.scikit-build.wheel.packages) so that, once installed,
# `retrieval_engine/_retrieval_engine_ext*` sits next to
# `retrieval_engine/__init__.py` and `from ._retrieval_engine_ext import ...`
# resolves as a sibling submodule.
install(TARGETS _retrieval_engine_ext LIBRARY DESTINATION retrieval_engine)
