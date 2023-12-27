cmake_minimum_required(VERSION 3.24)
project(MyMatrix)

set(CMAKE_CXX_STANDARD 17)
if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /utf-8 /arch:AVX2 /Qpar /Qpar-report:1")
    add_link_options(/LTCG)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS false)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
endif()

option(BUILD_GMOCK OFF)
include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
    nanobench
    GIT_REPOSITORY https://github.com/martinus/nanobench.git
    GIT_TAG v4.1.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(nanobench)

add_library(MyMatrix INTERFACE)
target_include_directories(MyMatrix INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_sources(MyMatrix INTERFACE MatMul.cpp)


add_executable(MyMatrix-tests MyMatrix-tests.cpp)
target_link_libraries(MyMatrix-tests GTest::gtest_main MyMatrix)
enable_testing()
include(GoogleTest)
gtest_discover_tests(MyMatrix-tests)

add_executable(MyMatrix-bench MyMatrix-bench.cpp)
target_link_libraries(MyMatrix-bench PRIVATE nanobench MyMatrix)

include(cython.cmake)
