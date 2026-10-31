cmake_minimum_required(VERSION 3.22)
project(raxis_lineage_benchmarks CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(add_loops_base add_loops.cpp)

add_executable(add_loops_avx2 add_loops.cpp)
if (MSVC)
    target_compile_options(add_loops_avx2 PRIVATE /arch:AVX2)
else()
    target_compile_options(add_loops_avx2 PRIVATE -mavx2)
endif()

# git clone https://github.com/pgmesa/multitensor, then -DMULTITENSOR_DIR=<clone>
if (MULTITENSOR_DIR)
    set(CPPTENSOR "${MULTITENSOR_DIR}/C++/cpptensor")
    add_executable(multitensor_check multitensor_check.cpp
        "${CPPTENSOR}/tensor.cpp" "${CPPTENSOR}/dtype.cpp" "${CPPTENSOR}/utils.cpp")
    target_include_directories(multitensor_check PRIVATE "${CPPTENSOR}")
endif()
