cmake_minimum_required(VERSION 3.22)
project(raxis_reduction_benchmarks CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# AVX2 for the vector loops, and no floating-point contraction or
# reassociation: each loop adds in exactly the order it is written
add_executable(sum_loops sum_loops.cpp)
if(MSVC)
    target_compile_options(sum_loops PRIVATE /arch:AVX2 /fp:precise)
else()
    target_compile_options(sum_loops PRIVATE -mavx2 -ffp-contract=off)
endif()
