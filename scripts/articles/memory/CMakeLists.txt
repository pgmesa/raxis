cmake_minimum_required(VERSION 3.22)
project(raxis_memory_benchmarks CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(memory_levels memory_levels.cpp)
