cmake_minimum_required(VERSION 3.15)
project(raxis)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Library
add_library(raxis
    src/array.cpp
    src/array_utils.cpp
)

target_include_directories(raxis PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Examples
add_executable(example
    examples/example.cpp
)

target_link_libraries(example PRIVATE raxis)