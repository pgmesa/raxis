cmake_minimum_required(VERSION 3.24)
project(raxis_cuda_benchmarks CUDA)

set(CMAKE_CUDA_STANDARD 17)
# The GPU of the machine that builds it
set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(latency latency.cu)
