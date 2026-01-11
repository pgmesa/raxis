
#include <string>
#include <cstdint>
#include <iostream>
#include <chrono>
#include <format>

#include <raxis/array.hpp>


using raxis::Dtype;
using raxis::NDArray;


int main() {
    NDArray arr1 = raxis::full({3, 1, 5}, 3, Dtype::Float64);
    NDArray arr2 = raxis::full({1, 4, 5}, 5, Dtype::Float64);

    std::cout << std::format("arr1: {}\n", arr1);
    std::cout << std::format("arr2: {}\n", arr2);

    auto t0 = std::chrono::high_resolution_clock::now();
    
    NDArray arr3 = arr1 + arr2;

    auto tf = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0);

    std::cout << std::format("Elapsed: {}\n", elapsed);

    return 0;
}