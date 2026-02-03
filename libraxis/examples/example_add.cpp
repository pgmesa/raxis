
#include <string>
#include <cstdint>
#include <iostream>
#include <chrono>

#include <raxis/raxis.hpp>

#include <spdlog/spdlog.h>
#include <raxis/format/array_fmt.hpp>

using raxis::NDArray;
namespace dtypes = raxis::dtypes;


int main() {
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

    raxis::set_default_dtype(dtypes::Int64);
    NDArray arr1 = raxis::full({400, 400}, 3);
    NDArray arr2 = raxis::full({400, 400}, 5, dtypes::Float32);

    spdlog::info("arr1: {}\n", arr1);
    spdlog::info("arr2: {}\n", arr2);

    NDArray arr_from_vec = raxis::array<uint8_t>({1, 2, 3, 4}, {2,2}, dtypes::Uint8);
    spdlog::info("arr_from_vec: {}\n", arr_from_vec);

    auto t0 = std::chrono::high_resolution_clock::now();
    
    NDArray arr3 = arr1 + arr2;

    auto tf = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration<double, std::milli>(tf - t0);

    spdlog::info("Elapsed: {:.2f} ms\n", elapsed.count());

    return 0;
}