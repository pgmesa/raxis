
#include <string>
#include <cstdint>
#include <iostream>
#include <chrono>

#include <raxis/raxis.hpp>

#include <spdlog/spdlog.h>
#include <raxis/format/array_fmt.hpp>

using raxis::NDArray;
namespace dtypes = raxis::dtypes;


struct Config {
    raxis::Dtype dtype1 = dtypes::Float32;  // defaults
    raxis::Dtype dtype2 = dtypes::Float32;
};

Config parse_args(int argc, char** argv) {
    Config cfg;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--dtype1" && i + 1 < argc) {
            cfg.dtype1 = raxis::parse_dtype(argv[++i]);
        }
        else if (arg == "--dtype2" && i + 1 < argc) {
            cfg.dtype2 = raxis::parse_dtype(argv[++i]);
        }
        else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  --dtype1 <type>   Dtype for arr1 (default: float32)\n"
                      << "  --dtype2 <type>   Dtype for arr2 (default: float32)\n"
                      << "Valid types: bool, uint8, uint16, int32, int64, float32, float64\n";
            std::exit(0);
        }
        else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }
    
    return cfg;
}


int main(int argc, char** argv) {
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

    // raxis::set_default_dtype(dtypes::Int64);

    auto cfg = parse_args(argc, argv);
    
    NDArray arr1 = raxis::full({400, 400}, 3, cfg.dtype1);
    NDArray arr2 = raxis::full({400, 400}, 5, cfg.dtype2);

    spdlog::info("arr1: {}\n", arr1);
    spdlog::info("arr2: {}\n", arr2);

    NDArray arr_from_vec = raxis::array<uint8_t>({1, 2, 3, 4}, {2,2}, dtypes::Uint8);
    spdlog::info("arr_from_vec: {}\n", arr_from_vec);

    auto t0 = std::chrono::high_resolution_clock::now();
    
    NDArray arr3 = arr1 + arr2;

    auto tf = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration<double, std::milli>(tf - t0);

    spdlog::info("Elapsed: {:.2f} ms\n", elapsed.count());
    spdlog::info("result: {}\n", arr3);

    return 0;
}