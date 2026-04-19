
#include <string>
#include <cstdint>
#include <iostream>
#include <chrono>

#include <raxis/raxis.h>
#include <raxis/format.h>

#include <spdlog/spdlog.h>

using raxis::Tensor;
namespace dtypes = raxis::dtypes;


struct Config {
    raxis::Dtype dtype1 = dtypes::Float32;
};

Config parse_args(int argc, char** argv) {
    Config cfg;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--dtype1" && i + 1 < argc) {
            cfg.dtype1 = raxis::parse_dtype(argv[++i]);
        }
        else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  --dtype1 <type>   Dtype for tensor (default: float32)\n"
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
    try {
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

        auto cfg = parse_args(argc, argv);
        
        Tensor t1 = raxis::full({400, 400}, 3, cfg.dtype1);

        spdlog::info("t1: {}\n", t1);

        auto t0 = std::chrono::high_resolution_clock::now();
        Tensor t3 = t1.sum();
        auto tf = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double, std::milli>(tf - t0);

        spdlog::info("Elapsed: {:.2f} ms\n", elapsed.count());
        spdlog::info("result: {}\n", t3);

    } catch (const std::exception& e) {
        spdlog::error("Exception: {}", e.what());
        return 1;
    }

    return 0;
}