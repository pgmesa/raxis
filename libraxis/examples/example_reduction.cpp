#include <string>
#include <cstdint>
#include <iostream>
#include <chrono>

#include <raxis/raxis.h>


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
        auto cfg = parse_args(argc, argv);

        Tensor t1 = raxis::full({400, 400}, 3, cfg.dtype1);

        std::cout << "t1: " << t1.to_string() << "\n";

        auto t0 = std::chrono::high_resolution_clock::now();
        Tensor t3 = t1.sum();
        auto tf = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration<double, std::milli>(tf - t0);

        std::cout << "Elapsed: " << elapsed.count() << " ms\n";
        std::cout << "result: " << t3.to_string() << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}