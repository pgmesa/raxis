
#include <string>
#include <cstdint>
#include <iostream>
#include <chrono>
#include <format>

#include <raxis/array.hpp>


using raxis::Dtype;


template <>
struct std::formatter<std::vector<int64_t>> : std::formatter<int64_t> {
    auto format(const std::vector<int64_t>& v, format_context& ctx) const {
        auto out = ctx.out();
        *out++ = '[';

        for (size_t i = 0; i < v.size(); ++i) {
            out = std::formatter<int64_t>::format(v[i], ctx);
            if (i + 1 < v.size()) {
                *out++ = ',';
                *out++ = ' ';
            }
        }

        *out++ = ']';
        return out;
    }
};


int main() {
    auto arr1 = raxis::full({400, 400}, 3, Dtype::Float64);
    auto arr2 = raxis::full({1, 400}, 5, Dtype::Float64);

    std::cout << std::format("arr1: strides={}\n", arr1.strides());
    std::cout << std::format("arr2: strides={}\n", arr2.strides());

    auto t0 = std::chrono::high_resolution_clock::now();
    
    raxis::NDArray arr3 = arr1 + arr2;

    auto tf = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(tf - t0);

    std::cout << std::format("Elapsed: {}\n", elapsed);

    return 0;
}