#pragma once

#include <array>

#include <raxis/array_types.hpp>
#include <raxis/core/utils/range.hpp>


namespace raxis {

struct DimCounter {
    DimCounter(const IntArray& shape, Range range);

    void increment(const std::array<int64_t, 2>& step);
    bool is_done() const;
    std::array<int64_t, 2> max_2d_step() const;

    const IntArray& shape;
    Range range;
    std::vector<int64_t> values;
    int64_t offset;
};

}