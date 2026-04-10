#pragma once

#include <array>

#include <raxis/core/types.h>
#include <raxis/util/range.h>
#include <raxis/util/small_buffer.h>


namespace raxis::detail {

struct DimCounter {
    DimCounter(IntArrayRef shape, Range range);

    void increment(const std::array<int64_t, 2>& step);
    bool is_done() const;
    std::array<int64_t, 2> max_2d_step() const;

    IntArrayRef shape;
    Range range;
    SmallBuffer<int64_t, 4> values;
    int64_t offset;
};

}