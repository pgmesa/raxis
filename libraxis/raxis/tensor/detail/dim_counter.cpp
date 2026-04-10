
#include <raxis/tensor/detail/dim_counter.h>

#include <cassert>

#include <raxis/core/types.h>


namespace raxis::detail {

DimCounter::DimCounter(IntArrayRef shape, Range range)
  : shape(shape), 
    range(range), 
    values(shape.size()), 
    offset(range.begin) 
{
    std::fill(values.begin(), values.end(), 0);
    if (range.begin == 0) {
        return;
    }

    int64_t linear_offset = range.begin;
    size_t ndim = values.size();
    for (size_t dim = 0; dim < ndim; dim++) {
        int64_t size = shape[dim];
        if (size > 0) {
            values[dim] = linear_offset % size;
            linear_offset /= size;
        }
    }
}

bool DimCounter::is_done() const {
    return offset >= range.end;
}

void DimCounter::increment(const std::array<int64_t, 2>& step) {
    offset += step[0] * step[1];
    auto ndim = values.size();
    int64_t overflow = step[0];
    size_t i = 0;
    if (step[1] != 1) {
        i = 1;
        overflow = step[1];
    }
    for (; i < ndim && overflow > 0; i++) {
        auto size = shape[i];
        auto prev = values[i];
        auto value = prev + overflow;
        if (value >= size) {
            overflow = 1;
            value -= size;
        } else {
            overflow = 0;
        }
        values[i] = static_cast<int64_t>(value);
    }
}

std::array<int64_t, 2> DimCounter::max_2d_step() const {
    int64_t step0 = std::min(shape[0] - values[0], range.end - offset);
    int64_t step1 = 1;
    if (step0 == shape[0] && !shape.empty()) {
        step1 = std::min(shape[1] - values[1], (range.end - offset) / shape[0]);
    }
    return {step0, step1};
}

}

