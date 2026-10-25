#include <raxis/ops/range_factories.h>

#include <cmath>
#include <cstdint>
#include <limits>

#include <raxis/util/exception.h>

#include <raxis/tensor/factories.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

namespace {

// From: aten/src/ATen/native/RangeUtils.h (compute_arange_size)
// The size is computed in double whatever the dtype, so it is the same on
// every device; int64 bounds get the exact integer formula instead, since
// a double cannot hold every int64
int64_t compute_arange_size(Scalar start, Scalar end, Scalar step, Dtype dtype) {
    const double dstart = start.to<double>();
    const double dend = end.to<double>();
    const double dstep = step.to<double>();
    RAXIS_CHECK(dstep > 0 || dstep < 0, "step must be nonzero");
    RAXIS_CHECK(std::isfinite(dstart) && std::isfinite(dend), "unsupported range: ", dstart, " -> ", dend);
    RAXIS_CHECK((dstep > 0 && dend >= dstart) || (dstep < 0 && dend <= dstart),
        "upper bound and lower bound inconsistent with step sign");

    double size_d;
    if (dtype == dtypes::Int64 && !start.is_floating() && !end.is_floating() && !step.is_floating()) {
        const int64_t istart = start.to<int64_t>();
        const int64_t iend = end.to<int64_t>();
        const int64_t istep = step.to<int64_t>();
        const int64_t sgn = (istep > 0) - (istep < 0);
        // Integer division: adding step - sgn first turns it into a ceil
        size_d = static_cast<double>((iend - istart + istep - sgn) / istep);
    } else {
        size_d = std::ceil((dend - dstart) / dstep);
    }
    RAXIS_CHECK(size_d >= 0 && size_d <= static_cast<double>(std::numeric_limits<int64_t>::max()),
        "invalid size, possible overflow?");
    return static_cast<int64_t>(size_d);
}

} // namespace


namespace ops {

// From: aten/src/ATen/native/TensorFactories.cpp (arange) and
// RangeFactories.cpp (arange_out)
Tensor arange(Scalar start, Scalar end, Scalar step, TensorOptions options) {
    if (!options.has_dtype()) {
        // Bool counts as integral, as in torch
        const bool integral = !start.is_floating() && !end.is_floating() && !step.is_floating();
        options = options.dtype(integral ? dtypes::Int64 : default_float_dtype());
    }
    const int64_t size = compute_arange_size(start, end, step, options.dtype());
    Tensor result = empty({size}, options);
    if (size > 0) {
        arange_stub(DeviceType::CPU, result, start, step);
    }
    return result;
}

} // namespace ops

Tensor arange(Scalar start, Scalar end, Scalar step, TensorOptions options) {
    return ops::arange(start, end, step, options);
}

Tensor arange(Scalar end, TensorOptions options) {
    return ops::arange(0, end, 1, options);
}

DEFINE_DISPATCH(arange_stub);

} // namespace raxis
