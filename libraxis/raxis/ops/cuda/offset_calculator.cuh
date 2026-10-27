#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

#include <raxis/ops/cuda/integer_divider.cuh>
#include <raxis/tensor/tensor_iterator.h>
#include <raxis/util/exception.h>


namespace raxis::cuda {

// From: aten/src/ATen/cuda/detail/OffsetCalculator.cuh
constexpr int kMaxDims = 25;

/// Turns a linear element index into the byte offset of that element in
/// each operand, for operands that share a shape but not their strides
/// (broadcast inputs have stride 0, permuted ones any order). Dim 0 of a
/// TensorIterator is the fastest-moving one, so the index is decomposed from
/// dim 0 up: idx % size0 is the position in dim 0, idx / size0 carries on.
///
/// Strides are signed (PyTorch makes that optional): flip walks its output
/// backwards with negative strides. The offsets fit in int32 because
/// gpu_kernel splits any iterator that would not.
template <int NARGS>
struct OffsetCalculator {
    // A zero-sized array is not allowed in device code: nullary ops (fill)
    // keep a size-1 placeholder
    using offset_type = std::array<int32_t, std::max(NARGS, 1)>;

    OffsetCalculator(int dims, const int64_t* sizes, const int64_t* const* strides) : dims(dims) {
        RAXIS_CHECK(dims <= kMaxDims, "tensor has too many (>", kMaxDims, ") dims");
        for (int i = 0; i < dims; i++) {
            sizes_[i] = IntDivider(static_cast<uint32_t>(sizes[i]));
            for (int arg = 0; arg < NARGS; arg++) {
                strides_[i][arg] = static_cast<int32_t>(strides[arg][i]);
            }
        }
    }

    __host__ __device__ offset_type get(uint32_t linear_idx) const {
        offset_type offsets;
        #pragma unroll
        for (int arg = 0; arg < NARGS; arg++) {
            offsets[arg] = 0;
        }

        // A loop bound known at compile time, so the compiler can unroll
        // it and keep sizes_ / strides_ in registers
        #pragma unroll
        for (int dim = 0; dim < kMaxDims; ++dim) {
            if (dim == dims) {
                break;
            }
            auto divmod = sizes_[dim].divmod(linear_idx);
            linear_idx = divmod.div;
            #pragma unroll
            for (int arg = 0; arg < NARGS; arg++) {
                offsets[arg] += static_cast<int32_t>(divmod.mod) * strides_[dim][arg];
            }
        }
        return offsets;
    }

    int dims;
    IntDivider sizes_[kMaxDims];
    int32_t strides_[kMaxDims][std::max(NARGS, 1)];
};

/// Byte offsets of the first N operands of `iter`
template <int N>
OffsetCalculator<N> make_offset_calculator(const TensorIterator& iter) {
    RAXIS_INTERNAL_ASSERT(N <= iter.num_operands());
    std::array<const int64_t*, std::max(N, 1)> strides;
    for (int i = 0; i < N; i++) {
        strides[i] = iter.strides(i).data();
    }
    return OffsetCalculator<N>(iter.ndim(), iter.shape().data(), strides.data());
}

} // namespace raxis::cuda
