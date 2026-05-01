#pragma once

#include <cstddef>
#include <cstdint>

#include <raxis/util/function_traits.h>


namespace raxis {

// Check if all operands (output + inputs) are contiguous in the inner dimension.
// "Contiguous" means stride == sizeof(result_type) for all operands.
template <typename traits>
inline bool is_contiguous(const int64_t* strides) {
    using result_type = typename traits::result_type;
    constexpr int ntensors = traits::arity + 1;
    for (int i = 0; i < ntensors; i++) {
        if (strides[i] != sizeof(result_type)) {
            return false;
        }
    }
    return true;
}

// Check if all operands are contiguous except operand S, which has stride 0
// (i.e., it's a scalar broadcast). S is 1-indexed from the full operand list
// (0 = output, 1 = first input, etc.)
template <typename traits, size_t S>
inline bool is_contiguous_scalar(const int64_t* strides) {
    using result_type = typename traits::result_type;
    constexpr int ntensors = traits::arity + 1;
    for (int i = 0; i < ntensors; i++) {
        if (i == static_cast<int>(S)) {
            if (strides[i] != 0) {
                return false;
            }
        } else {
            if (strides[i] != sizeof(result_type)) {
                return false;
            }
        }
    }
    return true;
}

} // namespace raxis