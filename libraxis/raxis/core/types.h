#pragma once

#include <vector>
#include <ostream>

#include <raxis/util/array_ref.h>
#include <raxis/util/small_vector.h>
#include <raxis/util/small_buffer.h>


namespace raxis {

using IntArrayRef = ArrayRef<int64_t>;
using DimVector = SmallVector<int64_t, 5>;

// ------------------------------------------------------------------

template <typename T>
struct is_printable_container : std::false_type {};

template <typename T, size_t N>
struct is_printable_container<SmallVector<T, N>> : std::true_type {};

template <typename T, size_t N>
struct is_printable_container<SmallBuffer<T, N>> : std::true_type {};

template <typename T>
struct is_printable_container<ArrayRef<T>> : std::true_type {};

template <typename Container,
    std::enable_if_t<is_printable_container<Container>::value, int> = 0>
std::ostream& operator<<(std::ostream& os, const Container& c) {
    os << '[';
    bool first = true;
    for (const auto& v : c) {
        if (!first) os << ", ";
        os << v;
        first = false;
    }
    return os << ']';
}
    
}