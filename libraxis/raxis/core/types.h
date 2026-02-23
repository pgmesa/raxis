#pragma once

#include <vector>
#include <ostream>


namespace raxis {

using IntArray = std::vector<int64_t>;

using index_t = int64_t;
using IndexArray = std::vector<index_t>;


inline std::ostream& operator<<(std::ostream& os, const IntArray& v) {
    os << '[';
    for (size_t i = 0; i < v.size(); ++i) {
        os << v[i];
        if (i + 1 < v.size()) os << ", ";
    }
    return os << ']';
}
    
}