
#include <raxis/util/range.h>

#include <ostream>


namespace raxis {

std::ostream& operator<<(std::ostream& out, const Range& range) {
    out << "Range[" << range.begin << ", " << range.end << "]";
    return out;
}

}