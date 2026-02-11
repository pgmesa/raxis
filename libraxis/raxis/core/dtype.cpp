
#include <raxis/core/dtype.h>

#include <sstream>
#include <string>


namespace raxis {

void Dtype::fill(void* data, size_t n, double value) const {
    dispatch_all_types([&](auto type_tag) {
        using T = typename decltype(type_tag)::type;
        auto* ptr = static_cast<T*>(data);
        std::fill(ptr, ptr + n, static_cast<T>(value));
    });
}

std::string Dtype::to_string(const void* data) const {
    return dispatch_all_types([&](auto type_tag) -> std::string {
        using T = typename decltype(type_tag)::type;
        const T* ptr = static_cast<const T*>(data);
        
        if constexpr (std::is_same_v<T, bool>) {
            return (*ptr) ? "true" : "false";
        } else if constexpr (std::is_floating_point_v<T>) {
            std::ostringstream oss;
            oss << *ptr;
            return oss.str();
        } else {
            return std::to_string(*ptr);
        }
    });
}

namespace detail {
    Dtype& get_default_dtype_ref() {
        static Dtype default_dtype = dtypes::Float32;
        return default_dtype;
    }
}

Dtype default_dtype() {
    return detail::get_default_dtype_ref();
}

void set_default_dtype(Dtype dtype) {
    if (!dtype.defined()) {
        throw std::invalid_argument("default dtype must be defined");
    }
    detail::get_default_dtype_ref() = dtype;
}

} // namespace raxis