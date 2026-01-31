#pragma once

#include <vector>
#include <functional>
#include <stdexcept>

#include <raxis/api.hpp>
#include <raxis/array.hpp>
#include <raxis/array_types.hpp>


namespace raxis {

using Deleter = std::function<void(void*)>;

RAXIS_API NDArray from_blob(
    void* data, 
    const IntArray& shape, 
    const Dtype& dtype = default_dtype()
);

RAXIS_API NDArray from_blob(
    void* data,
    const IntArray& shape,          
    const Dtype& dtype,
    detail::DeleterFnPtr deleter
);

RAXIS_API NDArray empty(
    const IntArray& shape, 
    const Dtype& dtype = default_dtype()
);

RAXIS_API NDArray full(
    const IntArray& shape, 
    double value, 
    const Dtype& dtype = default_dtype()
);

RAXIS_API NDArray ones(const IntArray& shape, const Dtype& dtype = default_dtype());
RAXIS_API NDArray zeros(const IntArray& shape, const Dtype& dtype = default_dtype());

template<typename T>
NDArray array(
    const std::vector<T>& data,
    const IntArray& shape,
    const Dtype& dtype = default_dtype()
) {
    int64_t n = compute_numel(shape);
    if (data.size() != n) {
        throw std::runtime_error("data size mismatch");
    }

    NDArray out = empty(shape, dtype);
    T* out_data = out.mutable_data_ptr<T>();
    
    std::copy(data.begin(), data.end(), out_data);
    return out;
}
}
