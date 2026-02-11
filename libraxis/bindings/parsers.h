#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <raxis/raxis.h>


namespace py = pybind11;

#if defined(_MSC_VER) && !defined(ssize_t)
typedef std::ptrdiff_t ssize_t;
#endif


py::dtype numpy_dtype(const raxis::Dtype& dt) {

    switch (dt.kind()) {
        case raxis::DtypeKind::Float32: return py::dtype::of<float>();
        case raxis::DtypeKind::Float64: return py::dtype::of<double>();
        case raxis::DtypeKind::Int32:   return py::dtype::of<int32_t>();
        case raxis::DtypeKind::Int64:   return py::dtype::of<int64_t>();
        case raxis::DtypeKind::Uint8:   return py::dtype::of<uint8_t>();
        case raxis::DtypeKind::Uint16:  return py::dtype::of<uint16_t>();
        case raxis::DtypeKind::Bool:    return py::dtype::of<bool>();
        default:
            throw std::runtime_error("Unsupported dtype for numpy()");
    }
}

py::array tensor_to_numpy(py::object self, const raxis::Tensor& a) {
    auto& shape_ = a.shape();
    std::vector<ssize_t> shape(shape_.begin(), shape_.end());
    auto strides_ = a.byte_strides();
    std::vector<ssize_t> strides(strides_.begin(), strides_.end());

    return py::array(
        numpy_dtype(a.dtype()),    // dtype
        shape,                     // shape
        strides,                   // strides (bytes!)
        a.data_ptr(),              // data pointer
        self                       // base object (keeps Tensor alive)
    );
}