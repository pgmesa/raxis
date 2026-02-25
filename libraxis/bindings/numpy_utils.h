#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <raxis/raxis.h>


namespace py = pybind11;

#if defined(_MSC_VER) && !defined(ssize_t)
typedef std::ptrdiff_t ssize_t;
#endif

namespace raxis::python {

py::dtype numpy_dtype(const Dtype& dt) {

    switch (dt.kind()) {
        case DtypeKind::Float32: return py::dtype::of<float>();
        case DtypeKind::Float64: return py::dtype::of<double>();
        case DtypeKind::Int32:   return py::dtype::of<int32_t>();
        case DtypeKind::Int64:   return py::dtype::of<int64_t>();
        case DtypeKind::Uint8:   return py::dtype::of<uint8_t>();
        case DtypeKind::Uint16:  return py::dtype::of<uint16_t>();
        case DtypeKind::Bool:    return py::dtype::of<bool>();
        default:
            throw std::runtime_error("Unsupported dtype for numpy()");
    }
}

inline Dtype dtype_from_numpy(py::dtype dt) {
    if (dt.is(py::dtype::of<float>()))    return dtypes::Float32;
    if (dt.is(py::dtype::of<double>()))   return dtypes::Float64;
    if (dt.is(py::dtype::of<int32_t>()))  return dtypes::Int32;
    if (dt.is(py::dtype::of<int64_t>()))  return dtypes::Int64;
    if (dt.is(py::dtype::of<uint8_t>()))  return dtypes::Uint8;
    if (dt.is(py::dtype::of<uint16_t>())) return dtypes::Uint16;
    if (dt.is(py::dtype::of<bool>()))     return dtypes::Bool;
    throw py::type_error("unsupported numpy dtype: " + py::str(dt).cast<std::string>());
}

Tensor tensor_from_numpy(py::array arr) {
    arr = py::array::ensure(arr, py::array::c_style);
        
    IntArray shape(arr.ndim());
    for (int i = 0; i < arr.ndim(); ++i)
        shape[i] = arr.shape(i);
    
    Dtype dtype = dtype_from_numpy(arr.dtype());
    
    // Hold reference: prevent GC from collecting the numpy array
    auto* ref = new py::object(arr);

    return from_blob(
        arr.mutable_data(),
        ref,
        [](void* ctx) {
            delete static_cast<py::object*>(ctx);
        },
        shape,
        dtype
    );
}

py::array tensor_to_numpy(py::object self, const Tensor& a) {
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

}
