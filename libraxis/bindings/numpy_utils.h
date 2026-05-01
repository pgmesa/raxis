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
    if (dt.is(py::dtype::of<bool>()))     return dtypes::Bool;
    throw py::type_error("unsupported numpy dtype: " + py::str(dt).cast<std::string>());
}

Tensor tensor_from_numpy(py::array arr) {
    DimVector shape(arr.ndim());
    DimVector strides(arr.ndim());
    for (int i = 0; i < arr.ndim(); ++i) {
        shape[i] = arr.shape(i);
        strides[i] = arr.strides(i) / arr.itemsize(); // byte strides -> element strides
    }
    
    Dtype dtype = dtype_from_numpy(arr.dtype());
    
    // Hold reference: prevent GC from collecting the numpy array
    auto* ref = new py::object(arr);

    return from_blob(
        arr.mutable_data(),
        ref,
        [](void* ctx) {
            delete static_cast<py::object*>(ctx);
        },
        std::move(shape), 
        std::move(strides),
        dtype
    );
}

py::array tensor_to_numpy(py::object self, const Tensor& a) {
    IntArrayRef shape_ = a.shape();
    std::vector<ssize_t> shape(shape_.begin(), shape_.end());
    DimVector strides_ = a.byte_strides();
    std::vector<ssize_t> strides(strides_.begin(), strides_.end());

    return py::array(
        numpy_dtype(a.dtype()),    // dtype
        std::move(shape),                     // shape
        std::move(strides),                   // strides (bytes!)
        a.data_ptr(),              // data pointer
        self                       // base object (keeps Tensor alive)
    );
}

}
