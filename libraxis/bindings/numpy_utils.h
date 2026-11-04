#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <raxis/raxis.h>


namespace py = pybind11;

#if defined(_MSC_VER) && !defined(ssize_t)
typedef std::ptrdiff_t ssize_t;
#endif

namespace raxis::python {

// numpy is optional, but py::isinstance<py::array> imports it and throws when
// it is missing. If nothing has imported numpy yet, obj cannot be an ndarray
inline bool is_numpy_array(const py::handle& obj) {
    py::dict modules = py::module_::import("sys").attr("modules");
    if (!modules.contains("numpy") || modules["numpy"].is_none()) {
        return false;
    }
    return py::isinstance<py::array>(obj);
}

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

// By kind and size, not by descriptor identity: numpy has two int32 types on
// Windows (C long 'l' and int 'i'), and arithmetic on one may return the other
inline Dtype dtype_from_numpy(py::dtype dt) {
    const char kind = dt.kind();
    const auto size = dt.itemsize();
    if (dt.byteorder() != '>') {
        if (kind == 'f' && size == 4) return dtypes::Float32;
        if (kind == 'f' && size == 8) return dtypes::Float64;
        if (kind == 'i' && size == 4) return dtypes::Int32;
        if (kind == 'i' && size == 8) return dtypes::Int64;
        if (kind == 'u' && size == 1) return dtypes::Uint8;
        if (kind == 'b' && size == 1) return dtypes::Bool;
    }
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
    if (!a.is_cpu()) {
        throw py::type_error("can't convert " + a.device().str() + " device type tensor to "
            "numpy. Use Tensor.cpu() to copy the tensor to host memory first.");
    }
    IntArrayRef shape_ = a.shape();
    std::vector<ssize_t> shape(shape_.begin(), shape_.end());
    DimVector strides_ = a.byte_strides();
    std::vector<ssize_t> strides(strides_.begin(), strides_.end());

    // From: torch/csrc/utils/tensor_numpy.cpp (tensor_to_numpy): the array
    // reads this memory directly, so resize_ must not move it from now on
    a.storage()->set_resizable(false);

    return py::array(
        numpy_dtype(a.dtype()),    // dtype
        std::move(shape),                     // shape
        std::move(strides),                   // strides (bytes!)
        a.data_ptr(),              // data pointer
        self                       // base object (keeps Tensor alive)
    );
}

}
