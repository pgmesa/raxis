#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <string>
#include <vector>

#include <raxis/core/dtype.h>
#include <raxis/util/exception.h>
#include <raxis/core/scalar.h>
#include <raxis/core/types.h>
#include <raxis/tensor/tensor.h>
#include <raxis/tensor/tensor_options.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/tensor_indexing.h>


namespace py = pybind11;


namespace raxis::python {


inline Dtype dtype_from_python(py::handle obj) {
    // Already a raxis.Dtype
    if (py::isinstance<Dtype>(obj)) {
        return obj.cast<Dtype>();
    }
    // Python builtins
    if (obj.ptr() == (PyObject*)&PyBool_Type)   return dtypes::Bool;
    if (obj.ptr() == (PyObject*)&PyLong_Type)   return dtypes::Int64;
    if (obj.ptr() == (PyObject*)&PyFloat_Type)  return dtypes::Float64;
    // String
    if (py::isinstance<py::str>(obj)) {
        return parse_dtype(obj.cast<std::string>());
    }

    throw py::type_error(
        "expected a dtype, got " + py::str(obj.get_type()).cast<std::string>()
    );
}


inline Dtype py_scalar_dtype(const py::handle& obj) {
    if (py::isinstance<py::bool_>(obj)) return dtypes::Bool;
    if (py::isinstance<py::int_>(obj))  return dtypes::Int64;
    if (py::isinstance<py::float_>(obj)) return dtypes::Float64;
    RAXIS_CHECK(
        false,
        "tensor(): unsupported scalar type: ",
        py::str(obj.get_type()).cast<std::string>()
    );
    return dtypes::Undefined; // unreachable
}

// ============================================================================
// Dtype promotion for mixed-type lists
// ============================================================================
//
// When a list contains mixed types (e.g. [1, 2.0, 3]), we promote:
//   Bool + Int   → Int
//   Bool + Float → Float
//   Int  + Float → Float

inline Dtype promote_py_dtypes(Dtype a, Dtype b) {
    if (a == b) return a;
    return promote_types(a, b);
}

// ============================================================================
// Recursive structure parsing
// ============================================================================
//
// Three things are extracted from the Python object:
//   1. shape   - by measuring list lengths at each nesting level
//   2. dtype   - from leaf scalar types (promoted if mixed)
//   3. scalars - leaf values flattened in row-major (C-contiguous) order

/// Infer the shape of a (potentially nested) Python list/tuple.
/// Validates that the structure is rectangular (not ragged).
inline void infer_shape(const py::handle& obj, IntArray& shape, int64_t depth) {
    if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
        auto seq = py::reinterpret_borrow<py::sequence>(obj);
        auto len = static_cast<int64_t>(seq.size());

        if (depth == static_cast<int64_t>(shape.size())) {
            // First time seeing this depth — record the size
            shape.push_back(len);
        } else {
            RAXIS_CHECK(
                shape[depth] == len,
                "tensor(): expected size ", shape[depth],
                " at dimension ", depth,
                ", but got ", len, " (input is not rectangular)"
            );
        }

        for (int64_t i = 0; i < len; ++i) {
            infer_shape(seq[i], shape, depth + 1);
        }
    }
    // else: scalar leaf - contributes no dimensions
}

/// Recursively flatten leaf scalars into a vector, and infer/promote dtype.
inline void flatten_py_data(
    const py::handle& obj,
    std::vector<Scalar>& out,
    Dtype& dtype,
    bool& dtype_set
) {
    if (py::isinstance<py::list>(obj) || py::isinstance<py::tuple>(obj)) {
        auto seq = py::reinterpret_borrow<py::sequence>(obj);
        for (size_t i = 0; i < seq.size(); ++i) {
            flatten_py_data(seq[i], out, dtype, dtype_set);
        }
    } else {
        // Leaf scalar
        Dtype elem_dtype = py_scalar_dtype(obj);

        if (!dtype_set) {
            dtype = elem_dtype;
            dtype_set = true;
        } else {
            dtype = promote_py_dtypes(dtype, elem_dtype);
        }

        // Extract the value.
        // Check bool before int because Python bool is a subclass of int.
        if (py::isinstance<py::bool_>(obj)) {
            out.emplace_back(obj.cast<bool>());
        } else if (py::isinstance<py::int_>(obj)) {
            out.emplace_back(obj.cast<int64_t>());
        } else if (py::isinstance<py::float_>(obj)) {
            out.emplace_back(obj.cast<double>());
        }
    }
}

inline Tensor tensor_from_python(
    const py::object& data,
    std::optional<Dtype> dtype_opt,
    std::optional<Device> device_opt
) {
    bool is_scalar_input = !py::isinstance<py::list>(data) 
                        && !py::isinstance<py::tuple>(data);

    // Infer shape
    IntArray shape;
    if (!is_scalar_input) {
        infer_shape(data, shape, 0);
    }

    // Flatten data + infer dtype
    std::vector<Scalar> flat;
    Dtype inferred_dtype = dtypes::Undefined;
    bool dtype_set = false;

    if (is_scalar_input) {
        inferred_dtype = py_scalar_dtype(data);
        dtype_set = true;

        if (py::isinstance<py::bool_>(data)) {
            flat.emplace_back(data.cast<bool>());
        } else if (py::isinstance<py::int_>(data)) {
            flat.emplace_back(data.cast<int64_t>());
        } else if (py::isinstance<py::float_>(data)) {
            flat.emplace_back(data.cast<double>());
        }
    } else {
        // Compute total elements for reserve
        int64_t numel = 1;
        for (auto s : shape) numel *= s;
        flat.reserve(static_cast<size_t>(numel));

        flatten_py_data(data, flat, inferred_dtype, dtype_set);
    }

    // Handle empty tensor: tensor([]) → shape {0}, use default dtype
    if (!dtype_set) {
        inferred_dtype = default_dtype();
    }

    // Resolve target dtype
    // If user didn't specify, apply the same promotion rules as C++ path
    Dtype target_dtype = dtype_opt.value_or(compute_desired_dtype(inferred_dtype));

    // Build TensorOptions and create tensor
    TensorOptions options = TensorOptions()
        .dtype(target_dtype)
        .device(device_opt.value_or(DeviceType::CPU));

    return tensor_from_scalars(flat, shape, options);
}


std::vector<indexing::TensorIndex> py_index_to_tensor_indices(py::handle idx) {
    std::vector<indexing::TensorIndex> result;
    
    auto convert_one = [&](py::handle obj) {
        if (obj.is_none()) {
            result.emplace_back(indexing::None);
        } else if (py::isinstance<py::ellipsis>(obj)) {
            result.emplace_back(indexing::Ellipsis);
        } else if (py::isinstance<py::bool_>(obj)) {
            result.emplace_back(obj.cast<bool>());
        } else if (py::isinstance<py::int_>(obj)) {
            result.emplace_back(obj.cast<int64_t>());
        } else if (py::isinstance<py::slice>(obj)) {
            auto s = obj.cast<py::slice>();
            auto extract_opt = [](py::handle h) -> std::optional<int64_t> {
                if (h.is_none()) return std::nullopt;
                return h.cast<int64_t>();
            };

            py::object start, stop, step;
            start = s.attr("start");
            stop = s.attr("stop");  
            step = s.attr("step");

            result.emplace_back(indexing::Slice(
                extract_opt(start), extract_opt(stop), extract_opt(step)
            ));
        } else {
            throw py::type_error("unsupported index type");
        }
    };
    
    if (py::isinstance<py::tuple>(idx)) {
        for (auto item : idx.cast<py::tuple>()) {
            convert_one(item);
        }
    } else {
        convert_one(idx);
    }
    return result;
}

inline py::object scalar_to_pyobject(Scalar s, Dtype dtype) {
    if (dtype.is_floating()) return py::cast(s.to<double>());
    if (dtype.is_bool())     return py::cast(s.to<bool>());
    return py::cast(s.to<int64_t>());
}

inline py::object tensor_to_list(const Tensor& tensor) {
    if (tensor.ndim() == 0) {
        // Scalar — return a Python scalar, not a list
        return scalar_to_pyobject(tensor.item(), tensor.dtype());
    }
    
    if (tensor.ndim() == 1) {
        // Base case: build a flat Python list from the data
        int64_t n = tensor.shape()[0];
        py::list result(n);
        for (int64_t i = 0; i < n; ++i) {
            result[i] = scalar_to_pyobject(tensor[i].item(), tensor.dtype());
        }
        return result;
    }
    
    // Recursive case: list of sublists
    int64_t n = tensor.shape()[0];
    py::list result(n);
    for (int64_t i = 0; i < n; ++i) {
        result[i] = tensor_to_list(tensor[i]);
    }
    return result;
}

} // namespace raxis::python