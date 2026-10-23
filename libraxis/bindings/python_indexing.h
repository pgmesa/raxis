#pragma once

#include <pybind11/pybind11.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <raxis/core/dtype.h>
#include <raxis/util/exception.h>
#include <raxis/tensor/tensor.h>
#include <raxis/tensor/tensor_indexing.h>

#include "numpy_utils.h"
#include "python_tensor_factory.h"


namespace py = pybind11;


namespace raxis::python {

// From: torch/csrc/autograd/python_variable_indexing.cpp
//
// Converts a Python index (x[...]) into the TensorIndex list that
// indexing::get_item / set_item take.

// From: torch/csrc/utils/tensor_new.cpp (indexing_tensor_from_data)
// A list used as an index: a list of bools is a mask, anything else becomes
// int64 positions (an empty list too)
inline Tensor indexing_tensor_from_data(const py::object& seq) {
    Tensor tensor = tensor_from_python(seq, std::nullopt, std::nullopt);
    return tensor.dtype() == dtypes::Bool ? tensor : tensor.to(dtypes::Int64);
}

// From: torch/csrc/autograd/python_variable_indexing.cpp (treatSequenceAsTuple)
// NumPy's legacy rule: a short list that holds slices, Ellipsis, None,
// tensors or other sequences is a tuple of indices, so x[[[0, 1], [1, 0]]]
// is x[[0, 1], [1, 0]]. PyTorch keeps it with a deprecation warning.
inline bool treat_sequence_as_tuple(py::handle index) {
    if (py::isinstance<py::tuple>(index)) {
        return true;
    }
    if (py::isinstance<Tensor>(index) || is_numpy_array(index) || !PySequence_Check(index.ptr())) {
        return false;
    }
    const Py_ssize_t n = PySequence_Size(index.ptr());
    if (n < 0) {
        PyErr_Clear();
        return false;
    }
    if (n >= 32) {
        return false;
    }
    py::sequence seq = py::reinterpret_borrow<py::sequence>(index);
    for (Py_ssize_t i = 0; i < n; i++) {
        py::object obj = seq[i];
        if (py::isinstance<Tensor>(obj) || PySequence_Check(obj.ptr()) || py::isinstance<py::slice>(obj)
            || py::isinstance<py::ellipsis>(obj) || obj.is_none()) {
            if (PyErr_WarnEx(PyExc_UserWarning,
                    "Using a non-tuple sequence for multidimensional indexing is deprecated; "
                    "use x[tuple(seq)] instead of x[seq]. It will be interpreted as a tensor "
                    "index, x[raxis.tensor(seq)], which will result either in an error or a "
                    "different result", 1) < 0) {
                throw py::error_already_set();
            }
            return true;
        }
    }
    return false;
}

inline indexing::TensorIndex to_tensor_index(py::handle obj) {
    if (obj.is_none()) {
        return indexing::None;
    }
    if (py::isinstance<py::ellipsis>(obj)) {
        return indexing::Ellipsis;
    }
    // bool before int: bool is a subclass of int in Python
    if (py::isinstance<py::bool_>(obj)) {
        return obj.cast<bool>();
    }
    if (py::isinstance<py::int_>(obj)) {
        return obj.cast<int64_t>();
    }
    if (py::isinstance<py::slice>(obj)) {
        auto as_opt = [](py::handle h) -> std::optional<int64_t> {
            if (h.is_none()) return std::nullopt;
            return h.cast<int64_t>();
        };
        return indexing::Slice(as_opt(obj.attr("start")), as_opt(obj.attr("stop")), as_opt(obj.attr("step")));
    }
    if (py::isinstance<Tensor>(obj)) {
        return obj.cast<Tensor>();
    }
    if (is_numpy_array(obj)) {
        return tensor_from_numpy(py::reinterpret_borrow<py::array>(obj));
    }
    if (PySequence_Check(obj.ptr()) && !py::isinstance<py::str>(obj)) {
        return indexing_tensor_from_data(py::reinterpret_borrow<py::object>(obj));
    }
    // Anything with __index__ (numpy integers)
    PyObject* as_index = PyNumber_Index(obj.ptr());
    if (as_index == nullptr) {
        PyErr_Clear();
        throw IndexError(
            "only integers, slices (`:`), ellipsis (`...`), None and long or byte "
            "Variables are valid indices (got " + std::string(Py_TYPE(obj.ptr())->tp_name) + ")");
    }
    return py::reinterpret_steal<py::object>(as_index).cast<int64_t>();
}

// From: torch/csrc/autograd/python_variable_indexing.cpp (wrapTuple)
inline std::vector<indexing::TensorIndex> py_index_to_tensor_indices(py::handle index) {
    std::vector<indexing::TensorIndex> result;
    if (treat_sequence_as_tuple(index)) {
        for (py::handle item : py::tuple(py::reinterpret_borrow<py::object>(index))) {
            result.push_back(to_tensor_index(item));
        }
    } else {
        result.push_back(to_tensor_index(index));
    }
    return result;
}

// index_put_(indices, ...) takes a tuple of tensors or None (whole dim)
inline std::vector<Tensor> optional_tensor_list(const std::vector<std::optional<Tensor>>& indices) {
    std::vector<Tensor> result;
    result.reserve(indices.size());
    for (const auto& index : indices) {
        result.push_back(index.value_or(Tensor()));
    }
    return result;
}

} // namespace raxis::python
