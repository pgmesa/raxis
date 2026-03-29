#pragma once

#include <vector>
#include <cstdint>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <raxis/core/types.h>
#include <raxis/core/scalar.h>


namespace py = pybind11;


namespace raxis {
    struct DimArray : IntArray {
        using IntArray::IntArray;
        using IntArray::operator=;
    };
}

namespace pybind11::detail {

template <>
struct type_caster<raxis::IntArray> {
public:
    PYBIND11_TYPE_CASTER(raxis::IntArray, _("list[int] | tuple[int, ...]"));

    // Python -> C++
    bool load(handle src, bool) {
        value.clear();

        if (py::isinstance<py::sequence>(src)) {
            for (auto item : py::reinterpret_borrow<py::sequence>(src)) {
                value.push_back(py::cast<int64_t>(item));
            }
            return true;
        }

        return false;
    }

    // C++ -> Python
    static handle cast(
        const raxis::IntArray& src,
        return_value_policy,
        handle
    ) {
        py::list lst;
        for (auto v : src) {
            lst.append(py::int_(v));
        }
        return lst.release();
    }
};


template <>
struct type_caster<raxis::DimArray> {
public:
    PYBIND11_TYPE_CASTER(raxis::DimArray, _("list[int] | tuple[int, ...] | int"));

    bool load(handle src, bool) {
        raxis::DimArray arr;
        if (py::isinstance<py::int_>(src)) {
            arr.push_back(py::cast<int64_t>(src));
            value = std::move(arr);
            return true;
        }

        if (py::isinstance<py::sequence>(src)) {
            for (auto item : py::reinterpret_borrow<py::sequence>(src)) {
                arr.push_back(py::cast<int64_t>(item));
            }
            value = std::move(arr);
            return true;
        }

        return false;
    }

    static handle cast(
        const raxis::DimArray& src,
        return_value_policy policy,
        handle parent
    ) {
        return type_caster<raxis::IntArray>::cast(
            static_cast<const raxis::IntArray&>(src), policy, parent);
    }
};


template <>
struct type_caster<raxis::Scalar> {
    PYBIND11_TYPE_CASTER(raxis::Scalar, const_name("int | float | bool"));

    bool load(handle src, bool) {
        if (PyBool_Check(src.ptr())) {
            value = raxis::Scalar(src.ptr() == Py_True);
        } else if (PyLong_Check(src.ptr())) {
            value = raxis::Scalar(PyLong_AsLongLong(src.ptr()));
        } else if (PyFloat_Check(src.ptr())) {
            value = raxis::Scalar(PyFloat_AsDouble(src.ptr()));
        } else {
            return false;
        }
        return true;
    }

    static handle cast(const raxis::Scalar& s, return_value_policy, handle) {
        switch (s.tag()) {
            case raxis::Scalar::Tag::Bool:
                return py::bool_(s.to<bool>()).release();
            case raxis::Scalar::Tag::Integral:
                return py::int_(s.to<int64_t>()).release();
            case raxis::Scalar::Tag::Floating:
                return py::float_(s.to<double>()).release();
        }
        return py::none().release();
    }
};

}
