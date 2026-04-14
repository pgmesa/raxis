#pragma once

#include <vector>
#include <cstdint>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <raxis/core/types.h>
#include <raxis/core/scalar.h>


namespace py = pybind11;


namespace raxis {

    struct Vec : DimVector {
        using DimVector::DimVector;
        using DimVector::operator=;
    };

    struct VecOrInt : DimVector {
        using DimVector::DimVector;
        using DimVector::operator=;
    };

}

namespace pybind11::detail {

template <>
struct type_caster<raxis::DimVector> {
public:
    PYBIND11_TYPE_CASTER(raxis::DimVector, _("tuple[int, ...]"));

    bool load(handle src, bool) {
        return false;
    }

    static handle cast(const raxis::DimVector& src, return_value_policy, handle) {
        py::tuple tup(src.size());
        for (size_t i = 0; i < src.size(); ++i) {
            tup[i] = py::int_(src[i]);
        }
        return tup.release();
    }
};

template <>
struct type_caster<raxis::IntArrayRef> {
public:
    PYBIND11_TYPE_CASTER(raxis::IntArrayRef, _("list[int] | tuple[int, ...]"));

    // No load, ArrayRef is non-owning, never constructed from Python
    bool load(handle, bool) {
        return false;
    }

    static handle cast(raxis::IntArrayRef src, return_value_policy, handle) {
        py::tuple tup(src.size());
        for (size_t i = 0; i < src.size(); ++i) {
            tup[i] = py::int_(src[i]);
        }
        return tup.release();
    }
};

template <>
struct type_caster<raxis::Vec> {
public:
    PYBIND11_TYPE_CASTER(raxis::Vec, _("list[int] | tuple[int, ...]"));

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

    static handle cast(const raxis::Vec& src, return_value_policy policy, handle parent) {
        return type_caster<raxis::DimVector>::cast(src, policy, parent);
    }
};

template <>
struct type_caster<raxis::VecOrInt> {
public:
    PYBIND11_TYPE_CASTER(raxis::VecOrInt, _("list[int] | tuple[int, ...] | int"));

    bool load(handle src, bool) {
        value.clear();
        if (py::isinstance<py::int_>(src)) {
            value.push_back(py::cast<int64_t>(src));
            return true;
        }
        if (py::isinstance<py::sequence>(src)) {
            for (auto item : py::reinterpret_borrow<py::sequence>(src)) {
                value.push_back(py::cast<int64_t>(item));
            }
            return true;
        }
        return false;
    }

    static handle cast(const raxis::VecOrInt& src, return_value_policy policy, handle parent) {
        return type_caster<raxis::DimVector>::cast(src, policy, parent);
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
