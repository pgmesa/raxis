
#include <memory>
#include <optional>

#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <raxis/raxis.h>

#include "casters.h"
#include "numpy_utils.h"
#include "python_tensor_factory.h"


namespace py = pybind11;
namespace dtypes = raxis::dtypes;


PYBIND11_MODULE(_C, c) {
    c.doc() = "Raxis Python bindings";

    // Exceptions
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const std::exception& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });

    // Dtype binding
    py::class_<raxis::Dtype>(c, "dtype", py::module_local())
        .def(py::init([](py::handle obj) {
            return raxis::python::dtype_from_python(obj);
        }))
        .def_property_readonly("name", &raxis::Dtype::name)
        .def_property_readonly("itemsize", &raxis::Dtype::itemsize)
        .def_property_readonly("is_floating", &raxis::Dtype::is_floating)
        .def_property_readonly("is_integral", &raxis::Dtype::is_integral)
        .def_property_readonly("is_signed", &raxis::Dtype::is_signed)
        .def("__repr__", [](const raxis::Dtype& d) {
            return d.name();
        })
        .def("__eq__", [](const raxis::Dtype& a, py::object other) -> bool {
            if (!py::isinstance<raxis::Dtype>(other)) return false;
            return a == other.cast<raxis::Dtype>();
        })
        .def("__hash__", [](const raxis::Dtype& d) {
            return static_cast<size_t>(d.kind());
        });
    
    py::implicitly_convertible<py::handle, raxis::Dtype>();

    c.attr("float32") = py::cast(dtypes::Float32);
    c.attr("float64") = py::cast(dtypes::Float64);
    c.attr("int32")   = py::cast(dtypes::Int32);
    c.attr("int64")   = py::cast(dtypes::Int64);
    c.attr("uint8")   = py::cast(dtypes::Uint8);
    c.attr("bool")    = py::cast(dtypes::Bool);
    
    c.def("set_default_dtype", &raxis::set_default_dtype, py::arg("dtype"));

    // Device binding
    py::class_<raxis::Device>(c, "device", py::module_local())
        .def(py::init<const std::string&>())
        .def_property_readonly("type", [](const raxis::Device& d) {
            return raxis::device_type_name(d.type(), true);
        })
        .def_property_readonly("index", &raxis::Device::index)
        .def("__repr__", &raxis::Device::str)
        .def("__eq__", [](const raxis::Device& a, py::object other) -> bool {
            if (!py::isinstance<raxis::Device>(other)) return false;
            return a == other.cast<raxis::Device>();
        })
        .def("__hash__", [](const raxis::Device& d) { });

    // Implicit conversion from str -> Device (so device="cuda:0" works)
    py::implicitly_convertible<std::string, raxis::Device>();

    // Constants
    c.attr("inf") = raxis::inf;
    c.attr("nan") = raxis::nan;
    c.attr("pi") = raxis::pi;
    c.attr("e") = raxis::e;

    // Tensor
    py::class_<raxis::Tensor>(c, "Tensor")
        .def_property_readonly("shape", [](const raxis::Tensor& self) -> py::tuple {
            pybind11::tuple py_shape = pybind11::cast(self.shape());
            return py_shape;
        })
        .def_property_readonly("strides", [](const raxis::Tensor& self) -> py::tuple {
            pybind11::tuple py_strides = pybind11::cast(self.strides());
            return py_strides;
        })
        .def_property_readonly("byte_strides", [](const raxis::Tensor& self) -> py::tuple {
            pybind11::tuple py_byte_strides = pybind11::cast(self.byte_strides());
            return py_byte_strides;
        })
        .def_property_readonly("dtype", &raxis::Tensor::dtype, "Data type")
        .def_property_readonly("device", &raxis::Tensor::device, "Device")
        .def("numel", &raxis::Tensor::numel)
        .def("dim", &raxis::Tensor::ndim)
        .def_property_readonly("ndim", &raxis::Tensor::ndim)
        .def("tolist", [](const raxis::Tensor& self) -> py::object {
            return raxis::python::tensor_to_list(self);
        })
        .def("item", [](const raxis::Tensor& self) -> py::object {
            raxis::Scalar s = self.item();
            return raxis::python::scalar_to_pyobject(s, self.dtype());
        })
        .def("numpy", [](py::object self) {
            auto& a = self.cast<raxis::Tensor&>();
            return raxis::python::tensor_to_numpy(self, a);
        })

        .def("__array__", [](py::object self, py::object dtype, py::object copy) {
            auto& a = self.cast<raxis::Tensor&>();
            py::array arr = raxis::python::tensor_to_numpy(self, a);

            if (!dtype.is_none()) {
                arr = arr.attr("astype")(dtype, py::arg("copy") = false);
            }
            if (!copy.is_none() && copy.cast<bool>()) {
                arr = arr.attr("copy")();
            }

            return arr;
        }, py::arg("dtype") = py::none(), py::arg("copy") = py::none())

        .def("__len__", [](const raxis::Tensor& self) {
            return self.shape()[0];
        })
        .def("__getitem__", [](const raxis::Tensor& self, py::object idx) {
            return self.index(raxis::python::py_index_to_tensor_indices(idx));
        })
        .def("__setitem__", [](raxis::Tensor& self, py::object idx, const raxis::Tensor& value) {
            self.index_put_(raxis::python::py_index_to_tensor_indices(idx), value);
        })
        .def("__setitem__", [](raxis::Tensor& self, py::object idx, raxis::Scalar value) {
            self.index_put_(raxis::python::py_index_to_tensor_indices(idx), value);
        })

        // Shape / view operations
        .def("squeeze", &raxis::Tensor::squeeze, py::arg("dim") = py::none())
        .def("unsqueeze", &raxis::Tensor::unsqueeze, py::arg("dim"))
        .def_property_readonly("mT", &raxis::Tensor::mT)
        .def("transpose", &raxis::Tensor::transpose, py::arg("dim0"), py::arg("dim1"))
        .def("permute", [](const raxis::Tensor& self, raxis::Vec dims) {
            return self.permute(dims);
        }, py::arg("dims"))
        .def("swapdims", &raxis::Tensor::swapdims, py::arg("dim0"), py::arg("dim1"))
        .def("movedim",
            py::overload_cast<int64_t, int64_t>(&raxis::Tensor::movedim, py::const_),
            py::arg("source"), py::arg("destination"))
        .def("movedim", [](const raxis::Tensor& self, raxis::VecOrInt source, raxis::VecOrInt destination) {
            return self.movedim(source, destination);
        }, py::arg("source"), py::arg("destination"))
        .def("view", [](const raxis::Tensor& self, raxis::Vec shape) {
            return self.view(shape);
        }, py::arg("shape"))
        .def("reshape", [](const raxis::Tensor& self, raxis::Vec shape) {
            return self.reshape(shape);
        }, py::arg("shape"))
        .def("broadcast_to", [](const raxis::Tensor& self, raxis::Vec shape) {
            return self.broadcast_to(shape);
        }, py::arg("shape"))

        .def("contiguous", &raxis::Tensor::contiguous)
        .def("is_contiguous", &raxis::Tensor::is_contiguous)

        // Normal basic operations
        .def("__add__",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::add, py::const_),
            py::arg("other"))
        .def("__add__",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::add, py::const_),
            py::arg("other"))

        .def("__sub__",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::sub, py::const_),
            py::arg("other"))
        .def("__sub__",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::sub, py::const_),
            py::arg("other"))

        .def("__mul__",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::mul, py::const_),
            py::arg("other"))
        .def("__mul__",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::mul, py::const_),
            py::arg("other"))

        .def("__truediv__",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::div, py::const_),
            py::arg("other"))
        .def("__truediv__",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::div, py::const_),
            py::arg("other"))

        // In-place ops
        .def("__iadd__",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::add_, py::const_),
            py::arg("other"))
        .def("__iadd__",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::add_, py::const_),
            py::arg("other"))

        .def("__isub__",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::sub_, py::const_),
            py::arg("other"))
        .def("__isub__",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::sub_, py::const_),
            py::arg("other"))

        .def("__imul__",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::mul_, py::const_),
            py::arg("other"))
        .def("__imul__",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::mul_, py::const_),
            py::arg("other"))

        .def("__itruediv__",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::div_, py::const_),
            py::arg("other"))
        .def("__itruediv__",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::div_, py::const_),
            py::arg("other"))

        // Reverse ops
        .def("__radd__",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::add, py::const_),
            py::arg("other"))
        .def("__radd__",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::add, py::const_),
            py::arg("other"))

        .def("__rsub__",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::rsub, py::const_),
            py::arg("other"))
        .def("__rsub__",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::rsub, py::const_),
            py::arg("other"))

        .def("__rmul__",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::mul, py::const_),
            py::arg("other"))
        .def("__rmul__",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::mul, py::const_),
            py::arg("other"))

        .def("__rtruediv__",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::rdiv, py::const_),
            py::arg("other"))
        .def("__rtruediv__",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::rdiv, py::const_),
            py::arg("other"))
        
        // Functions to access operations
        .def("clone",
            py::overload_cast<>(&raxis::Tensor::clone, py::const_))
        .def("copy_",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::copy_, py::const_),
            py::arg("other"))
        .def("to",
            py::overload_cast<raxis::Dtype>(&raxis::Tensor::to, py::const_),
            py::arg("dtype"))
        .def("detach",
            py::overload_cast<>(&raxis::Tensor::clone, py::const_))
        .def("cpu", &raxis::Tensor::cpu)

        .def("zero_", 
            py::overload_cast<>(&raxis::Tensor::zero_, py::const_))
        .def("fill_", 
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::fill_, py::const_))
        .def("uniform_",
            &raxis::Tensor::uniform_,
            py::arg("low") = 0.0, py::arg("high") = 1.0)
        .def("normal_",
            &raxis::Tensor::normal_,
            py::arg("mean") = 0.0, py::arg("std") = 1.0)
        .def("randint_",
            &raxis::Tensor::randint_,
            py::arg("low"), py::arg("high"))
        
        .def("dot", &raxis::Tensor::dot, py::arg("other"))
        .def("mv", &raxis::Tensor::mv, py::arg("vec"))
        .def("mm", &raxis::Tensor::mm, py::arg("mat2"))
        .def("bmm", &raxis::Tensor::bmm, py::arg("mat2"))
        .def("matmul", &raxis::Tensor::matmul, py::arg("other"))
        .def("__matmul__", &raxis::Tensor::matmul, py::arg("other"))

        .def("add",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::add, py::const_),
            py::arg("other"))
        .def("add",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::add, py::const_),
            py::arg("other"))

        .def("sub",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::sub, py::const_),
            py::arg("other"))
        .def("sub",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::sub, py::const_),
            py::arg("other"))

        .def("mul",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::mul, py::const_),
            py::arg("other"))
        .def("mul",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::mul, py::const_),
            py::arg("other"))

        .def("div",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::div, py::const_),
            py::arg("other"))
        .def("div",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::div, py::const_),
            py::arg("other"))

        .def("add_",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::add_, py::const_),
            py::arg("other"))
        .def("add_",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::add_, py::const_),
            py::arg("other"))

        .def("sub_",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::sub_, py::const_),
            py::arg("other"))
        .def("sub_",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::sub_, py::const_),
            py::arg("other"))

        .def("mul_",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::mul_, py::const_),
            py::arg("other"))
        .def("mul_",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::mul_, py::const_),
            py::arg("other"))

        .def("div_",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::div_, py::const_),
            py::arg("other"))
        .def("div_",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::div_, py::const_),
            py::arg("other"))

        // Reversed
        .def("rsub",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::rsub, py::const_),
            py::arg("other"))
        .def("rsub",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::rsub, py::const_),
            py::arg("other"))

        .def("rdiv",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::rdiv, py::const_),
            py::arg("other"))
        .def("rdiv",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::rdiv, py::const_),
            py::arg("other"))

        .def("rsub_",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::rsub_, py::const_),
            py::arg("other"))
        .def("rsub_",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::rsub_, py::const_),
            py::arg("other"))

        .def("rdiv_",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::rdiv_, py::const_),
            py::arg("other"))
        .def("rdiv_",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::rdiv_, py::const_),
            py::arg("other"))

        // Comparison operators
        .def("__gt__",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::gt, py::const_),
            py::arg("other"))
        .def("__gt__",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::gt, py::const_),
            py::arg("other"))
 
        .def("__ge__",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::ge, py::const_),
            py::arg("other"))
        .def("__ge__",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::ge, py::const_),
            py::arg("other"))
 
        .def("__lt__",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::lt, py::const_),
            py::arg("other"))
        .def("__lt__",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::lt, py::const_),
            py::arg("other"))
 
        .def("__le__",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::le, py::const_),
            py::arg("other"))
        .def("__le__",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::le, py::const_),
            py::arg("other"))
 
        .def("__eq__",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::eq, py::const_),
            py::arg("other"))
        .def("__eq__",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::eq, py::const_),
            py::arg("other"))
 
        .def("__ne__",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::ne, py::const_),
            py::arg("other"))
        .def("__ne__",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::ne, py::const_),
            py::arg("other"))
 
        // Comparison methods
        .def("gt",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::gt, py::const_),
            py::arg("other"))
        .def("gt",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::gt, py::const_),
            py::arg("other"))
 
        .def("ge",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::ge, py::const_),
            py::arg("other"))
        .def("ge",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::ge, py::const_),
            py::arg("other"))
 
        .def("lt",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::lt, py::const_),
            py::arg("other"))
        .def("lt",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::lt, py::const_),
            py::arg("other"))
 
        .def("le",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::le, py::const_),
            py::arg("other"))
        .def("le",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::le, py::const_),
            py::arg("other"))
 
        .def("eq",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::eq, py::const_),
            py::arg("other"))
        .def("eq",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::eq, py::const_),
            py::arg("other"))
 
        .def("ne",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::ne, py::const_),
            py::arg("other"))
        .def("ne",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::ne, py::const_),
            py::arg("other"))
 
        // Elementwise methods
        .def("maximum",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::maximum, py::const_),
            py::arg("other"))
        .def("minimum",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::minimum, py::const_),
            py::arg("other"))
 
        .def("pow",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::pow, py::const_),
            py::arg("exponent"))
        .def("pow",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::pow, py::const_),
            py::arg("exponent"))
        .def("__pow__",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::pow, py::const_),
            py::arg("exponent"))
        .def("__pow__",
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::pow, py::const_),
            py::arg("exponent"))
 
        // Logical methods
        .def("logical_and",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::logical_and, py::const_),
            py::arg("other"))
        .def("logical_or",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::logical_or, py::const_),
            py::arg("other"))

        // Unary operators
        .def("__neg__", &raxis::Tensor::neg)
        .def("__abs__", &raxis::Tensor::abs)
        .def("__invert__", &raxis::Tensor::logical_not)
 
        // Unary methods
        .def("isnan", &raxis::Tensor::isnan)
        .def("neg", &raxis::Tensor::neg)
        .def("abs", &raxis::Tensor::abs)
        .def("ceil", &raxis::Tensor::ceil)
        .def("floor", &raxis::Tensor::floor)
        .def("round", &raxis::Tensor::round)
        .def("exp", &raxis::Tensor::exp)
        .def("log", &raxis::Tensor::log)
        .def("sqrt", &raxis::Tensor::sqrt)
        .def("rsqrt", &raxis::Tensor::rsqrt)
        .def("reciprocal", &raxis::Tensor::reciprocal)
        .def("sigmoid", &raxis::Tensor::sigmoid)
        .def("logical_not", &raxis::Tensor::logical_not)

        .def("neg_", &raxis::Tensor::neg_)
        .def("abs_", &raxis::Tensor::abs_)
        .def("ceil_", &raxis::Tensor::ceil_)
        .def("floor_", &raxis::Tensor::floor_)
        .def("round_", &raxis::Tensor::round_)
        .def("exp_", &raxis::Tensor::exp_)
        .def("log_", &raxis::Tensor::log_)
        .def("sqrt_", &raxis::Tensor::sqrt_)
        .def("rsqrt_", &raxis::Tensor::rsqrt_)
        .def("reciprocal_", &raxis::Tensor::reciprocal_)
        .def("sigmoid_", &raxis::Tensor::sigmoid_)
        .def("logical_not_", &raxis::Tensor::logical_not_)
        
        // Reduction methods
        .def("sum", [](
                const raxis::Tensor& self, 
                std::optional<raxis::VecOrInt> dim,
                bool keepdim,
                std::optional<raxis::Dtype> dtype
            ) -> raxis::Tensor {
                return dim.has_value()? self.sum(*dim, keepdim, dtype) : self.sum(dtype);
            },
            py::arg("dim") = py::none(),
            py::arg("keepdim") = false,
            py::arg("dtype") = py::none()
        )
        .def("nansum", [](
                const raxis::Tensor& self, 
                std::optional<raxis::VecOrInt> dim,
                bool keepdim,
                std::optional<raxis::Dtype> dtype
            ) -> raxis::Tensor {
                return dim.has_value()? self.nansum(*dim, keepdim, dtype) : self.nansum(dtype);
            },
            py::arg("dim") = py::none(),
            py::arg("keepdim") = false,
            py::arg("dtype") = py::none()
        )

        .def("mean", [](
                const raxis::Tensor& self, 
                std::optional<raxis::VecOrInt> dim,
                bool keepdim,
                std::optional<raxis::Dtype> dtype
            ) -> raxis::Tensor {
                return dim.has_value()? self.mean(*dim, keepdim, dtype) : self.mean(dtype);
            },
            py::arg("dim") = py::none(),
            py::arg("keepdim") = false,
            py::arg("dtype") = py::none()
        )
        .def("nanmean", [](
                const raxis::Tensor& self, 
                std::optional<raxis::VecOrInt> dim,
                bool keepdim,
                std::optional<raxis::Dtype> dtype
            ) -> raxis::Tensor {
                return dim.has_value()? self.nanmean(*dim, keepdim, dtype) : self.nanmean(dtype);
            },
            py::arg("dim") = py::none(),
            py::arg("keepdim") = false,
            py::arg("dtype") = py::none()
        )

        .def("prod", [](
                const raxis::Tensor& self, 
                std::optional<raxis::VecOrInt> dim,
                bool keepdim,
                std::optional<raxis::Dtype> dtype
            ) -> raxis::Tensor {
                return dim.has_value()? self.prod(*dim, keepdim, dtype) : self.prod(dtype);
            },
            py::arg("dim") = py::none(),
            py::arg("keepdim") = false,
            py::arg("dtype") = py::none()
        )

        .def("max", &raxis::Tensor::max)
        .def("min", &raxis::Tensor::min)

        .def("amax", [](
                const raxis::Tensor& self, 
                raxis::VecOrInt dim,
                bool keepdim
            ) -> raxis::Tensor {
                return self.amax(dim, keepdim);
            },
            py::arg("dim") = raxis::Vec{},
            py::arg("keepdim") = false
        )
        .def("amin", [](
                const raxis::Tensor& self, 
                raxis::VecOrInt dim,
                bool keepdim
            ) -> raxis::Tensor {
                return self.amin(dim, keepdim);
            },
            py::arg("dim") = raxis::Vec{},
            py::arg("keepdim") = false
        )
        
        ;
    
    // Factories
    c.def("tensor", [](py::object data, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device) {
        return raxis::python::tensor_from_python(data, dtype, device);
    },
    py::arg("data"),
    py::arg("dtype") = py::none(),
    py::arg("device") = py::none(),
    R"doc(
        Create a tensor from data.

        Args:
            data: A scalar, list, or nested list of scalars (int, float, bool).
            dtype: Optional target dtype. If None, inferred from data.
            device: Optional target device. Defaults to CPU.

        Returns:
            A new Tensor with the given data, dtype, and device.

        Examples:
            >>> raxis.tensor(5)                          # 0-d int tensor
            >>> raxis.tensor([1, 2, 3])                  # 1-d int tensor
            >>> raxis.tensor([[1.0, 2.0], [3.0, 4.0]])   # 2-d float tensor
            >>> raxis.tensor([1, 2.0, 3])                # promotes to float
    )doc");

    c.def("from_numpy", [](py::array arr) {
        return raxis::python::tensor_from_numpy(arr);
    });

    c.def("empty", [](raxis::VecOrInt shape, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device);
        return raxis::empty(shape, options);
    }, py::arg("shape"), py::arg("dtype") = py::none(), py::arg("device") = py::none());

    c.def("empty_like", [](const raxis::Tensor& other, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device);
        return raxis::empty_like(other, options);
    }, py::arg("other"), py::arg("dtype") = py::none(), py::arg("device") = py::none());

    c.def("zeros", [](raxis::VecOrInt shape, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device);
        return raxis::zeros(shape, options);
    }, py::arg("shape"), py::arg("dtype") = py::none(), py::arg("device") = py::none());

    c.def("zeros_like", [](const raxis::Tensor& other, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device);
        return raxis::zeros_like(other, options);
    }, py::arg("other"), py::arg("dtype") = py::none(), py::arg("device") = py::none());

    c.def("ones", [](raxis::VecOrInt shape, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device);
        return raxis::ones(shape, options);
    }, py::arg("shape"), py::arg("dtype") = py::none(), py::arg("device") = py::none());

    c.def("ones_like", [](const raxis::Tensor& other, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device);
        return raxis::ones_like(other, options);
    }, py::arg("other"), py::arg("dtype") = py::none(), py::arg("device") = py::none());

    c.def("full", [](raxis::VecOrInt shape, raxis::Scalar fill_value, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device);
        return raxis::full(shape, fill_value, options);
    }, py::arg("shape"), py::arg("fill_value"), py::arg("dtype") = py::none(), py::arg("device") = py::none());

    c.def("full_like", [](const raxis::Tensor& other, raxis::Scalar fill_value, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device);
        return raxis::full_like(other, fill_value, options);
    }, py::arg("other"), py::arg("fill_value"), py::arg("dtype") = py::none(), py::arg("device") = py::none());

    c.def("rand", [](raxis::VecOrInt shape, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device);
        return raxis::rand(shape, options);
    }, py::arg("shape"), py::arg("dtype") = py::none(), py::arg("device") = py::none());

    c.def("rand_like", [](const raxis::Tensor& other, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device);
        return raxis::rand_like(other, options);
    }, py::arg("other"), py::arg("dtype") = py::none(), py::arg("device") = py::none());

    c.def("randn", [](raxis::VecOrInt shape, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device);
        return raxis::randn(shape, options);
    }, py::arg("shape"), py::arg("dtype") = py::none(), py::arg("device") = py::none());

    c.def("randn_like", [](const raxis::Tensor& other, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device);
        return raxis::randn_like(other, options);
    }, py::arg("other"), py::arg("dtype") = py::none(), py::arg("device") = py::none());

    c.def("randint", [](int64_t low, int64_t high, raxis::VecOrInt shape, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device);
        return raxis::randint(low, high, shape, options);
    }, py::arg("low"), py::arg("high"), py::arg("shape"), py::arg("dtype") = py::none(), py::arg("device") = py::none());

    c.def("randint_like", [](const raxis::Tensor& other, int64_t low, int64_t high, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device);
        return raxis::randint_like(other, low, high, options);
    }, py::arg("other"), py::arg("low"), py::arg("high"), py::arg("dtype") = py::none(), py::arg("device") = py::none());

    c.def("manual_seed", &raxis::manual_seed, py::arg("seed"));
    

    // Shape operations
    c.def("squeeze", &raxis::squeeze, py::arg("input"), py::arg("dim") = py::none());
    c.def("unsqueeze", &raxis::unsqueeze, py::arg("input"), py::arg("dim"));
    c.def("transpose", &raxis::transpose, py::arg("input"), py::arg("dim0"), py::arg("dim1"));
    c.def("permute", &raxis::permute, py::arg("input"), py::arg("dims"));
    c.def("swapdims", &raxis::swapdims, py::arg("input"), py::arg("dim0"), py::arg("dim1"));
    c.def("movedim",
        py::overload_cast<const raxis::Tensor&, int64_t, int64_t>(&raxis::movedim),
        py::arg("input"), py::arg("source"), py::arg("destination"));
    
    c.def("movedim", [](const raxis::Tensor& input, raxis::VecOrInt source, raxis::VecOrInt destination) {
        return raxis::movedim(input, source, destination);
    }, py::arg("input"), py::arg("source"), py::arg("destination"));
    c.def("reshape", [](const raxis::Tensor& input, raxis::Vec shape) {
        return raxis::reshape(input, shape);
    }, py::arg("input"), py::arg("shape"));
    c.def("broadcast_to", [](const raxis::Tensor& input, raxis::Vec shape) {
        return raxis::broadcast_to(input, shape);
    }, py::arg("input"), py::arg("shape"));

    c.def("broadcast_tensors",  [](std::vector<raxis::Tensor> tensors) {
        return raxis::broadcast_tensors(tensors);
    }, py::arg("tensors"));
    c.def("broadcast_shapes", [](std::vector<raxis::Vec> shapes) {
        std::vector<raxis::IntArrayRef> refs;
        refs.reserve(shapes.size());
        for (const auto& s : shapes) {
            refs.emplace_back(s);
        }
        return raxis::broadcast_shapes(refs);
    }, py::arg("shapes"));

    // Linear algebra
    c.def("dot", &raxis::dot, py::arg("input"), py::arg("other"));
    c.def("mv", &raxis::mv, py::arg("input"), py::arg("vec"));
    c.def("addmv", &raxis::addmv,
        py::arg("input"), py::arg("mat"), py::arg("vec"),
        py::arg("beta") = 1, py::arg("alpha") = 1);
    c.def("mm", &raxis::mm, py::arg("input"), py::arg("mat2"));
    c.def("addmm", &raxis::addmm,
        py::arg("input"), py::arg("mat1"), py::arg("mat2"),
        py::arg("beta") = 1, py::arg("alpha") = 1);
    c.def("bmm", &raxis::bmm, py::arg("input"), py::arg("mat2"));
    c.def("addbmm", &raxis::addbmm,
        py::arg("input"), py::arg("batch1"), py::arg("batch2"),
        py::arg("beta") = 1, py::arg("alpha") = 1);
    c.def("matmul", &raxis::matmul, py::arg("input"), py::arg("other"));

    // Basic operations
    c.def(
        "_copy_from",
        py::overload_cast<const raxis::Tensor&, raxis::Tensor&>(&raxis::copy), 
        py::arg("src"), py::arg("dest"));

    c.def(
        "add",
        py::overload_cast<const raxis::Tensor&, const raxis::Tensor&>(&raxis::add), 
        py::arg("t1"), py::arg("t2"));
    c.def(
        "add",
        py::overload_cast<const raxis::Tensor&, raxis::Scalar>(&raxis::add), 
        py::arg("t1"), py::arg("value"));
    
    c.def(
        "sub", 
        py::overload_cast<const raxis::Tensor&, const raxis::Tensor&>(&raxis::sub), 
        py::arg("t1"), py::arg("t2"));
    c.def(
        "sub", 
        py::overload_cast<const raxis::Tensor&, raxis::Scalar>(&raxis::sub), 
        py::arg("t1"), py::arg("value"));

    c.def(
        "mul", 
        py::overload_cast<const raxis::Tensor&, const raxis::Tensor&>(&raxis::mul), 
        py::arg("t1"), py::arg("t2"));
    c.def(
        "mul", 
        py::overload_cast<const raxis::Tensor&, raxis::Scalar>(&raxis::mul), 
        py::arg("t1"), py::arg("value"));
        
    c.def("div", 
        py::overload_cast<const raxis::Tensor&, const raxis::Tensor&>(&raxis::div), 
        py::arg("t1"), py::arg("t2"));
    c.def("div", 
        py::overload_cast<const raxis::Tensor&, raxis::Scalar>(&raxis::div), 
        py::arg("t1"), py::arg("value"));

    c.def(
        "rsub", 
        py::overload_cast<const raxis::Tensor&, const raxis::Tensor&>(&raxis::rsub), 
        py::arg("t1"), py::arg("t2"));
    c.def(
        "rsub", 
        py::overload_cast<const raxis::Tensor&, raxis::Scalar>(&raxis::rsub), 
        py::arg("t1"), py::arg("value"));
        
    c.def("rdiv", 
        py::overload_cast<const raxis::Tensor&, const raxis::Tensor&>(&raxis::rdiv), 
        py::arg("t1"), py::arg("t2"));
    c.def("rdiv", 
        py::overload_cast<const raxis::Tensor&, raxis::Scalar>(&raxis::rdiv), 
        py::arg("t1"), py::arg("value"));


    // Comparison Ops
    c.def("gt",
        py::overload_cast<const raxis::Tensor&, const raxis::Tensor&>(&raxis::gt),
        py::arg("input"), py::arg("other"));
    c.def("gt",
        py::overload_cast<const raxis::Tensor&, raxis::Scalar>(&raxis::gt),
        py::arg("input"), py::arg("other"));
 
    c.def("ge",
        py::overload_cast<const raxis::Tensor&, const raxis::Tensor&>(&raxis::ge),
        py::arg("input"), py::arg("other"));
    c.def("ge",
        py::overload_cast<const raxis::Tensor&, raxis::Scalar>(&raxis::ge),
        py::arg("input"), py::arg("other"));
 
    c.def("lt",
        py::overload_cast<const raxis::Tensor&, const raxis::Tensor&>(&raxis::lt),
        py::arg("input"), py::arg("other"));
    c.def("lt",
        py::overload_cast<const raxis::Tensor&, raxis::Scalar>(&raxis::lt),
        py::arg("input"), py::arg("other"));
 
    c.def("le",
        py::overload_cast<const raxis::Tensor&, const raxis::Tensor&>(&raxis::le),
        py::arg("input"), py::arg("other"));
    c.def("le",
        py::overload_cast<const raxis::Tensor&, raxis::Scalar>(&raxis::le),
        py::arg("input"), py::arg("other"));
 
    c.def("eq",
        py::overload_cast<const raxis::Tensor&, const raxis::Tensor&>(&raxis::eq),
        py::arg("input"), py::arg("other"));
    c.def("eq",
        py::overload_cast<const raxis::Tensor&, raxis::Scalar>(&raxis::eq),
        py::arg("input"), py::arg("other"));
 
    c.def("ne",
        py::overload_cast<const raxis::Tensor&, const raxis::Tensor&>(&raxis::ne),
        py::arg("input"), py::arg("other"));
    c.def("ne",
        py::overload_cast<const raxis::Tensor&, raxis::Scalar>(&raxis::ne),
        py::arg("input"), py::arg("other"));
 
    // Elementwise Ops
    c.def("maximum",
        py::overload_cast<const raxis::Tensor&, const raxis::Tensor&>(&raxis::maximum),
        py::arg("input"), py::arg("other"));
    c.def("minimum",
        py::overload_cast<const raxis::Tensor&, const raxis::Tensor&>(&raxis::minimum),
        py::arg("input"), py::arg("other"));
 
    c.def("pow",
        py::overload_cast<const raxis::Tensor&, const raxis::Tensor&>(&raxis::pow),
        py::arg("base"), py::arg("exponent"));
    c.def("pow",
        py::overload_cast<const raxis::Tensor&, raxis::Scalar>(&raxis::pow),
        py::arg("base"), py::arg("exponent"));
    c.def("pow",
        py::overload_cast<raxis::Scalar, const raxis::Tensor&>(&raxis::pow),
        py::arg("base"), py::arg("exponent"));
 
    // Logical Ops
    c.def("logical_and",
        py::overload_cast<const raxis::Tensor&, const raxis::Tensor&>(&raxis::logical_and),
        py::arg("input"), py::arg("other"));
    c.def("logical_or",
        py::overload_cast<const raxis::Tensor&, const raxis::Tensor&>(&raxis::logical_or),
        py::arg("input"), py::arg("other"));


    // Unary Ops
    c.def("isnan",
        py::overload_cast<const raxis::Tensor&>(&raxis::isnan),
        py::arg("input"));
    c.def("neg",
        py::overload_cast<const raxis::Tensor&>(&raxis::neg),
        py::arg("input"));
    c.def("abs",
        py::overload_cast<const raxis::Tensor&>(&raxis::abs),
        py::arg("input"));
    c.def("ceil",
        py::overload_cast<const raxis::Tensor&>(&raxis::ceil),
        py::arg("input"));
    c.def("floor",
        py::overload_cast<const raxis::Tensor&>(&raxis::floor),
        py::arg("input"));
    c.def("round",
        py::overload_cast<const raxis::Tensor&>(&raxis::round),
        py::arg("input"));
    c.def("exp",
        py::overload_cast<const raxis::Tensor&>(&raxis::exp),
        py::arg("input"));
    c.def("log",
        py::overload_cast<const raxis::Tensor&>(&raxis::log),
        py::arg("input"));
    c.def("sqrt",
        py::overload_cast<const raxis::Tensor&>(&raxis::sqrt),
        py::arg("input"));
    c.def("rsqrt",
        py::overload_cast<const raxis::Tensor&>(&raxis::rsqrt),
        py::arg("input"));
    c.def("reciprocal",
        py::overload_cast<const raxis::Tensor&>(&raxis::reciprocal),
        py::arg("input"));
    c.def("sigmoid",
        py::overload_cast<const raxis::Tensor&>(&raxis::sigmoid),
        py::arg("input"));
    c.def("logical_not",
        py::overload_cast<const raxis::Tensor&>(&raxis::logical_not),
        py::arg("input"));

    // Reduction Ops
    c.def("sum", [](
            const raxis::Tensor& tensor, 
            std::optional<raxis::VecOrInt> dim,
            bool keepdim,
            std::optional<raxis::Dtype> dtype
        ) -> raxis::Tensor {
            return dim.has_value()? raxis::sum(tensor, *dim, keepdim, dtype) : raxis::sum(tensor, dtype);
        },
        py::arg("tensor"),
        py::arg("dim") = py::none(),
        py::arg("keepdim") = false,
        py::arg("dtype") = py::none()
    );
    c.def("nansum", [](
            const raxis::Tensor& tensor, 
            std::optional<raxis::VecOrInt> dim,
            bool keepdim,
            std::optional<raxis::Dtype> dtype
        ) -> raxis::Tensor {
            return dim.has_value()? raxis::nansum(tensor, *dim, keepdim, dtype) : raxis::nansum(tensor, dtype);
        },
        py::arg("tensor"),
        py::arg("dim") = py::none(),
        py::arg("keepdim") = false,
        py::arg("dtype") = py::none()
    );

    c.def("mean", [](
            const raxis::Tensor& tensor, 
            std::optional<raxis::VecOrInt> dim,
            bool keepdim,
            std::optional<raxis::Dtype> dtype
        ) -> raxis::Tensor {
            return dim.has_value()? raxis::mean(tensor, *dim, keepdim, dtype) : raxis::mean(tensor, dtype);
        },
        py::arg("tensor"),
        py::arg("dim") = py::none(),
        py::arg("keepdim") = false,
        py::arg("dtype") = py::none()
    );
    c.def("nanmean", [](
            const raxis::Tensor& tensor, 
            std::optional<raxis::VecOrInt> dim,
            bool keepdim,
            std::optional<raxis::Dtype> dtype
        ) -> raxis::Tensor {
            return dim.has_value()? raxis::nanmean(tensor, *dim, keepdim, dtype) : raxis::nanmean(tensor, dtype);
        },
        py::arg("tensor"),
        py::arg("dim") = py::none(),
        py::arg("keepdim") = false,
        py::arg("dtype") = py::none()
    );

    c.def("prod", [](
            const raxis::Tensor& tensor, 
            std::optional<raxis::VecOrInt> dim,
            bool keepdim,
            std::optional<raxis::Dtype> dtype
        ) -> raxis::Tensor {
            return dim.has_value()? raxis::prod(tensor, *dim, keepdim, dtype) : raxis::prod(tensor, dtype);
        },
        py::arg("tensor"),
        py::arg("dim") = py::none(),
        py::arg("keepdim") = false,
        py::arg("dtype") = py::none()
    );

    c.def("max", py::overload_cast<const raxis::Tensor&>(&raxis::max),
    py::arg("input"));
    c.def("min", py::overload_cast<const raxis::Tensor&>(&raxis::min),
        py::arg("input"));

    c.def("amax", [](
            const raxis::Tensor& tensor, 
            raxis::VecOrInt dim,
            bool keepdim
        ) -> raxis::Tensor {
            return raxis::amax(tensor, dim, keepdim);
        },
        py::arg("tensor"),
        py::arg("dim") = raxis::Vec{},
        py::arg("keepdim") = false
    );
    c.def("amin", [](
            const raxis::Tensor& tensor, 
            raxis::VecOrInt dim,
            bool keepdim
        ) -> raxis::Tensor {
            return raxis::amin(tensor, dim, keepdim);
        },
        py::arg("tensor"),
        py::arg("dim") = raxis::Vec{},
        py::arg("keepdim") = false
    );

} // pybind11_module
