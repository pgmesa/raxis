
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
    c.attr("uint16")  = py::cast(dtypes::Uint16);
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

    // Implicit conversion from str → Device (so device="cuda:0" works)
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

        .def("__array__", [](py::object self) {
            auto& a = self.cast<raxis::Tensor&>();
            return raxis::python::tensor_to_numpy(self, a);
        })

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

        .def("zero_", 
            py::overload_cast<>(&raxis::Tensor::zero_, py::const_))
        .def("fill_", 
            py::overload_cast<raxis::Scalar>(&raxis::Tensor::fill_, py::const_))

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

        ;
    
    // Initializers
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

    c.def("empty", [](raxis::IntArray shape, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device);
        return raxis::empty(shape, options);
    }, py::arg("shape"), py::arg("dtype") = py::none(), py::arg("device") = py::none());

    c.def("empty_like", [](const raxis::Tensor& other, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device);
        return raxis::empty_like(other, options);
    }, py::arg("other"), py::arg("dtype") = py::none(), py::arg("device") = py::none());

    c.def("zeros", [](raxis::IntArray shape, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device);
        return raxis::zeros(shape, options);
    }, py::arg("shape"), py::arg("dtype") = py::none(), py::arg("device") = py::none());

    c.def("zeros_like", [](const raxis::Tensor& other, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device);
        return raxis::zeros_like(other, options);
    }, py::arg("other"), py::arg("dtype") = py::none(), py::arg("device") = py::none());

    c.def("ones", [](raxis::IntArray shape, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device);
        return raxis::ones(shape, options);
    }, py::arg("shape"), py::arg("dtype") = py::none(), py::arg("device") = py::none());

    c.def("ones_like", [](const raxis::Tensor& other, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device);
        return raxis::ones_like(other, options);
    }, py::arg("other"), py::arg("dtype") = py::none(), py::arg("device") = py::none());

    c.def("full", [](raxis::IntArray shape, raxis::Scalar fill_value, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device);
        return raxis::full(shape, fill_value, options);
    }, py::arg("shape"), py::arg("fill_value"), py::arg("dtype") = py::none(), py::arg("device") = py::none());

    c.def("full_like", [](const raxis::Tensor& other, raxis::Scalar fill_value, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device);
        return raxis::full_like(other, fill_value, options);
    }, py::arg("other"), py::arg("fill_value"), py::arg("dtype") = py::none(), py::arg("device") = py::none());
    

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

} // pybind11_module
