
#include <memory>
#include <optional>

#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <raxis/raxis.hpp>

#include "casters.hpp"
#include "parsers.hpp"


namespace py = pybind11;


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

    // Dtype
    py::class_<raxis::Dtype>(c, "dtype", py::module_local())
        .def_property_readonly("name", &raxis::Dtype::name)
        .def_property_readonly("itemsize", &raxis::Dtype::itemsize)
        .def("__repr__", [](const raxis::Dtype& d) {
            return d.name();
        })
        .def("__eq__", [](const raxis::Dtype& a, const raxis::Dtype& b) {
            return a == b;
        })
        .def("__hash__", [](const raxis::Dtype& d) {
            return static_cast<size_t>(d.kind());
        });

    c.attr("float32") = py::cast(raxis::Dtype::Float32);
    c.attr("float64") = py::cast(raxis::Dtype::Float64);
    c.attr("int32")   = py::cast(raxis::Dtype::Int32);
    c.attr("int64")   = py::cast(raxis::Dtype::Int64);
    c.attr("uint8")   = py::cast(raxis::Dtype::Uint8);
    c.attr("bool")    = py::cast(raxis::Dtype::Bool);
    
    c.def("set_default_dtype", &raxis::set_default_dtype, py::arg("dtype"));

    // NDArray
    py::class_<raxis::NDArray>(c, "ndarray")
        .def(py::init([](const raxis::NDArray& array) {
            return array;
        }))
        .def_property_readonly("shape", [](const raxis::NDArray& self) -> py::tuple {
            pybind11::tuple py_shape = pybind11::cast(self.shape());
            return py_shape;
        })
        .def_property_readonly("dtype", &raxis::NDArray::dtype, "Data type")
        .def("numel", &raxis::NDArray::numel)
        .def("dim", &raxis::NDArray::ndim)
        .def("numpy", [](py::object self) {
            auto& a = self.cast<raxis::NDArray&>();
            return ndarray_to_numpy(self, a);
        })
        .def("__array__", [](py::object self) {
            auto& a = self.cast<raxis::NDArray&>();
            return ndarray_to_numpy(self, a);
        })
        .def("__repr__", [](const raxis::NDArray& self) -> std::string {
            return self.to_string();
        })

        .def("__len__", [](const raxis::NDArray& self) {
            return self.shape()[0];
        })

        // Normal basic operations
        .def("__add__", [](const raxis::NDArray& self, const raxis::NDArray& other) {
            return raxis::add(self, other);
        }, py::arg("other"))
        // .def("__add__", py::overload_cast<double>(&raxis::add, py::const_), py::arg("other"))

        // In-place operations
        // .def("__iadd__", py::overload_cast<const raxis::NDArray&>(&raxis::NDArray::add_), py::arg("other"))
        // .def("__iadd__", py::overload_cast<double>(&raxis::NDArray::add_), py::arg("other"))

        // Reverse operations
        //.def("__radd__", py::overload_cast<const raxis::NDArray&>(&raxis::add, py::const_), py::arg("other"))
        // .def("__radd__", py::overload_cast<double>(&raxis::NDArray::add, py::const_), py::arg("other"))
        ;
    
    // Initializers
    // c.def("array", [](PyNDArrayData data, bool requires_grad, PyDevice device, PyDtype dtype) {
    //     torch::NDArray NDArray = data.get().to(device.get()).to(dtype.get());
    //     return raxis::NDArray(NDArray, requires_grad);
    // }, py::arg("data"), py::arg("requires_grad") = false, py::arg("device") = py::none(), py::arg("dtype") = py::none());


   c.def(
        "ones", &raxis::ones,
        py::arg("shape"), py::arg("dtype") = raxis::default_dtype()
    );

    c.def(
        "zeros", &raxis::zeros,
        py::arg("shape"), py::arg("dtype") = raxis::default_dtype()
    );

    // c.def("full", [](PyDims shape, double fill_value, bool requires_grad, PyDevice device, PyDtype dtype) {
    //     torch::NDArrayOptions options = torch::NDArrayOptions();
    //     options = options.device(device.get()).dtype(dtype.get());
        
    //     return raxis::full(shape.get(), fill_value, requires_grad, options);
    // }, py::arg("shape"), py::arg("fill_value"), py::arg("requires_grad") = false, py::arg("device") = py::none(), py::arg("dtype") = py::none());
       

    // // Basic operations
    // c.def("add", py::overload_cast<const raxis::NDArray&, const raxis::NDArray&>(&raxis::add), py::arg("t1"), py::arg("t2"));
    // c.def("add", py::overload_cast<const raxis::NDArray&, double>(&raxis::add), py::arg("NDArray"), py::arg("scalar"));

}
