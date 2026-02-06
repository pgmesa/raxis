
#include <memory>
#include <optional>

#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <raxis/raxis.hpp>

#include "casters.hpp"
#include "parsers.hpp"


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

    c.attr("float32") = py::cast(dtypes::Float32);
    c.attr("float64") = py::cast(dtypes::Float64);
    c.attr("int32")   = py::cast(dtypes::Int32);
    c.attr("int64")   = py::cast(dtypes::Int64);
    c.attr("uint8")   = py::cast(dtypes::Uint8);
    c.attr("uint16")  = py::cast(dtypes::Uint16);
    c.attr("bool")    = py::cast(dtypes::Bool);
    
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
        .def_property_readonly("strides", [](const raxis::NDArray& self) -> py::tuple {
            pybind11::tuple py_strides = pybind11::cast(self.strides());
            return py_strides;
        })
        .def_property_readonly("byte_strides", [](const raxis::NDArray& self) -> py::tuple {
            pybind11::tuple py_byte_strides = pybind11::cast(self.byte_strides());
            return py_byte_strides;
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
        .def("__repr__", [](const raxis::NDArray& arr) -> std::string {
            // Create numpy view directly
            py::array np_arr = ndarray_to_numpy(
                py::cast(&arr, py::return_value_policy::reference), arr
            );
            std::string repr_str = py::repr(np_arr).cast<std::string>();
            
            size_t pos = repr_str.find("array(");
            if (pos != std::string::npos) {
                repr_str.replace(pos, 6, "raxis(");
            }
            
            return repr_str;
        })

        .def("__len__", [](const raxis::NDArray& self) {
            return self.shape()[0];
        })

        // Normal basic operations
        .def("__add__",
            py::overload_cast<const raxis::NDArray&>(&raxis::NDArray::add, py::const_),
            py::arg("other"))
        // .def("__add__",
        //     py::overload_cast<double>(&raxis::NDArray::add, py::const_),
        //     py::arg("other"))

        .def("__sub__",
            py::overload_cast<const raxis::NDArray&>(&raxis::NDArray::sub, py::const_),
            py::arg("other"))
        // .def("__sub__",
        //     py::overload_cast<double>(&raxis::NDArray::sub, py::const_),
        //     py::arg("other"))

        .def("__mul__",
            py::overload_cast<const raxis::NDArray&>(&raxis::NDArray::mul, py::const_),
            py::arg("other"))
        // .def("__mul__",
        //     py::overload_cast<double>(&raxis::NDArray::mul, py::const_),
        //     py::arg("other"))

        .def("__truediv__",
            py::overload_cast<const raxis::NDArray&>(&raxis::NDArray::div, py::const_),
            py::arg("other"))
        // .def("__truediv__",
        //     py::overload_cast<double>(&raxis::NDArray::div, py::const_),
        //     py::arg("other"))

        // In-place ops (must return NDArray&)
        .def("__iadd__",
            py::overload_cast<const raxis::NDArray&>(&raxis::NDArray::add_, py::const_),
            py::arg("other"))
        // .def("__iadd__",
        //     py::overload_cast<double>(&raxis::NDArray::add_),
        //     py::arg("other"))

        .def("__isub__",
            py::overload_cast<const raxis::NDArray&>(&raxis::NDArray::sub_, py::const_),
            py::arg("other"))
        // .def("__isub__",
        //     py::overload_cast<double>(&raxis::NDArray::sub_),
        //     py::arg("other"))

        .def("__imul__",
            py::overload_cast<const raxis::NDArray&>(&raxis::NDArray::mul_, py::const_),
            py::arg("other"))
        // .def("__imul__",
        //     py::overload_cast<double>(&raxis::NDArray::mul_),
        //     py::arg("other"))

        .def("__itruediv__",
            py::overload_cast<const raxis::NDArray&>(&raxis::NDArray::div_, py::const_),
            py::arg("other"))
        // .def("__itruediv__",
        //     py::overload_cast<double>(&raxis::NDArray::div_),
        //     py::arg("other"))

        // Reverse ops
        .def("__radd__",
            py::overload_cast<const raxis::NDArray&>(&raxis::NDArray::add, py::const_),
            py::arg("other"))
        // .def("__radd__",
        //     py::overload_cast<double>(&raxis::NDArray::add, py::const_),
        //     py::arg("other"))

        .def("__rsub__",
            py::overload_cast<const raxis::NDArray&>(&raxis::NDArray::rsub, py::const_),
            py::arg("other"))
        // .def("__rsub__",
        //     py::overload_cast<double>(&raxis::NDArray::rsub, py::const_),
        //     py::arg("other"))

        .def("__rmul__",
            py::overload_cast<const raxis::NDArray&>(&raxis::NDArray::mul, py::const_),
            py::arg("other"))
        // .def("__rmul__",
        //     py::overload_cast<double>(&raxis::NDArray::mul, py::const_),
        //     py::arg("other"))

        .def("__rtruediv__",
            py::overload_cast<const raxis::NDArray&>(&raxis::NDArray::rdiv, py::const_),
            py::arg("other"))
        // .def("__rtruediv__",
        //     py::overload_cast<double>(&raxis::NDArray::rdiv, py::const_),
        //     py::arg("other"))
        
        // Functions to access operations
        .def("copy_",
            py::overload_cast<const raxis::NDArray&>(&raxis::NDArray::copy_, py::const_),
            py::arg("other"))
        .def("to",
            py::overload_cast<raxis::Dtype>(&raxis::NDArray::to, py::const_),
            py::arg("dtype"))
        

        .def("add",
            py::overload_cast<const raxis::NDArray&>(&raxis::NDArray::add, py::const_),
            py::arg("other"))
        // .def("add",
        //     py::overload_cast<double>(&raxis::NDArray::add, py::const_),
        //     py::arg("other"))

        .def("sub",
            py::overload_cast<const raxis::NDArray&>(&raxis::NDArray::sub, py::const_),
            py::arg("other"))
        // .def("sub",
        //     py::overload_cast<double>(&raxis::NDArray::sub, py::const_),
        //     py::arg("other"))

        .def("mul",
            py::overload_cast<const raxis::NDArray&>(&raxis::NDArray::mul, py::const_),
            py::arg("other"))
        // .def("mul",
        //     py::overload_cast<double>(&raxis::NDArray::mul, py::const_),
        //     py::arg("other"))

        .def("div",
            py::overload_cast<const raxis::NDArray&>(&raxis::NDArray::div, py::const_),
            py::arg("other"))
        // .def("div",
        //     py::overload_cast<double>(&raxis::NDArray::div, py::const_),
        //     py::arg("other"))

        .def("add_",
            py::overload_cast<const raxis::NDArray&>(&raxis::NDArray::add_, py::const_),
            py::arg("other"))
        // .def("add_",
        //     py::overload_cast<double>(&raxis::NDArray::add_, py::const_),
        //     py::arg("other"))

        .def("sub_",
            py::overload_cast<const raxis::NDArray&>(&raxis::NDArray::sub_, py::const_),
            py::arg("other"))
        // .def("sub_",
        //     py::overload_cast<double>(&raxis::NDArray::sub_, py::const_),
        //     py::arg("other"))

        .def("mul_",
            py::overload_cast<const raxis::NDArray&>(&raxis::NDArray::mul_, py::const_),
            py::arg("other"))
        // .def("mul_",
        //     py::overload_cast<double>(&raxis::NDArray::mul_, py::const_),
        //     py::arg("other"))

        .def("div_",
            py::overload_cast<const raxis::NDArray&>(&raxis::NDArray::div_, py::const_),
            py::arg("other"))
        // .def("div_",
        //     py::overload_cast<double>(&raxis::NDArray::div_, py::const_),
        //     py::arg("other"))

        // Reversed
        .def("rsub",
            py::overload_cast<const raxis::NDArray&>(&raxis::NDArray::rsub, py::const_),
            py::arg("other"))
        // .def("rsub",
        //     py::overload_cast<double>(&raxis::NDArray::rsub, py::const_),
        //     py::arg("other"))

        .def("rdiv",
            py::overload_cast<const raxis::NDArray&>(&raxis::NDArray::rdiv, py::const_),
            py::arg("other"))
        // .def("rdiv",
        //     py::overload_cast<double>(&raxis::NDArray::rdiv, py::const_),
        //     py::arg("other"))

        .def("rsub_",
            py::overload_cast<const raxis::NDArray&>(&raxis::NDArray::rsub_, py::const_),
            py::arg("other"))
        // .def("rsub_",
        //     py::overload_cast<double>(&raxis::NDArray::rsub_, py::const_),
        //     py::arg("other"))

        .def("rdiv_",
            py::overload_cast<const raxis::NDArray&>(&raxis::NDArray::rdiv_, py::const_),
            py::arg("other"))
        // .def("rdiv_",
        //     py::overload_cast<double>(&raxis::NDArray::rdiv_, py::const_),
        //     py::arg("other"))

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
       

    // Basic operations
    c.def(
        "_copy_from",
        py::overload_cast<const raxis::NDArray&, raxis::NDArray&>(&raxis::copy), 
        py::arg("src"), py::arg("dest"));

    c.def(
        "add",
        py::overload_cast<const raxis::NDArray&, const raxis::NDArray&>(&raxis::add), 
        py::arg("arr1"), py::arg("arr2"));

    c.def(
        "sub", 
        py::overload_cast<const raxis::NDArray&, const raxis::NDArray&>(&raxis::sub), 
        py::arg("arr1"), py::arg("arr2"));

    c.def(
        "mul", 
        py::overload_cast<const raxis::NDArray&, const raxis::NDArray&>(&raxis::mul), 
        py::arg("arr1"), py::arg("arr2"));
        
    c.def("div", 
        py::overload_cast<const raxis::NDArray&, const raxis::NDArray&>(&raxis::div), 
        py::arg("arr1"), py::arg("arr2"));

    c.def(
        "rsub", 
        py::overload_cast<const raxis::NDArray&, const raxis::NDArray&>(&raxis::rsub), 
        py::arg("arr1"), py::arg("arr2"));
        
    c.def("rdiv", 
        py::overload_cast<const raxis::NDArray&, const raxis::NDArray&>(&raxis::rdiv), 
        py::arg("arr1"), py::arg("arr2"));
}
