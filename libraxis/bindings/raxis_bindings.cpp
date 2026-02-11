
#include <memory>
#include <optional>

#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <raxis/raxis.h>

#include "casters.h"
#include "parsers.h"


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
        .def("numel", &raxis::Tensor::numel)
        .def("dim", &raxis::Tensor::ndim)
        .def("numpy", [](py::object self) {
            auto& a = self.cast<raxis::Tensor&>();
            return tensor_to_numpy(self, a);
        })
        .def("__array__", [](py::object self) {
            auto& a = self.cast<raxis::Tensor&>();
            return tensor_to_numpy(self, a);
        })
        .def("__repr__", [](const raxis::Tensor& t) -> std::string {
            // Create numpy view directly
            py::array np_arr = tensor_to_numpy(
                py::cast(&t, py::return_value_policy::reference), t
            );
            std::string repr_str = py::repr(np_arr).cast<std::string>();
            
            size_t pos = repr_str.find("array(");
            if (pos != std::string::npos) {
                repr_str.replace(pos, 6, "raxis(");
            }
            
            return repr_str;
        })

        .def("__len__", [](const raxis::Tensor& self) {
            return self.shape()[0];
        })

        // Normal basic operations
        .def("__add__",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::add, py::const_),
            py::arg("other"))
        // .def("__add__",
        //     py::overload_cast<double>(&raxis::Tensor::add, py::const_),
        //     py::arg("other"))

        .def("__sub__",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::sub, py::const_),
            py::arg("other"))
        // .def("__sub__",
        //     py::overload_cast<double>(&raxis::Tensor::sub, py::const_),
        //     py::arg("other"))

        .def("__mul__",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::mul, py::const_),
            py::arg("other"))
        // .def("__mul__",
        //     py::overload_cast<double>(&raxis::Tensor::mul, py::const_),
        //     py::arg("other"))

        .def("__truediv__",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::div, py::const_),
            py::arg("other"))
        // .def("__truediv__",
        //     py::overload_cast<double>(&raxis::Tensor::div, py::const_),
        //     py::arg("other"))

        // In-place ops (must return Tensor&)
        .def("__iadd__",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::add_, py::const_),
            py::arg("other"))
        // .def("__iadd__",
        //     py::overload_cast<double>(&raxis::Tensor::add_),
        //     py::arg("other"))

        .def("__isub__",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::sub_, py::const_),
            py::arg("other"))
        // .def("__isub__",
        //     py::overload_cast<double>(&raxis::Tensor::sub_),
        //     py::arg("other"))

        .def("__imul__",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::mul_, py::const_),
            py::arg("other"))
        // .def("__imul__",
        //     py::overload_cast<double>(&raxis::Tensor::mul_),
        //     py::arg("other"))

        .def("__itruediv__",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::div_, py::const_),
            py::arg("other"))
        // .def("__itruediv__",
        //     py::overload_cast<double>(&raxis::Tensor::div_),
        //     py::arg("other"))

        // Reverse ops
        .def("__radd__",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::add, py::const_),
            py::arg("other"))
        // .def("__radd__",
        //     py::overload_cast<double>(&raxis::Tensor::add, py::const_),
        //     py::arg("other"))

        .def("__rsub__",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::rsub, py::const_),
            py::arg("other"))
        // .def("__rsub__",
        //     py::overload_cast<double>(&raxis::Tensor::rsub, py::const_),
        //     py::arg("other"))

        .def("__rmul__",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::mul, py::const_),
            py::arg("other"))
        // .def("__rmul__",
        //     py::overload_cast<double>(&raxis::Tensor::mul, py::const_),
        //     py::arg("other"))

        .def("__rtruediv__",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::rdiv, py::const_),
            py::arg("other"))
        // .def("__rtruediv__",
        //     py::overload_cast<double>(&raxis::Tensor::rdiv, py::const_),
        //     py::arg("other"))
        
        // Functions to access operations
        .def("copy_",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::copy_, py::const_),
            py::arg("other"))
        .def("to",
            py::overload_cast<raxis::Dtype>(&raxis::Tensor::to, py::const_),
            py::arg("dtype"))
        

        .def("add",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::add, py::const_),
            py::arg("other"))
        // .def("add",
        //     py::overload_cast<double>(&raxis::Tensor::add, py::const_),
        //     py::arg("other"))

        .def("sub",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::sub, py::const_),
            py::arg("other"))
        // .def("sub",
        //     py::overload_cast<double>(&raxis::Tensor::sub, py::const_),
        //     py::arg("other"))

        .def("mul",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::mul, py::const_),
            py::arg("other"))
        // .def("mul",
        //     py::overload_cast<double>(&raxis::Tensor::mul, py::const_),
        //     py::arg("other"))

        .def("div",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::div, py::const_),
            py::arg("other"))
        // .def("div",
        //     py::overload_cast<double>(&raxis::Tensor::div, py::const_),
        //     py::arg("other"))

        .def("add_",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::add_, py::const_),
            py::arg("other"))
        // .def("add_",
        //     py::overload_cast<double>(&raxis::Tensor::add_, py::const_),
        //     py::arg("other"))

        .def("sub_",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::sub_, py::const_),
            py::arg("other"))
        // .def("sub_",
        //     py::overload_cast<double>(&raxis::Tensor::sub_, py::const_),
        //     py::arg("other"))

        .def("mul_",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::mul_, py::const_),
            py::arg("other"))
        // .def("mul_",
        //     py::overload_cast<double>(&raxis::Tensor::mul_, py::const_),
        //     py::arg("other"))

        .def("div_",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::div_, py::const_),
            py::arg("other"))
        // .def("div_",
        //     py::overload_cast<double>(&raxis::Tensor::div_, py::const_),
        //     py::arg("other"))

        // Reversed
        .def("rsub",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::rsub, py::const_),
            py::arg("other"))
        // .def("rsub",
        //     py::overload_cast<double>(&raxis::Tensor::rsub, py::const_),
        //     py::arg("other"))

        .def("rdiv",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::rdiv, py::const_),
            py::arg("other"))
        // .def("rdiv",
        //     py::overload_cast<double>(&raxis::Tensor::rdiv, py::const_),
        //     py::arg("other"))

        .def("rsub_",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::rsub_, py::const_),
            py::arg("other"))
        // .def("rsub_",
        //     py::overload_cast<double>(&raxis::Tensor::rsub_, py::const_),
        //     py::arg("other"))

        .def("rdiv_",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::rdiv_, py::const_),
            py::arg("other"))
        // .def("rdiv_",
        //     py::overload_cast<double>(&raxis::Tensor::rdiv_, py::const_),
        //     py::arg("other"))

        ;
    
    // Initializers
    // c.def("array", [](PyTensorData data, bool requires_grad, PyDevice device, PyDtype dtype) {
    //     torch::Tensor Tensor = data.get().to(device.get()).to(dtype.get());
    //     return raxis::Tensor(Tensor, requires_grad);
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
    //     torch::TensorOptions options = torch::TensorOptions();
    //     options = options.device(device.get()).dtype(dtype.get());
        
    //     return raxis::full(shape.get(), fill_value, requires_grad, options);
    // }, py::arg("shape"), py::arg("fill_value"), py::arg("requires_grad") = false, py::arg("device") = py::none(), py::arg("dtype") = py::none());
       

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
        "sub", 
        py::overload_cast<const raxis::Tensor&, const raxis::Tensor&>(&raxis::sub), 
        py::arg("t1"), py::arg("t2"));

    c.def(
        "mul", 
        py::overload_cast<const raxis::Tensor&, const raxis::Tensor&>(&raxis::mul), 
        py::arg("t1"), py::arg("t2"));
        
    c.def("div", 
        py::overload_cast<const raxis::Tensor&, const raxis::Tensor&>(&raxis::div), 
        py::arg("t1"), py::arg("t2"));

    c.def(
        "rsub", 
        py::overload_cast<const raxis::Tensor&, const raxis::Tensor&>(&raxis::rsub), 
        py::arg("t1"), py::arg("t2"));
        
    c.def("rdiv", 
        py::overload_cast<const raxis::Tensor&, const raxis::Tensor&>(&raxis::rdiv), 
        py::arg("t1"), py::arg("t2"));
}
