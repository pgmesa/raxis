
#include <memory>
#include <optional>

#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <raxis/raxis.h>
#include <raxis/util/build_config.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/util/parallel.h>
#include <raxis/autograd/grad_mode.h>
#include <raxis/autograd/graph.h>

#include "casters.h"
#include "numpy_utils.h"
#include "python_tensor_factory.h"


namespace py = pybind11;
namespace dtypes = raxis::dtypes;


PYBIND11_MODULE(_C, c) {
    c.doc() = "Raxis Python bindings";
   
    c.def("get_build_config", &raxis::get_build_config, "Get build configuration");
    c.def("get_cpu_capability", []() -> std::string {
        return raxis::cpu_capability_name(raxis::get_cpu_capability());
    });
    c.def("get_num_threads", &raxis::get_num_threads, 
            "Returns the number of threads used for parallelizing CPU operations");
    c.def("set_num_threads", &raxis::set_num_threads, 
            "Sets the number of threads used for intraop parallelism on CPU.");

    c.def("is_grad_enabled", &raxis::autograd::GradMode::is_enabled);
    c.def("set_grad_enabled", &raxis::autograd::GradMode::set_enabled, py::arg("enabled"));

    // Exceptions
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const py::builtin_exception& e) {
            // py::type_error, py::index_error, ...: keep their Python type
            e.set_error();
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

    // Memory formats, as torch.preserve_format and torch.contiguous_format
    py::enum_<raxis::MemoryFormat>(c, "memory_format", py::module_local())
        .value("preserve_format", raxis::MemoryFormat::Preserve)
        .value("contiguous_format", raxis::MemoryFormat::Contiguous)
        .export_values();

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

    py::class_<raxis::autograd::Node, std::shared_ptr<raxis::autograd::Node>>(c, "Node")
        .def("name", &raxis::autograd::Node::name);

    // Tensor
    py::class_<raxis::Tensor>(c, "Tensor")
        // Aliases `data` (shares its TensorImpl). Lets Python subclasses such
        // as nn.Parameter wrap an existing tensor.
        .def(py::init([](const raxis::Tensor& data) { return data; }), py::arg("data"))
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
            RAXIS_CHECK(self.ndim() > 0, "len() of a 0-d tensor");
            return self.shape()[0];
        })
        // Iterates over dim 0, as torch does (iter(self.unbind(0)))
        .def("__iter__", [](const raxis::Tensor& self) {
            if (self.ndim() == 0) throw py::type_error("iteration over a 0-d tensor");
            return py::iter(py::cast(raxis::unbind(self, 0)));
        })
        
        // Autograd
        .def_property("requires_grad",
            &raxis::Tensor::requires_grad,
            [](raxis::Tensor& self, bool requires_grad) {
                self.requires_grad_(requires_grad);
            })

        .def("requires_grad_", &raxis::Tensor::requires_grad_,
            py::arg("requires_grad") = true)

        .def_property("grad",
            [](const raxis::Tensor& self) -> std::optional<raxis::Tensor> {
                raxis::Tensor g = self.grad();
                return g.defined() ? std::optional(g) : std::nullopt;
            },
            [](raxis::Tensor& self, std::optional<raxis::Tensor> value) {
                self.set_grad(value.value_or(raxis::Tensor{}));
            })

        .def_property_readonly("grad_fn",
            [](const raxis::Tensor& self) -> std::optional<raxis::autograd::NodePtr> {
                auto fn = self.grad_fn();
                return fn ? std::optional(fn) : std::nullopt;
            })

        .def_property_readonly("is_leaf", &raxis::Tensor::is_leaf)
        .def_property_readonly("retains_grad", &raxis::Tensor::retains_grad)

        .def("retain_grad", &raxis::Tensor::retain_grad)

        .def("detach", &raxis::Tensor::detach)
        .def("_use_count", &raxis::Tensor::use_count)

        .def("backward",
            [](const raxis::Tensor& self, std::optional<raxis::Tensor> gradient, bool retain_graph) {
                self.backward(gradient.value_or(raxis::Tensor{}), retain_graph);
            },
            py::arg("gradient") = py::none(),
            py::arg("retain_graph") = false)


        .def("__getitem__", [](const raxis::Tensor& self, py::object idx) {
            return self.index(raxis::python::py_index_to_tensor_indices(idx));
        })
        .def("__setitem__", [](raxis::Tensor& self, py::object idx, const raxis::Tensor& value) {
            self.index_put_(raxis::python::py_index_to_tensor_indices(idx), value);
        })
        .def("__setitem__", [](raxis::Tensor& self, py::object idx, raxis::Scalar value) {
            self.index_put_(raxis::python::py_index_to_tensor_indices(idx), value);
        })

        .def("narrow", &raxis::Tensor::narrow,
            py::arg("dim"), py::arg("start"), py::arg("length"))
        .def("select", &raxis::Tensor::select, py::arg("dim"), py::arg("index"))

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
        .def("reshape", [](const raxis::Tensor& self, raxis::VecOrInt shape) {
            return self.reshape(shape);
        }, py::arg("shape"))
        .def("flatten", &raxis::Tensor::flatten,
            py::arg("start_dim") = 0, py::arg("end_dim") = -1)
        .def("broadcast_to", [](const raxis::Tensor& self, raxis::Vec shape) {
            return self.broadcast_to(shape);
        }, py::arg("shape"))
        .def("expand", [](const raxis::Tensor& self, raxis::Vec shape) {
            return self.expand(shape);
        }, py::arg("size"))
        .def("unbind", [](const raxis::Tensor& self, int64_t dim) {
            return raxis::unbind(self, dim);
        }, py::arg("dim") = 0)
        .def("split", [](const raxis::Tensor& self, int64_t split_size, int64_t dim) {
            return raxis::split(self, split_size, dim);
        }, py::arg("split_size"), py::arg("dim") = 0)
        .def("split", [](const raxis::Tensor& self, raxis::Vec split_sizes, int64_t dim) {
            return raxis::split_with_sizes(self, split_sizes, dim);
        }, py::arg("split_size"), py::arg("dim") = 0)
        .def("chunk", [](const raxis::Tensor& self, int64_t chunks, int64_t dim) {
            return raxis::chunk(self, chunks, dim);
        }, py::arg("chunks"), py::arg("dim") = 0)

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
        .def("clone", &raxis::Tensor::clone,
            py::arg_v("memory_format", raxis::MemoryFormat::Preserve, "preserve_format"))
        .def("copy_",
            py::overload_cast<const raxis::Tensor&>(&raxis::Tensor::copy_, py::const_),
            py::arg("other"))
        .def("to",
            py::overload_cast<raxis::Dtype>(&raxis::Tensor::to, py::const_),
            py::arg("dtype"))
        .def("detach",
            py::overload_cast<>(&raxis::Tensor::detach, py::const_))
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
        .def("sign", &raxis::Tensor::sign)
        .def("ceil", &raxis::Tensor::ceil)
        .def("floor", &raxis::Tensor::floor)
        .def("round", &raxis::Tensor::round)
        .def("exp", &raxis::Tensor::exp)
        .def("log", &raxis::Tensor::log)
        .def("sqrt", &raxis::Tensor::sqrt)
        .def("rsqrt", &raxis::Tensor::rsqrt)
        .def("reciprocal", &raxis::Tensor::reciprocal)
        .def("sigmoid", &raxis::Tensor::sigmoid)
        .def("relu", &raxis::Tensor::relu)
        .def("softmax", &raxis::Tensor::softmax, py::arg("dim"))
        .def("log_softmax", &raxis::Tensor::log_softmax, py::arg("dim"))
        .def("logical_not", &raxis::Tensor::logical_not)
        .def("expm1", &raxis::Tensor::expm1)
        .def("log2", &raxis::Tensor::log2)
        .def("log10", &raxis::Tensor::log10)
        .def("log1p", &raxis::Tensor::log1p)
        .def("sin", &raxis::Tensor::sin)
        .def("cos", &raxis::Tensor::cos)
        .def("tan", &raxis::Tensor::tan)
        .def("asin", &raxis::Tensor::asin)
        .def("acos", &raxis::Tensor::acos)
        .def("atan", &raxis::Tensor::atan)
        .def("sinh", &raxis::Tensor::sinh)
        .def("cosh", &raxis::Tensor::cosh)
        .def("tanh", &raxis::Tensor::tanh)
        .def("erf", &raxis::Tensor::erf)
        .def("erfc", &raxis::Tensor::erfc)

        .def("neg_", &raxis::Tensor::neg_)
        .def("abs_", &raxis::Tensor::abs_)
        .def("sign_", &raxis::Tensor::sign_)
        .def("clamp", [](const raxis::Tensor& self, std::optional<raxis::Scalar> min, std::optional<raxis::Scalar> max) {
            return raxis::clamp(self, min, max);
        }, py::arg("min") = py::none(), py::arg("max") = py::none())
        .def("clip", [](const raxis::Tensor& self, std::optional<raxis::Scalar> min, std::optional<raxis::Scalar> max) {
            return raxis::clamp(self, min, max);
        }, py::arg("min") = py::none(), py::arg("max") = py::none())
        .def("clamp_", [](raxis::Tensor& self, std::optional<raxis::Scalar> min, std::optional<raxis::Scalar> max) -> raxis::Tensor& {
            return raxis::clamp_(self, min, max);
        }, py::arg("min") = py::none(), py::arg("max") = py::none())
        .def("clamp_min", &raxis::clamp_min, py::arg("min"))
        .def("clamp_max", &raxis::clamp_max, py::arg("max"))
        // x.where(condition, y) is where(condition, x, y)
        .def("where", [](const raxis::Tensor& self, const raxis::Tensor& condition, const raxis::Tensor& other) {
            return raxis::where(condition, self, other);
        }, py::arg("condition"), py::arg("other"))
        .def("where", [](const raxis::Tensor& self, const raxis::Tensor& condition, raxis::Scalar other) {
            return raxis::where(condition, self, other);
        }, py::arg("condition"), py::arg("other"))
        .def("ceil_", &raxis::Tensor::ceil_)
        .def("floor_", &raxis::Tensor::floor_)
        .def("round_", &raxis::Tensor::round_)
        .def("exp_", &raxis::Tensor::exp_)
        .def("log_", &raxis::Tensor::log_)
        .def("sqrt_", &raxis::Tensor::sqrt_)
        .def("rsqrt_", &raxis::Tensor::rsqrt_)
        .def("reciprocal_", &raxis::Tensor::reciprocal_)
        .def("sigmoid_", &raxis::Tensor::sigmoid_)
        .def("relu_", &raxis::Tensor::relu_)
        .def("logical_not_", &raxis::Tensor::logical_not_)
        .def("expm1_", &raxis::Tensor::expm1_)
        .def("log2_", &raxis::Tensor::log2_)
        .def("log10_", &raxis::Tensor::log10_)
        .def("log1p_", &raxis::Tensor::log1p_)
        .def("sin_", &raxis::Tensor::sin_)
        .def("cos_", &raxis::Tensor::cos_)
        .def("tan_", &raxis::Tensor::tan_)
        .def("asin_", &raxis::Tensor::asin_)
        .def("acos_", &raxis::Tensor::acos_)
        .def("atan_", &raxis::Tensor::atan_)
        .def("sinh_", &raxis::Tensor::sinh_)
        .def("cosh_", &raxis::Tensor::cosh_)
        .def("tanh_", &raxis::Tensor::tanh_)
        .def("erf_", &raxis::Tensor::erf_)
        .def("erfc_", &raxis::Tensor::erfc_)
        
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
        .def("max", [](const raxis::Tensor& self, int64_t dim, bool keepdim) {
            return raxis::max(self, dim, keepdim);
        }, py::arg("dim"), py::arg("keepdim") = false)
        .def("min", [](const raxis::Tensor& self, int64_t dim, bool keepdim) {
            return raxis::min(self, dim, keepdim);
        }, py::arg("dim"), py::arg("keepdim") = false)

        .def("gather", &raxis::gather, py::arg("dim"), py::arg("index"))
        .def("scatter",
            py::overload_cast<const raxis::Tensor&, int64_t, const raxis::Tensor&, const raxis::Tensor&>(&raxis::scatter),
            py::arg("dim"), py::arg("index"), py::arg("src"))
        .def("scatter",
            py::overload_cast<const raxis::Tensor&, int64_t, const raxis::Tensor&, raxis::Scalar>(&raxis::scatter),
            py::arg("dim"), py::arg("index"), py::arg("value"))
        .def("scatter_", [](raxis::Tensor& self, int64_t dim, const raxis::Tensor& index, const raxis::Tensor& src) -> raxis::Tensor& {
            return raxis::scatter_(self, dim, index, src);
        }, py::arg("dim"), py::arg("index"), py::arg("src"))
        .def("scatter_", [](raxis::Tensor& self, int64_t dim, const raxis::Tensor& index, raxis::Scalar value) -> raxis::Tensor& {
            return raxis::scatter_(self, dim, index, value);
        }, py::arg("dim"), py::arg("index"), py::arg("value"))
        .def("scatter_add", &raxis::scatter_add, py::arg("dim"), py::arg("index"), py::arg("src"))
        .def("scatter_add_", [](raxis::Tensor& self, int64_t dim, const raxis::Tensor& index, const raxis::Tensor& src) -> raxis::Tensor& {
            return raxis::scatter_add_(self, dim, index, src);
        }, py::arg("dim"), py::arg("index"), py::arg("src"))
        .def("index_select", &raxis::index_select, py::arg("dim"), py::arg("index"))
        .def("index_add", &raxis::index_add, py::arg("dim"), py::arg("index"), py::arg("source"))
        .def("index_add_", [](raxis::Tensor& self, int64_t dim, const raxis::Tensor& index, const raxis::Tensor& source) -> raxis::Tensor& {
            return raxis::index_add_(self, dim, index, source);
        }, py::arg("dim"), py::arg("index"), py::arg("source"))
        .def("argmax", [](const raxis::Tensor& self, std::optional<int64_t> dim, bool keepdim) {
            return raxis::argmax(self, dim, keepdim);
        }, py::arg("dim") = py::none(), py::arg("keepdim") = false)
        .def("argmin", [](const raxis::Tensor& self, std::optional<int64_t> dim, bool keepdim) {
            return raxis::argmin(self, dim, keepdim);
        }, py::arg("dim") = py::none(), py::arg("keepdim") = false)
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
    c.def("tensor", [](py::object data, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device, bool requires_grad) {
        raxis::Tensor t;
        if (raxis::python::is_numpy_array(data)) {
            RAXIS_CHECK(!device || device->is_cpu(),
                "tensor(): only CPU tensors can be created from a numpy array");
            // tensor() always copies (from_numpy is the aliasing variant)
            raxis::Tensor src = raxis::python::tensor_from_numpy(data.cast<py::array>());
            t = dtype && *dtype != src.dtype() ? src.to(*dtype) : src.clone();
        } else {
            t = raxis::python::tensor_from_python(data, dtype, device);
        }
        if (requires_grad) t.requires_grad_(true);
        return t;
    },
    py::arg("data"),
    py::arg("dtype") = py::none(),
    py::arg("device") = py::none(),
    py::arg("requires_grad") = false,
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

    c.def("empty", [](raxis::VecOrInt shape, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device, bool requires_grad) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device).requires_grad(requires_grad);
        return raxis::empty(shape, options);
    }, py::arg("shape"), py::arg("dtype") = py::none(), py::arg("device") = py::none(), py::arg("requires_grad") = false);

    c.def("empty_like", [](const raxis::Tensor& other, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device, bool requires_grad, raxis::MemoryFormat memory_format) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device).requires_grad(requires_grad);
        return raxis::empty_like(other, options, memory_format);
    }, py::arg("other"), py::arg("dtype") = py::none(), py::arg("device") = py::none(), py::arg("requires_grad") = false,
       py::arg_v("memory_format", raxis::MemoryFormat::Preserve, "preserve_format"));

    c.def("zeros", [](raxis::VecOrInt shape, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device, bool requires_grad) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device).requires_grad(requires_grad);
        return raxis::zeros(shape, options);
    }, py::arg("shape"), py::arg("dtype") = py::none(), py::arg("device") = py::none(), py::arg("requires_grad") = false);

    c.def("zeros_like", [](const raxis::Tensor& other, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device, bool requires_grad, raxis::MemoryFormat memory_format) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device).requires_grad(requires_grad);
        return raxis::zeros_like(other, options, memory_format);
    }, py::arg("other"), py::arg("dtype") = py::none(), py::arg("device") = py::none(), py::arg("requires_grad") = false,
       py::arg_v("memory_format", raxis::MemoryFormat::Preserve, "preserve_format"));

    c.def("ones", [](raxis::VecOrInt shape, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device, bool requires_grad) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device).requires_grad(requires_grad);
        return raxis::ones(shape, options);
    }, py::arg("shape"), py::arg("dtype") = py::none(), py::arg("device") = py::none(), py::arg("requires_grad") = false);

    c.def("ones_like", [](const raxis::Tensor& other, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device, bool requires_grad, raxis::MemoryFormat memory_format) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device).requires_grad(requires_grad);
        return raxis::ones_like(other, options, memory_format);
    }, py::arg("other"), py::arg("dtype") = py::none(), py::arg("device") = py::none(), py::arg("requires_grad") = false,
       py::arg_v("memory_format", raxis::MemoryFormat::Preserve, "preserve_format"));

    c.def("full", [](raxis::VecOrInt shape, raxis::Scalar fill_value, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device, bool requires_grad) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device).requires_grad(requires_grad);
        return raxis::full(shape, fill_value, options);
    }, py::arg("shape"), py::arg("fill_value"), py::arg("dtype") = py::none(), py::arg("device") = py::none(), py::arg("requires_grad") = false);

    c.def("full_like", [](const raxis::Tensor& other, raxis::Scalar fill_value, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device, bool requires_grad, raxis::MemoryFormat memory_format) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device).requires_grad(requires_grad);
        return raxis::full_like(other, fill_value, options, memory_format);
    }, py::arg("other"), py::arg("fill_value"), py::arg("dtype") = py::none(), py::arg("device") = py::none(), py::arg("requires_grad") = false,
       py::arg_v("memory_format", raxis::MemoryFormat::Preserve, "preserve_format"));

    c.def("rand", [](raxis::VecOrInt shape, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device, bool requires_grad) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device).requires_grad(requires_grad);
        return raxis::rand(shape, options);
    }, py::arg("shape"), py::arg("dtype") = py::none(), py::arg("device") = py::none(), py::arg("requires_grad") = false);

    c.def("rand_like", [](const raxis::Tensor& other, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device, bool requires_grad, raxis::MemoryFormat memory_format) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device).requires_grad(requires_grad);
        return raxis::rand_like(other, options, memory_format);
    }, py::arg("other"), py::arg("dtype") = py::none(), py::arg("device") = py::none(), py::arg("requires_grad") = false,
       py::arg_v("memory_format", raxis::MemoryFormat::Preserve, "preserve_format"));

    c.def("randn", [](raxis::VecOrInt shape, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device, bool requires_grad) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device).requires_grad(requires_grad);
        return raxis::randn(shape, options);
    }, py::arg("shape"), py::arg("dtype") = py::none(), py::arg("device") = py::none(), py::arg("requires_grad") = false);

    c.def("randn_like", [](const raxis::Tensor& other, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device, bool requires_grad, raxis::MemoryFormat memory_format) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device).requires_grad(requires_grad);
        return raxis::randn_like(other, options, memory_format);
    }, py::arg("other"), py::arg("dtype") = py::none(), py::arg("device") = py::none(), py::arg("requires_grad") = false,
       py::arg_v("memory_format", raxis::MemoryFormat::Preserve, "preserve_format"));

    c.def("randint", [](int64_t low, int64_t high, raxis::VecOrInt shape, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device);
        return raxis::randint(low, high, shape, options);
    }, py::arg("low"), py::arg("high"), py::arg("shape"), py::arg("dtype") = py::none(), py::arg("device") = py::none());

    c.def("randint_like", [](const raxis::Tensor& other, int64_t low, int64_t high, std::optional<raxis::Dtype> dtype, std::optional<raxis::Device> device, raxis::MemoryFormat memory_format) {
        auto options = raxis::TensorOptions().dtype(dtype).device(device);
        return raxis::randint_like(other, low, high, options, memory_format);
    }, py::arg("other"), py::arg("low"), py::arg("high"), py::arg("dtype") = py::none(), py::arg("device") = py::none(),
       py::arg_v("memory_format", raxis::MemoryFormat::Preserve, "preserve_format"));

    c.def("manual_seed", &raxis::manual_seed, py::arg("seed"));

    // Shape operations
    c.def("squeeze", &raxis::squeeze, py::arg("input"), py::arg("dim") = py::none());
    c.def("unsqueeze", &raxis::unsqueeze, py::arg("input"), py::arg("dim"));
    c.def("select", py::overload_cast<const raxis::Tensor&, int64_t, int64_t>(&raxis::select), py::arg("input"), py::arg("dim"), py::arg("index"));
    c.def("transpose", &raxis::transpose, py::arg("input"), py::arg("dim0"), py::arg("dim1"));
    c.def("permute", &raxis::permute, py::arg("input"), py::arg("dims"));
    c.def("swapdims", &raxis::swapdims, py::arg("input"), py::arg("dim0"), py::arg("dim1"));
    c.def("movedim",
        py::overload_cast<const raxis::Tensor&, int64_t, int64_t>(&raxis::movedim),
        py::arg("input"), py::arg("source"), py::arg("destination"));
    
    c.def("movedim", [](const raxis::Tensor& input, raxis::VecOrInt source, raxis::VecOrInt destination) {
        return raxis::movedim(input, source, destination);
    }, py::arg("input"), py::arg("source"), py::arg("destination"));
    c.def("reshape", [](const raxis::Tensor& input, raxis::VecOrInt shape) {
        return raxis::reshape(input, shape);
    }, py::arg("input"), py::arg("shape"));
    c.def("flatten", &raxis::flatten,
        py::arg("input"), py::arg("start_dim") = 0, py::arg("end_dim") = -1);
    c.def("broadcast_to", [](const raxis::Tensor& input, raxis::Vec shape) {
        return raxis::broadcast_to(input, shape);
    }, py::arg("input"), py::arg("shape"));

    c.def("broadcast_tensors", [](std::vector<raxis::Tensor> tensors) {
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

    c.def("cat", [](std::vector<raxis::Tensor> tensors, int64_t dim) {
        return raxis::cat(tensors, dim);
    }, py::arg("tensors"), py::arg("dim") = 0);
    c.def("concat", [](std::vector<raxis::Tensor> tensors, int64_t dim) {
        return raxis::cat(tensors, dim);
    }, py::arg("tensors"), py::arg("dim") = 0);
    c.def("stack", [](std::vector<raxis::Tensor> tensors, int64_t dim) {
        return raxis::stack(tensors, dim);
    }, py::arg("tensors"), py::arg("dim") = 0);
    c.def("unbind", &raxis::unbind, py::arg("input"), py::arg("dim") = 0);
    c.def("split", [](const raxis::Tensor& input, int64_t split_size, int64_t dim) {
        return raxis::split(input, split_size, dim);
    }, py::arg("tensor"), py::arg("split_size_or_sections"), py::arg("dim") = 0);
    c.def("split", [](const raxis::Tensor& input, raxis::Vec split_sizes, int64_t dim) {
        return raxis::split_with_sizes(input, split_sizes, dim);
    }, py::arg("tensor"), py::arg("split_size_or_sections"), py::arg("dim") = 0);
    c.def("chunk", &raxis::chunk, py::arg("input"), py::arg("chunks"), py::arg("dim") = 0);

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
    c.def("sign", py::overload_cast<const raxis::Tensor&>(&raxis::sign), py::arg("input"));
    c.def("clamp", [](const raxis::Tensor& input, std::optional<raxis::Scalar> min, std::optional<raxis::Scalar> max) {
        return raxis::clamp(input, min, max);
    }, py::arg("input"), py::arg("min") = py::none(), py::arg("max") = py::none());
    c.def("clip", [](const raxis::Tensor& input, std::optional<raxis::Scalar> min, std::optional<raxis::Scalar> max) {
        return raxis::clamp(input, min, max);
    }, py::arg("input"), py::arg("min") = py::none(), py::arg("max") = py::none());
    c.def("clamp_min", &raxis::clamp_min, py::arg("input"), py::arg("min"));
    c.def("clamp_max", &raxis::clamp_max, py::arg("input"), py::arg("max"));
    c.def("where", py::overload_cast<const raxis::Tensor&, const raxis::Tensor&, const raxis::Tensor&>(&raxis::where),
        py::arg("condition"), py::arg("input"), py::arg("other"));
    c.def("where", py::overload_cast<const raxis::Tensor&, const raxis::Tensor&, raxis::Scalar>(&raxis::where),
        py::arg("condition"), py::arg("input"), py::arg("other"));
    c.def("where", py::overload_cast<const raxis::Tensor&, raxis::Scalar, const raxis::Tensor&>(&raxis::where),
        py::arg("condition"), py::arg("input"), py::arg("other"));
    c.def("where", py::overload_cast<const raxis::Tensor&, raxis::Scalar, raxis::Scalar>(&raxis::where),
        py::arg("condition"), py::arg("input"), py::arg("other"));
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
    c.def("relu",
        py::overload_cast<const raxis::Tensor&>(&raxis::relu),
        py::arg("input"));
    c.def("relu_",
        py::overload_cast<raxis::Tensor&>(&raxis::relu_),
        py::arg("input"));
    c.def("softmax", &raxis::softmax, py::arg("input"), py::arg("dim"));
    c.def("log_softmax", &raxis::log_softmax, py::arg("input"), py::arg("dim"));
    c.def("dropout", &raxis::dropout,
        py::arg("input"), py::arg("p") = 0.5, py::arg("train") = true);
    c.def("logical_not",
        py::overload_cast<const raxis::Tensor&>(&raxis::logical_not),
        py::arg("input"));
    c.def("expm1",
        py::overload_cast<const raxis::Tensor&>(&raxis::expm1),
        py::arg("input"));
    c.def("log2",
        py::overload_cast<const raxis::Tensor&>(&raxis::log2),
        py::arg("input"));
    c.def("log10",
        py::overload_cast<const raxis::Tensor&>(&raxis::log10),
        py::arg("input"));
    c.def("log1p",
        py::overload_cast<const raxis::Tensor&>(&raxis::log1p),
        py::arg("input"));
    c.def("sin",
        py::overload_cast<const raxis::Tensor&>(&raxis::sin),
        py::arg("input"));
    c.def("cos",
        py::overload_cast<const raxis::Tensor&>(&raxis::cos),
        py::arg("input"));
    c.def("tan",
        py::overload_cast<const raxis::Tensor&>(&raxis::tan),
        py::arg("input"));
    c.def("asin",
        py::overload_cast<const raxis::Tensor&>(&raxis::asin),
        py::arg("input"));
    c.def("acos",
        py::overload_cast<const raxis::Tensor&>(&raxis::acos),
        py::arg("input"));
    c.def("atan",
        py::overload_cast<const raxis::Tensor&>(&raxis::atan),
        py::arg("input"));
    c.def("sinh",
        py::overload_cast<const raxis::Tensor&>(&raxis::sinh),
        py::arg("input"));
    c.def("cosh",
        py::overload_cast<const raxis::Tensor&>(&raxis::cosh),
        py::arg("input"));
    c.def("tanh",
        py::overload_cast<const raxis::Tensor&>(&raxis::tanh),
        py::arg("input"));
    c.def("erf",
        py::overload_cast<const raxis::Tensor&>(&raxis::erf),
        py::arg("input"));
    c.def("erfc",
        py::overload_cast<const raxis::Tensor&>(&raxis::erfc),
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

    c.def("max", [](const raxis::Tensor& input, int64_t dim, bool keepdim) {
        return raxis::max(input, dim, keepdim);
    }, py::arg("input"), py::arg("dim"), py::arg("keepdim") = false);
    c.def("min", [](const raxis::Tensor& input, int64_t dim, bool keepdim) {
        return raxis::min(input, dim, keepdim);
    }, py::arg("input"), py::arg("dim"), py::arg("keepdim") = false);
    c.def("max", py::overload_cast<const raxis::Tensor&>(&raxis::max),
    py::arg("input"));
    c.def("min", py::overload_cast<const raxis::Tensor&>(&raxis::min),
        py::arg("input"));

    c.def("gather", &raxis::gather, py::arg("input"), py::arg("dim"), py::arg("index"));
    c.def("scatter",
        py::overload_cast<const raxis::Tensor&, int64_t, const raxis::Tensor&, const raxis::Tensor&>(&raxis::scatter),
        py::arg("input"), py::arg("dim"), py::arg("index"), py::arg("src"));
    c.def("scatter",
        py::overload_cast<const raxis::Tensor&, int64_t, const raxis::Tensor&, raxis::Scalar>(&raxis::scatter),
        py::arg("input"), py::arg("dim"), py::arg("index"), py::arg("value"));
    c.def("scatter_add", &raxis::scatter_add, py::arg("input"), py::arg("dim"), py::arg("index"), py::arg("src"));
    c.def("index_select", &raxis::index_select, py::arg("input"), py::arg("dim"), py::arg("index"));
    c.def("index_add", &raxis::index_add, py::arg("input"), py::arg("dim"), py::arg("index"), py::arg("source"));
    c.def("argmax", [](const raxis::Tensor& input, std::optional<int64_t> dim, bool keepdim) {
        return raxis::argmax(input, dim, keepdim);
    }, py::arg("input"), py::arg("dim") = py::none(), py::arg("keepdim") = false);
    c.def("argmin", [](const raxis::Tensor& input, std::optional<int64_t> dim, bool keepdim) {
        return raxis::argmin(input, dim, keepdim);
    }, py::arg("input"), py::arg("dim") = py::none(), py::arg("keepdim") = false);
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

    // nn functions that torch exposes through torch._C._nn rather than at the
    // top level; raxis.nn.functional wraps them
    py::module_ nn = c.def_submodule("_nn");

    nn.def("linear", [](const raxis::Tensor& input, const raxis::Tensor& weight,
                        std::optional<raxis::Tensor> bias) {
        return raxis::linear(input, weight, bias.value_or(raxis::Tensor{}));
    }, py::arg("input"), py::arg("weight"), py::arg("bias") = py::none());

    nn.def("nll_loss", [](const raxis::Tensor& input, const raxis::Tensor& target,
                          const std::string& reduction, int64_t ignore_index) {
        return raxis::nll_loss(input, target, raxis::parse_reduction(reduction), ignore_index);
    }, py::arg("input"), py::arg("target"), py::arg("reduction") = "mean",
       py::arg("ignore_index") = -100);

    nn.def("cross_entropy", [](const raxis::Tensor& input, const raxis::Tensor& target,
                               const std::string& reduction, int64_t ignore_index) {
        return raxis::cross_entropy(input, target, raxis::parse_reduction(reduction), ignore_index);
    }, py::arg("input"), py::arg("target"), py::arg("reduction") = "mean",
       py::arg("ignore_index") = -100);

    nn.def("mse_loss", [](const raxis::Tensor& input, const raxis::Tensor& target,
                          const std::string& reduction) {
        return raxis::mse_loss(input, target, raxis::parse_reduction(reduction));
    }, py::arg("input"), py::arg("target"), py::arg("reduction") = "mean");

    nn.def("binary_cross_entropy", [](const raxis::Tensor& input, const raxis::Tensor& target,
                                      const std::string& reduction) {
        return raxis::binary_cross_entropy(input, target, raxis::parse_reduction(reduction));
    }, py::arg("input"), py::arg("target"), py::arg("reduction") = "mean");

    nn.def("binary_cross_entropy_with_logits", [](const raxis::Tensor& input,
                                                  const raxis::Tensor& target,
                                                  const std::string& reduction) {
        return raxis::binary_cross_entropy_with_logits(
            input, target, raxis::parse_reduction(reduction));
    }, py::arg("input"), py::arg("target"), py::arg("reduction") = "mean");

} // pybind11_module
