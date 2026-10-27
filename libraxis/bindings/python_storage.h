#pragma once

#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>

#include <raxis/core/allocator.h>
#include <raxis/core/dtype.h>
#include <raxis/util/exception.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/tensor.h>
#include <raxis/tensor/detail/storage_impl.h>

#include "casters.h"


namespace py = pybind11;


namespace raxis::python {

using detail::StorageImpl;

// From: torch/storage.py (UntypedStorage), torch/csrc/StorageMethods.cpp
// The raw bytes behind one or more tensors (a view shares its base's
// storage). Exposed for serialization: raxis.save writes each storage once
// and every tensor as (storage, offset, shape, strides), as torch does.
// The buffer protocol gives numpy a zero-copy view of the bytes (CPU only).
inline void bind_untyped_storage(py::module_& c) {
    py::class_<StorageImpl, std::shared_ptr<StorageImpl>>(c, "UntypedStorage", py::buffer_protocol())
        .def(py::init([](int64_t nbytes) {
            RAXIS_CHECK(nbytes >= 0, "UntypedStorage: negative size ", nbytes);
            const size_t size = static_cast<size_t>(nbytes);
            return std::make_shared<StorageImpl>(get_allocator(DeviceType::CPU)->allocate(size), size);
        }), py::arg("nbytes"))
        .def("nbytes", &StorageImpl::nbytes)
        .def("data_ptr", [](const StorageImpl& self) {
            return reinterpret_cast<uintptr_t>(self.data());
        })
        .def_property_readonly("device", &StorageImpl::device)
        // Address of the StorageImpl: identifies the storage across Python
        // wrappers (torch: UntypedStorage._cdata)
        .def_property_readonly("_cdata", [](const StorageImpl& self) {
            return reinterpret_cast<uintptr_t>(&self);
        })
        .def_buffer([](StorageImpl& self) {
            // numpy would read device memory from the host
            RAXIS_CHECK(self.device().is_cpu(),
                "UntypedStorage: only a CPU storage exposes its bytes, this one is on ", self.device());
            const auto nbytes = static_cast<py::ssize_t>(self.nbytes());
            return py::buffer_info(self.mutable_data(), 1, py::format_descriptor<uint8_t>::format(), 1,
                                   {nbytes}, {py::ssize_t(1)});
        });
}

// Registered after Tensor, which it returns
inline void bind_storage_functions(py::module_& c) {
    // From: torch/_utils.py (_rebuild_tensor)
    c.def("_tensor_from_storage", [](std::shared_ptr<StorageImpl> storage, Dtype dtype, int64_t storage_offset,
                                     Vec size, Vec stride) {
        return from_storage(std::move(storage), storage_offset, size, stride, dtype);
    }, py::arg("storage"), py::arg("dtype"), py::arg("storage_offset"), py::arg("size"), py::arg("stride"));
}

} // namespace raxis::python
