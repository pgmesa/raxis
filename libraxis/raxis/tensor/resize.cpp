#include <raxis/tensor/resize.h>

#include <algorithm>
#include <cstring>

#include <raxis/core/allocator.h>
#include <raxis/core/device_guard.h>
#include <raxis/util/exception.h>
#include <raxis/ops/copy.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/shape_utils.h>
#include <raxis/tensor/detail/tensor_impl.h>
#include <raxis/tensor/detail/storage_impl.h>


namespace raxis {

namespace {

// From: aten/src/ATen/native/Resize.cpp (resize_bytes_cpu) and
// aten/src/ATen/native/cuda/Resize.cpp (resize_bytes_cuda): new memory from
// the storage's allocator, with the old bytes copied over
void resize_bytes(detail::StorageImpl& storage, size_t nbytes) {
    RAXIS_CHECK(storage.resizable(), "Trying to resize storage that is not resizable");

    const Device device = storage.device();
    DeviceGuard guard(device);
    DataPtr new_data = storage.allocator()->allocate(nbytes);
    const size_t copy_bytes = std::min(nbytes, storage.nbytes());
    if (copy_bytes > 0 && storage.data() != nullptr) {
        if (device.is_cpu()) {
            std::memcpy(new_data.mutable_get(), storage.data(), copy_bytes);
        } else {
            // Byte views over both memories, so copy picks the device's path
            const int64_t n = static_cast<int64_t>(copy_bytes);
            auto bytes = [&](void* data) {
                return from_storage(std::make_shared<detail::StorageImpl>(DataPtr(data, device), copy_bytes),
                                    0, {n}, {1}, dtypes::Uint8);
            };
            Tensor dst = bytes(new_data.mutable_get());
            copy(bytes(storage.mutable_data()), dst);
        }
    }
    storage.set_data_ptr(std::move(new_data), nbytes);
}

// From: aten/src/ATen/native/Resize.cpp (resize_impl_cpu_) and
// aten/src/ATen/native/Resize.h (maybe_resize_storage_cpu)
void resize_impl(detail::TensorImpl& self, IntArrayRef shape, IntArrayRef strides) {
    DimVector new_strides = strides.empty() ? compute_strides(shape) : DimVector(strides);
    if (self.shape() == shape && self.strides() == IntArrayRef(new_strides)) return;

    self.set_shape_and_strides(shape, new_strides);
    // An empty tensor needs no memory, whatever its offset
    if (self.numel() == 0) return;

    const size_t itemsize = self.dtype().itemsize();
    const size_t nbytes = (static_cast<size_t>(self.storage_offset()) +
                           compute_numel_from_strides(shape, new_strides)) * itemsize;
    if (nbytes > self.storage()->nbytes()) {
        resize_bytes(*self.storage(), nbytes);
    }
}

} // anonymous namespace

const Tensor& resize_(const Tensor& self, IntArrayRef shape) {
    RAXIS_CHECK(!self.requires_grad(), "cannot resize variables that require grad");
    for (int64_t size : shape) {
        RAXIS_CHECK(size >= 0, "Trying to create tensor with negative dimension ", size, ": ", shape);
    }
    resize_impl(*self.unsafe_get_tensor_impl(), shape, {});
    return self;
}

void resize_output(const Tensor& output, IntArrayRef shape, IntArrayRef strides) {
    if (output.shape() == shape) return;

    RAXIS_CHECK(output.numel() == 0,
        "An output with one or more elements cannot be resized: it has shape ", output.shape(),
        ", which does not match the required output shape ", shape, ". PyTorch resizes it "
        "with a warning and plans to make it an error; raxis already does. Pass an output of "
        "the right shape, or one with zero elements, which is resized: you can reuse an out "
        "tensor t by resizing it, inplace, to zero elements with t.resize_(0).");

    if (output.unsafe_get_tensor_impl()->is_view()) {
        // Resizing in place would write into the memory of the base
        TensorOptions options = output.options().requires_grad(false);
        Tensor fresh = strides.empty() ? empty(shape, options) : empty_strided(shape, strides, options);
        output.unsafe_get_tensor_impl()->shallow_copy_from(*fresh.unsafe_get_tensor_impl());
        return;
    }
    resize_impl(*output.unsafe_get_tensor_impl(), shape, strides);
}

} // namespace raxis
