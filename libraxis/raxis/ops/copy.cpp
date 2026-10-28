
#include <raxis/ops/copy.h>

#include <raxis/core/cuda_hooks.h>
#include <raxis/ops/dispatch_stub.h>

#include <raxis/util/exception.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/shape_utils.h>
#include <raxis/tensor/tensor_iterator.h>


namespace raxis {

// From: aten/src/ATen/native/Copy.cpp (copy_impl)
// The only op whose operands may live on different devices: a copy between
// CPU and CUDA runs the CUDA kernel, which moves the bytes with cudaMemcpy
Tensor& copy(const Tensor& a, Tensor& out, bool non_blocking) {
    auto iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .check_all_same_device(false)
        .add_output(out)
        .add_input(a)
        .build();

    DeviceType device_type = iter.device_type(0);
    if (iter.device_type(1) == DeviceType::CUDA) {
        device_type = DeviceType::CUDA;
    }
    copy_stub(device_type, iter, non_blocking);
    out = iter.get_output(0);
    return out;
}

namespace ops {

Tensor clone(const Tensor& self, MemoryFormat memory_format) {
    Tensor out = empty_like(self, {}, memory_format);
    copy(self, out);
    return out;
}

Tensor to_copy(const Tensor& self, Dtype dtype, Device device, bool non_blocking) {
    Tensor out = empty_like(self, TensorOptions().dtype(dtype).device(device));
    copy(self, out, non_blocking);
    return out;
}

// From: aten/src/ATen/native/Memory.cpp (_pin_memory): the same sizes and
// strides over a storage from the pinned allocator
Tensor pin_memory(const Tensor& self) {
    RAXIS_CHECK(self.is_cpu(), "cannot pin a ", self.device(), " tensor: only dense CPU tensors can be pinned");
    const size_t nbytes = compute_numel_from_strides(self.shape(), self.strides()) * self.dtype().itemsize();
    auto storage = std::make_shared<detail::StorageImpl>(
        get_cuda_hooks().pinned_memory_allocator()->allocate(nbytes), nbytes);
    Tensor out = from_storage(std::move(storage), 0, self.shape(), self.strides(), self.dtype());
    copy(self, out);
    return out;
}

} // namespace ops

DEFINE_DISPATCH(copy_stub);

}
