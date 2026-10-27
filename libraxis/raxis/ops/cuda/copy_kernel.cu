#include <cuda_runtime.h>

#include <raxis/autograd/grad_mode.h>
#include <raxis/core/device_guard.h>
#include <raxis/cuda/cuda_exception.h>
#include <raxis/cuda/cuda_functions.h>
#include <raxis/ops/copy.h>
#include <raxis/ops/cuda/loops.cuh>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/tensor_iterator.h>


// copy_ when either side is on the GPU.
// From: aten/src/ATen/native/cuda/Copy.cu
//
// Between CPU and GPU only cudaMemcpy can move the bytes, and it copies one
// contiguous range. So a copy that is not "same dtype and both contiguous"
// first makes both sides contiguous with the same dtype, doing the
// conversion on the CPU side (PyTorch does it there too for blocking
// copies: no extra GPU memory), then copies the range.

namespace raxis {
namespace {

template <typename T>
struct IdentityFunctor {
    __device__ T operator()(T x) const { return x; }
};

/// GPU to GPU on one device: a kernel, or cudaMemcpyAsync when the bytes
/// can be moved as they are. The dtype conversion is in the kernel's casting
/// path (input read as its dtype, stored as the output's)
void copy_device_to_device(TensorIterator& iter) {
    RAXIS_CHECK(iter.device(0) == iter.device(1),
        "copies between two GPUs are not supported yet (", iter.device(1), " -> ", iter.device(0), ")");
    DeviceGuard guard(iter.device(0));
    if (iter.dtype(0) == iter.dtype(1) && iter.is_contiguous()) {
        void* dst = iter.data_ptr(0);
        void* src = iter.data_ptr(1);
        if (dst != src) {
            size_t nbytes = static_cast<size_t>(iter.numel()) * iter.element_size(0);
            RAXIS_CUDA_CHECK(cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToDevice, cuda::current_stream()));
        }
        return;
    }
    iter.dtype(0).dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_kernel(iter, IdentityFunctor<scalar_t>());
    });
}

void copy_kernel_cuda(TensorIterator& iter) {
    RAXIS_INTERNAL_ASSERT(iter.num_operands() == 2);
    Device dst_device = iter.device(0);
    Device src_device = iter.device(1);

    if (dst_device.is_cuda() && src_device.is_cuda()) {
        copy_device_to_device(iter);
        return;
    }

    // CPU <-> GPU. From: copy_requires_temporaries
    if (iter.dtype(0) != iter.dtype(1) || !iter.is_contiguous()) {
        // The temporaries are internal: no graph for them
        autograd::NoGradGuard no_grad;
        Tensor dst = iter.get_tensor(0);
        const Tensor& src = iter.get_tensor(1);
        Tensor dst_contig;
        Tensor src_contig;
        if (src_device.is_cpu()) {
            // To the GPU: convert and lay out on the CPU, then one transfer
            dst_contig = dst.is_contiguous() ? dst : empty_like(dst, {}, MemoryFormat::Contiguous);
            src_contig = src.to(dst.dtype()).expand(dst.shape()).contiguous();
        } else {
            // To the CPU: transfer as it is, then convert on the CPU
            bool same_dtype = dst.dtype() == src.dtype();
            dst_contig = (dst.is_contiguous() && same_dtype)
                ? dst : empty_like(dst, src.dtype(), MemoryFormat::Contiguous);
            src_contig = src.expand(dst.shape()).contiguous();
        }
        // Same dtype and contiguous: the recursive copy takes the memcpy path
        copy(src_contig, dst_contig);
        if (!dst_contig.is_same(dst)) {
            copy(dst_contig, dst);
        }
        return;
    }

    cudaMemcpyKind kind;
    DeviceGuard guard(dst_device.is_cuda() ? dst_device : src_device);
    if (dst_device.is_cuda()) {
        kind = cudaMemcpyHostToDevice;
    } else {
        kind = cudaMemcpyDeviceToHost;
    }
    void* dst = iter.data_ptr(0);
    void* src = iter.data_ptr(1);
    size_t nbytes = static_cast<size_t>(iter.numel()) * iter.element_size(0);
    // Blocking: torch's copy_(non_blocking=False). From pageable host memory
    // the transfer is staged by the driver anyway; pinned memory and
    // non_blocking come later
    cuda::memcpy_and_sync(dst, src, nbytes, kind, cuda::current_stream());
}

} // anonymous namespace

REGISTER_DISPATCH(copy_stub, &copy_kernel_cuda);

} // namespace raxis
