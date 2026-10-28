#include <cstdint>
#include <type_traits>

#include <raxis/core/device_guard.h>
#include <raxis/core/scalar.h>
#include <raxis/cuda/cuda_exception.h>
#include <raxis/cuda/cuda_functions.h>
#include <raxis/ops/cuda/loops.cuh>
#include <raxis/ops/cuda/offset_calculator.cuh>
#include <raxis/ops/tensor_advanced_indexing.h>
#include <raxis/tensor/tensor.h>
#include <raxis/tensor/tensor_iterator.h>
#include <raxis/tensor/tensor_shape.h>


// gather / scatter / scatter_add on the GPU.
// From: aten/src/ATen/native/cuda/ScatterGatherKernel.cu
//
// gather:  self[i][j] = src[index[i][j]][j]   (dim = 0)
// scatter: self[index[i][j]][j] = src[i][j]
//
// The CPU kernel squashes dim in its TensorIterator and loops along it
// itself; on the GPU every element of index gets its own thread. The tensor
// read or written through the index (self for scatter, src for gather) is
// restrided to index's shape with stride 0 along dim, so the iterator gives
// the offset of the row, and the thread adds index * stride along dim.
//
// scatter_add adds with atomics, as PyTorch: positions that repeat along
// dim are summed in the order the threads happen to run, so float results
// can change in the last bits between runs (see Known issues). scatter with
// repeated positions keeps any one of the values, as on the CPU.

namespace raxis {
namespace {

int64_t ensure_nonempty_size(const Tensor& t, int64_t dim) {
    return t.ndim() == 0 ? 1 : t.shape()[dim];
}

int64_t ensure_nonempty_stride(const Tensor& t, int64_t dim) {
    return t.ndim() == 0 ? 1 : t.strides()[dim];
}

// t viewed with index's shape: stride 0 along dim (the index picks the
// position there) when restride_dim, t's own strides otherwise
// From: aten/src/ATen/native/TensorAdvancedIndexingUtils.h (restride_dim)
Tensor restrided_as(const Tensor& t, int64_t dim, const Tensor& index, bool restride_dim) {
    DimVector shape = index.ndim() == 0 ? DimVector{1} : DimVector(index.shape());
    DimVector strides = t.ndim() == 0 ? DimVector{1} : DimVector(t.strides());
    if (restride_dim) {
        strides[dim] = 0;
    }
    return make_view(t, t.storage_offset(), shape, strides, /*is_differentiable_view=*/false);
}

// atomicAdd for every dtype: native for 32 and 64-bit types, a compare and
// swap of the aligned 32-bit word for bytes (a sum of bools is their OR)
// From: aten/src/ATen/cuda/Atomic.cuh (AtomicFPOp, AtomicAddIntegerImpl)
template <typename T>
__device__ inline void atomic_add(T* address, T value) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, int32_t>) {
        atomicAdd(address, value);
    } else if constexpr (std::is_same_v<T, int64_t>) {
        // Two's complement: the unsigned sum is the signed one
        atomicAdd(reinterpret_cast<unsigned long long*>(address), static_cast<unsigned long long>(value));
    } else {
        static_assert(sizeof(T) == 1, "atomic_add: unsupported type");
        const auto byte = reinterpret_cast<uintptr_t>(address) % 4;
        auto* word = reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(address) - byte);
        const uint32_t shift = byte * 8;
        uint32_t old = *word;
        uint32_t assumed;
        do {
            assumed = old;
            const auto current = static_cast<uint8_t>(assumed >> shift);
            uint8_t sum;
            if constexpr (std::is_same_v<T, bool>) {
                sum = current || value;
            } else {
                sum = static_cast<uint8_t>(current + value);
            }
            old = atomicCAS(word, assumed, (assumed & ~(0xffu << shift)) | (static_cast<uint32_t>(sum) << shift));
        } while (assumed != old);
    }
}

// The functors get the element of self and of src that one index value
// pairs (extended lambdas cannot be defined inside the dtype dispatch)
template <typename scalar_t>
struct AssignOp {
    __device__ void operator()(scalar_t* self, const scalar_t* src) const {
        *self = *src;
    }
};

template <typename scalar_t>
struct FillOp {
    scalar_t value;
    __device__ void operator()(scalar_t* self, const scalar_t*) const {
        *self = value;
    }
};

template <typename scalar_t>
struct AtomicAddOp {
    __device__ void operator()(scalar_t* self, const scalar_t* src) const {
        atomic_add(self, *src);
    }
};

template <bool is_scatter_like, typename scalar_t, typename func_t>
void scatter_gather_launch(TensorIterator& iter, int64_t index_size, int64_t index_stride, const func_t& f) {
    if (!iter.can_use_32bit_indexing()) {
        auto first_half = iter.split(iter.get_dim_to_split());
        scatter_gather_launch<is_scatter_like, scalar_t>(*first_half, index_size, index_stride, f);
        scatter_gather_launch<is_scatter_like, scalar_t>(iter, index_size, index_stride, f);
        return;
    }
    char* self_ptr = static_cast<char*>(iter.data_ptr(0));
    char* src_ptr = static_cast<char*>(iter.data_ptr(1));
    char* index_ptr = static_cast<char*>(iter.data_ptr(2));
    auto offset_calc = cuda::make_offset_calculator<3>(iter);
    // A device lambda (not GPU_LAMBDA): the bounds assert traps, device only
    cuda::launch_legacy_kernel<128, 4>(iter.numel(), [=] __device__(int i) {
        const auto offsets = offset_calc.get(i);
        const int64_t idx = *reinterpret_cast<const int64_t*>(index_ptr + offsets[2]);
        RAXIS_CUDA_KERNEL_ASSERT(idx >= 0 && idx < index_size && "scatter gather kernel index out of bounds");
        f(reinterpret_cast<scalar_t*>(self_ptr + offsets[0]) + (is_scatter_like ? idx * index_stride : 0),
          reinterpret_cast<const scalar_t*>(src_ptr + offsets[1]) + (is_scatter_like ? 0 : idx * index_stride));
    });
}

// src is undefined for scatter with a value: index stands in as the
// iterator's second operand and the functor never reads it
template <bool is_scatter_like, template <typename> class Op>
void scatter_gather_base_kernel(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src,
                                Scalar value = 0) {
    const Tensor& src_or_index = src.defined() ? src : index;
    const Tensor self_restrided = restrided_as(self, dim, index, is_scatter_like);
    const Tensor src_restrided = src.defined() ? restrided_as(src, dim, index, !is_scatter_like) : index;
    auto iter = TensorIteratorConfig()
        .check_mem_overlap(false)
        .check_all_same_dtype(false)
        .add_output(self_restrided)
        .add_input(src_restrided)
        .add_input(index)
        .build();
    if (iter.numel() == 0) {
        return;
    }
    // Along dim, the index selects in self (scatter) or in src (gather)
    const Tensor& indexed = is_scatter_like ? self : src_or_index;
    const int64_t index_size = ensure_nonempty_size(indexed, dim);
    const int64_t index_stride = ensure_nonempty_stride(indexed, dim);
    DeviceGuard guard(self.device());
    self.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        if constexpr (std::is_same_v<Op<scalar_t>, FillOp<scalar_t>>) {
            scatter_gather_launch<is_scatter_like, scalar_t>(iter, index_size, index_stride,
                                                             FillOp<scalar_t>{value.to<scalar_t>()});
        } else {
            scatter_gather_launch<is_scatter_like, scalar_t>(iter, index_size, index_stride, Op<scalar_t>{});
        }
    });
}

void gather_kernel_cuda(const Tensor& result, int64_t dim, const Tensor& index, const Tensor& src) {
    scatter_gather_base_kernel</*is_scatter_like=*/false, AssignOp>(result, dim, index, src);
}

void scatter_kernel_cuda(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    scatter_gather_base_kernel</*is_scatter_like=*/true, AssignOp>(self, dim, index, src);
}

void scatter_fill_kernel_cuda(const Tensor& self, int64_t dim, const Tensor& index, Scalar value) {
    scatter_gather_base_kernel</*is_scatter_like=*/true, FillOp>(self, dim, index, Tensor{}, value);
}

void scatter_add_kernel_cuda(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src) {
    scatter_gather_base_kernel</*is_scatter_like=*/true, AtomicAddOp>(self, dim, index, src);
}

} // anonymous namespace

REGISTER_DISPATCH(gather_stub, &gather_kernel_cuda);
REGISTER_DISPATCH(scatter_stub, &scatter_kernel_cuda);
REGISTER_DISPATCH(scatter_fill_stub, &scatter_fill_kernel_cuda);
REGISTER_DISPATCH(scatter_add_stub, &scatter_add_kernel_cuda);

} // namespace raxis
