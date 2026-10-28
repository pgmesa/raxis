#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <cub/device/device_radix_sort.cuh>

#include <raxis/core/device_guard.h>
#include <raxis/cuda/cuda_exception.h>
#include <raxis/cuda/cuda_functions.h>
#include <raxis/ops/cuda/integer_divider.cuh>
#include <raxis/ops/cuda/offset_calculator.cuh>
#include <raxis/ops/cuda/loops.cuh>
#include <raxis/ops/tensor_advanced_indexing.h>
#include <raxis/ops/tensor_transformations.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/tensor.h>
#include <raxis/tensor/tensor_iterator.h>


// Advanced indexing, index_select and masked_fill on the GPU.
// From: aten/src/ATen/native/cuda/IndexKernel.cu (gpu_index_kernel,
// masked_fill_kernel), Indexing.cu (index_select_out_cuda_impl)

namespace raxis {
namespace {

constexpr int kThreads = 256;

// The CPU's index layout (see ops/cpu/index_kernel.cpp): operand 0 is the
// output, 1 the input, 2.. the index tensors; self is restrided with stride 0
// over the indexed dims and every element adds the offset its index values
// select. PyTorch reads every index tensor with the strides of the first one
// (they are broadcast to the same shape); here each gets its own, so the
// offset calculator takes the number of indices as a template argument.
constexpr int kMaxIndices = 4;

template <int num_indices>
struct IndexOffsets {
    char* index_ptrs[num_indices];
    int64_t sizes[num_indices];
    int64_t strides[num_indices];

    // Byte offset selected by element `offsets` of the index tensors
    __device__ int64_t get(const int32_t* offsets) const {
        int64_t offset = 0;
        #pragma unroll
        for (int i = 0; i < num_indices; i++) {
            int64_t index = *reinterpret_cast<const int64_t*>(index_ptrs[i] + offsets[i]);
            RAXIS_CUDA_KERNEL_ASSERT(index >= -sizes[i] && index < sizes[i] && "index out of bounds");
            if (index < 0) {
                index += sizes[i];
            }
            offset += index * strides[i];
        }
        return offset;
    }
};

// f(out, in, offset) for every element: the offset applies to the input
// (index) or to the output (index_put), the functor decides
template <int num_indices, typename func_t>
void gpu_index_kernel_impl(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride, const func_t& f) {
    IndexOffsets<num_indices> index;
    for (int i = 0; i < num_indices; i++) {
        index.index_ptrs[i] = static_cast<char*>(iter.data_ptr(i + 2));
        index.sizes[i] = index_size[i];
        index.strides[i] = index_stride[i];
    }
    char* out_ptr = static_cast<char*>(iter.data_ptr(0));
    char* in_ptr = static_cast<char*>(iter.data_ptr(1));
    auto offset_calc = cuda::make_offset_calculator<num_indices + 2>(iter);
    cuda::launch_legacy_kernel<128, 4>(iter.numel(), [=] GPU_LAMBDA(int idx) {
        const auto offsets = offset_calc.get(idx);
        f(out_ptr + offsets[0], in_ptr + offsets[1], index.get(&offsets[2]));
    });
}

template <typename func_t>
void gpu_index_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride, const func_t& f) {
    if (iter.numel() == 0) {
        return;
    }
    if (!iter.can_use_32bit_indexing()) {
        auto first_half = iter.split(iter.get_dim_to_split());
        gpu_index_kernel(*first_half, index_size, index_stride, f);
        gpu_index_kernel(iter, index_size, index_stride, f);
        return;
    }
    DeviceGuard guard(iter.device(0));
    switch (index_size.size()) {
        case 1: return gpu_index_kernel_impl<1>(iter, index_size, index_stride, f);
        case 2: return gpu_index_kernel_impl<2>(iter, index_size, index_stride, f);
        case 3: return gpu_index_kernel_impl<3>(iter, index_size, index_stride, f);
        case 4: return gpu_index_kernel_impl<4>(iter, index_size, index_stride, f);
        default:
            RAXIS_CHECK_NOT_IMPLEMENTED(false,
                "indexing with more than ", kMaxIndices, " index tensors is not supported on CUDA yet");
    }
}

// The functors get the output and input element and the offset the indices
// select (extended lambdas cannot be defined inside the dtype dispatch)
template <typename scalar_t>
struct IndexFunctor {
    __device__ void operator()(char* out, char* in, int64_t offset) const {
        *reinterpret_cast<scalar_t*>(out) = *reinterpret_cast<const scalar_t*>(in + offset);
    }
};

template <typename scalar_t>
struct IndexPutFunctor {
    __device__ void operator()(char* out, char* in, int64_t offset) const {
        *reinterpret_cast<scalar_t*>(out + offset) = *reinterpret_cast<const scalar_t*>(in);
    }
};

template <typename scalar_t>
struct IndexPutAccumulateFunctor {
    __device__ void operator()(char* out, char* in, int64_t offset) const {
        auto& dst = *reinterpret_cast<scalar_t*>(out + offset);
        const scalar_t value = *reinterpret_cast<const scalar_t*>(in);
        // A sum of bools is their OR
        if constexpr (std::is_same_v<scalar_t, bool>) {
            dst = dst || value;
        } else {
            dst += value;
        }
    }
};

void index_kernel_cuda(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride) {
    iter.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        gpu_index_kernel(iter, index_size, index_stride, IndexFunctor<scalar_t>{});
    });
}

// Accumulate with positions that may repeat, deterministically.
// From: aten/src/ATen/native/cuda/Indexing.cu (index_put_with_sort_kernel,
// indexing_backward_kernel)
//
// Atomic adds would sum the values of a repeated position in whatever order
// the threads run, so the result could change between runs. PyTorch sorts
// the indices instead with a stable sort (equal indices keep their order),
// then one warp per distinct index adds its values in that order. Here the
// sort is per element rather than per index: every element of the iterator
// gets as key the byte offset it writes in self, the stable radix sort
// groups equal keys in iteration order, and the first thread of each group
// adds the group's values one after the other. Sorting elements costs more
// than sorting indices when an index selects a whole row, but it needs no
// linear index and no transposed copy of self, and a row is still spread
// over many threads.
template <int num_indices, typename scalar_t>
void index_put_accumulate_sorted_impl(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride) {
    IndexOffsets<num_indices> index;
    for (int i = 0; i < num_indices; i++) {
        index.index_ptrs[i] = static_cast<char*>(iter.data_ptr(i + 2));
        index.sizes[i] = index_size[i];
        index.strides[i] = index_stride[i];
    }
    char* out_ptr = static_cast<char*>(iter.data_ptr(0));
    const char* in_ptr = static_cast<const char*>(iter.data_ptr(1));
    auto offset_calc = cuda::make_offset_calculator<num_indices + 2>(iter);
    const auto n = static_cast<int>(iter.numel());

    // The largest key bounds the bits the radix sort has to look at
    uint64_t max_key = 0;
    bool negative_stride = false;
    for (int d = 0; d < iter.ndim(); d++) {
        negative_stride |= iter.strides(0)[d] < 0;
        max_key += static_cast<uint64_t>(iter.shape()[d] - 1) * static_cast<uint64_t>(iter.strides(0)[d]);
    }
    for (int i = 0; i < num_indices; i++) {
        negative_stride |= index_stride[i] < 0;
        max_key += static_cast<uint64_t>(index_size[i] - 1) * static_cast<uint64_t>(index_stride[i]);
    }
    const int end_bit = negative_stride ? 64 : std::max(1, static_cast<int>(std::bit_width(max_key)));

    const TensorOptions options = TensorOptions(dtypes::Int64).device(iter.device(0));
    Tensor keys = empty({n}, options);
    Tensor sorted_keys = empty({n}, options);
    Tensor order = empty({n}, options.dtype(dtypes::Int32));
    Tensor sorted_order = empty({n}, options.dtype(dtypes::Int32));
    auto* keys_ptr = reinterpret_cast<uint64_t*>(keys.mutable_data_ptr<int64_t>());
    auto* sorted_keys_ptr = reinterpret_cast<uint64_t*>(sorted_keys.mutable_data_ptr<int64_t>());
    int32_t* order_ptr = order.mutable_data_ptr<int32_t>();
    int32_t* sorted_order_ptr = sorted_order.mutable_data_ptr<int32_t>();

    cuda::launch_legacy_kernel<128, 4>(n, [=] __device__(int i) {
        const auto offsets = offset_calc.get(i);
        keys_ptr[i] = static_cast<uint64_t>(offsets[0] + index.get(&offsets[2]));
        order_ptr[i] = i;
    });
    const cudaStream_t stream = cuda::current_stream();
    size_t temp_bytes = 0;
    RAXIS_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, keys_ptr, sorted_keys_ptr, order_ptr,
                                                     sorted_order_ptr, n, 0, end_bit, stream));
    Tensor temp = empty({static_cast<int64_t>(temp_bytes)}, options.dtype(dtypes::Uint8));
    RAXIS_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(temp.mutable_data_ptr(), temp_bytes, keys_ptr, sorted_keys_ptr,
                                                     order_ptr, sorted_order_ptr, n, 0, end_bit, stream));

    cuda::launch_legacy_kernel<128, 4>(n, [=] __device__(int i) {
        const uint64_t key = sorted_keys_ptr[i];
        if (i > 0 && sorted_keys_ptr[i - 1] == key) {
            return;
        }
        scalar_t& dst = *reinterpret_cast<scalar_t*>(out_ptr + key);
        scalar_t acc = dst;
        for (int j = i; j < n && sorted_keys_ptr[j] == key; j++) {
            const auto offsets = offset_calc.get(sorted_order_ptr[j]);
            const scalar_t value = *reinterpret_cast<const scalar_t*>(in_ptr + offsets[1]);
            if constexpr (std::is_same_v<scalar_t, bool>) {
                acc = acc || value;   // a sum of bools is their OR
            } else {
                acc += value;
            }
        }
        dst = acc;
    });
}

template <typename scalar_t>
void index_put_accumulate_sorted(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride) {
    if (iter.numel() == 0) {
        return;
    }
    if (!iter.can_use_32bit_indexing()) {
        auto first_half = iter.split(iter.get_dim_to_split());
        index_put_accumulate_sorted<scalar_t>(*first_half, index_size, index_stride);
        index_put_accumulate_sorted<scalar_t>(iter, index_size, index_stride);
        return;
    }
    DeviceGuard guard(iter.device(0));
    switch (index_size.size()) {
        case 1: return index_put_accumulate_sorted_impl<1, scalar_t>(iter, index_size, index_stride);
        case 2: return index_put_accumulate_sorted_impl<2, scalar_t>(iter, index_size, index_stride);
        case 3: return index_put_accumulate_sorted_impl<3, scalar_t>(iter, index_size, index_stride);
        case 4: return index_put_accumulate_sorted_impl<4, scalar_t>(iter, index_size, index_stride);
        default:
            RAXIS_CHECK_NOT_IMPLEMENTED(false,
                "indexing with more than ", kMaxIndices, " index tensors is not supported on CUDA yet");
    }
}

// Without accumulate a repeated index may get any of its values, as on the
// CPU. With accumulate, positions that come from a mask cannot repeat and
// each thread adds into its own element; otherwise the sorted path above
void index_put_kernel_cuda(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride,
                           bool accumulate, bool unique_indices) {
    iter.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        if (accumulate && !unique_indices) {
            index_put_accumulate_sorted<scalar_t>(iter, index_size, index_stride);
        } else if (accumulate) {
            gpu_index_kernel(iter, index_size, index_stride, IndexPutAccumulateFunctor<scalar_t>{});
        } else {
            gpu_index_kernel(iter, index_size, index_stride, IndexPutFunctor<scalar_t>{});
        }
    });
}

// A copy: the flip is in the output strides ops::flip set up (negative
// along the flipped dims), and the dummy third operand is never read, so
// only the first two operands get offsets. The copy moves raw bytes of the
// item size.
// From: aten/src/ATen/native/cuda/IndexKernel.cu (flip_kernel_impl)
template <typename T>
void flip_kernel_impl(TensorIterator& iter) {
    if (!iter.can_use_32bit_indexing()) {
        auto first_half = iter.split(iter.get_dim_to_split());
        flip_kernel_impl<T>(*first_half);
        flip_kernel_impl<T>(iter);
        return;
    }
    char* out_ptr = static_cast<char*>(iter.data_ptr(0));
    const char* in_ptr = static_cast<const char*>(iter.data_ptr(1));
    auto offset_calc = cuda::make_offset_calculator<2>(iter);
    cuda::launch_legacy_kernel<128, 4>(iter.numel(), [=] GPU_LAMBDA(int i) {
        const auto offsets = offset_calc.get(i);
        *reinterpret_cast<T*>(out_ptr + offsets[0]) = *reinterpret_cast<const T*>(in_ptr + offsets[1]);
    });
}

void flip_kernel_cuda(TensorIterator& iter) {
    if (iter.numel() == 0) {
        return;
    }
    DeviceGuard guard(iter.device(0));
    switch (iter.element_size(0)) {
        case 1: return flip_kernel_impl<uint8_t>(iter);
        case 2: return flip_kernel_impl<uint16_t>(iter);
        case 4: return flip_kernel_impl<uint32_t>(iter);
        case 8: return flip_kernel_impl<uint64_t>(iter);
        default: RAXIS_CHECK(false, "flip: unsupported item size ", iter.element_size(0));
    }
}

// self viewed as [outer, dim_size, inner] (contiguous), result as
// [outer, num_indices, inner]: one thread per result element. PyTorch picks
// between a kernel per index (large slices) and one per element; per
// element is the general one, and consecutive threads still read
// consecutive elements of a slice
template <typename scalar_t>
__global__ void index_select_kernel(
    scalar_t* result, const scalar_t* self, const int64_t* index, uint32_t n,
    cuda::IntDivider inner_div, cuda::IntDivider num_indices_div, int64_t dim_size
) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const auto k = inner_div.divmod(i);
    const auto j = num_indices_div.divmod(k.div);
    const int64_t src = index[j.mod];
    RAXIS_CUDA_KERNEL_ASSERT(src >= 0 && src < dim_size && "index out of range in self");
    result[i] = self[(static_cast<int64_t>(j.div) * dim_size + src) * inner_div.divisor + k.mod];
}

void index_select_kernel_cuda(const Tensor& result, const Tensor& self_, int64_t dim, const Tensor& index) {
    DeviceGuard guard(self_.device());
    const int64_t n = result.numel();
    if (n == 0) {
        return;
    }
    const Tensor self = self_.contiguous();
    RAXIS_CHECK(n <= std::numeric_limits<int32_t>::max() && self.numel() <= std::numeric_limits<int32_t>::max(),
        "index_select: tensors over 2^31 elements are not supported on CUDA yet");
    int64_t inner = 1;
    for (int64_t d = dim + 1; d < self.ndim(); ++d) {
        inner *= self.shape()[d];
    }
    const int64_t dim_size = self.ndim() == 0 ? 1 : self.shape()[dim];
    const auto inner_div = cuda::IntDivider(static_cast<uint32_t>(inner));
    const auto num_indices_div = cuda::IntDivider(static_cast<uint32_t>(index.numel()));
    const auto blocks = static_cast<unsigned>((n + kThreads - 1) / kThreads);
    self.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        index_select_kernel<scalar_t><<<blocks, kThreads, 0, cuda::current_stream()>>>(
            result.mutable_data_ptr<scalar_t>(), self.data_ptr<scalar_t>(), index.data_ptr<int64_t>(),
            static_cast<uint32_t>(n), inner_div, num_indices_div, dim_size);
        RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
    });
}

template <typename T>
struct MaskedFillFunctor {
    T value;
    __device__ T operator()(T self, bool mask) const { return mask ? value : self; }
};

// The iterator is (self, self, mask): mask ? value : self for every element
void masked_fill_kernel_cuda(TensorIterator& iter, Scalar value) {
    iter.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cuda::gpu_kernel(iter, MaskedFillFunctor<scalar_t>{value.to<scalar_t>()});
    });
}

} // anonymous namespace

REGISTER_DISPATCH(index_stub, &index_kernel_cuda);
REGISTER_DISPATCH(flip_stub, &flip_kernel_cuda);
REGISTER_DISPATCH(index_put_stub, &index_put_kernel_cuda);
REGISTER_DISPATCH(index_select_stub, &index_select_kernel_cuda);
REGISTER_DISPATCH(masked_fill_stub, &masked_fill_kernel_cuda);

} // namespace raxis
