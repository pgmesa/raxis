#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include <cub/device/device_segmented_radix_sort.cuh>

#include <raxis/core/device_guard.h>
#include <raxis/cuda/cuda_exception.h>
#include <raxis/cuda/cuda_functions.h>
#include <raxis/ops/sorting.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/tensor.h>


// sort and topk on the GPU: a segmented radix sort of every slice (cub).
// From: aten/src/ATen/native/cuda/Sort.cpp, SortImpl.cu (segmented_sort_pairs)
// and SortUtils.cuh / TensorTopK.cu (TopKTypeConfig) for the radix keys.
//
// A radix sort orders unsigned integers by their bits, so each value is first
// mapped to an unsigned key with the same order: a float's bits order
// positive floats already, negative ones are flipped (all bits, so a larger
// magnitude becomes a smaller key) and positive ones get the sign bit set.
// NaN becomes the largest key, as on the CPU (last ascending, first
// descending), and -0.0 the key of 0.0, so they tie as the CPU's comparison
// says. Radix sorts are stable, so ties keep their position order and the
// result is the CPU's.
//
// PyTorch sorts small slices in shared memory (bitonic sort, up to 4096)
// and finds the top k with a radix select that never sorts the whole slice.
// Here both sort every slice with one cub call: simpler, and fast enough for
// sampling (one row of the vocabulary per token); a candidate to revisit
// for many short slices or a small k of a long one.

namespace raxis {
namespace {

constexpr int kThreads = 256;

template <typename scalar_t>
using radix_key_t = std::conditional_t<sizeof(scalar_t) == 8, uint64_t, uint32_t>;

template <typename scalar_t>
__device__ radix_key_t<scalar_t> to_radix_key(scalar_t v) {
    using key_t = radix_key_t<scalar_t>;
    if constexpr (std::is_floating_point_v<scalar_t>) {
        constexpr key_t sign = key_t{1} << (sizeof(key_t) * 8 - 1);
        if (v != v) {
            return ~key_t{0};
        }
        if (v == 0) {
            v = 0;
        }
        key_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        return bits ^ ((bits & sign) ? ~key_t{0} : sign);
    } else if constexpr (std::is_signed_v<scalar_t>) {
        constexpr key_t sign = key_t{1} << (sizeof(scalar_t) * 8 - 1);
        return static_cast<key_t>(v) ^ sign;
    } else {
        return static_cast<key_t>(v);
    }
}

// Keys and positions of every element of [segments, size], and the segment
// offsets (segment s is [offsets[s], offsets[s + 1]))
template <typename scalar_t>
__global__ void make_keys_kernel(
    radix_key_t<scalar_t>* keys, int64_t* positions, int32_t* offsets, const scalar_t* self,
    int32_t n, int32_t size, int32_t segments
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        keys[i] = to_radix_key(self[i]);
        positions[i] = i % size;
    }
    if (i <= segments) {
        offsets[i] = i * size;
    }
}

// The first k sorted entries of every segment: values read back from self
template <typename scalar_t>
__global__ void gather_sorted_kernel(
    scalar_t* values, int64_t* indices, const scalar_t* self, const int64_t* sorted_positions,
    int32_t n, int32_t size, int32_t k
) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int32_t segment = i / k;
    const int64_t position = sorted_positions[static_cast<int64_t>(segment) * size + i % k];
    values[i] = self[static_cast<int64_t>(segment) * size + position];
    indices[i] = position;
}

unsigned blocks_for(int64_t n) {
    return static_cast<unsigned>((n + kThreads - 1) / kThreads);
}

// values / indices get the first k entries of every slice along dim, sorted
void sort_slices(const Tensor& values, const Tensor& indices, const Tensor& self_, int64_t dim, int64_t k,
                 bool descending) {
    DeviceGuard guard(self_.device());
    cudaStream_t stream = cuda::current_stream();
    // Slices along the last dim, contiguous: what the segmented sort takes
    const Tensor self = self_.transpose(dim, -1).contiguous();
    const int64_t n = self.numel();
    const int64_t size = self.shape()[self.ndim() - 1];
    const int64_t segments = n / size;
    RAXIS_CHECK(n <= std::numeric_limits<int32_t>::max(),
        "sort / topk: tensors over 2^31 elements are not supported on CUDA yet");

    DimVector out_shape(self.shape().begin(), self.shape().end());
    out_shape.back() = k;
    Tensor sorted_values = empty(out_shape, self.options());
    Tensor sorted_indices = empty(out_shape, self.options().dtype(dtypes::Int64));
    const TensorOptions bytes = self.options().dtype(dtypes::Uint8);

    self.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        using key_t = radix_key_t<scalar_t>;
        const auto nbytes = [](int64_t count, size_t elem) { return static_cast<int64_t>(count * elem); };
        Tensor keys = empty({nbytes(n, sizeof(key_t))}, bytes);
        Tensor keys_out = empty({nbytes(n, sizeof(key_t))}, bytes);
        Tensor positions = empty({n}, self.options().dtype(dtypes::Int64));
        Tensor positions_out = empty({n}, self.options().dtype(dtypes::Int64));
        Tensor offsets = empty({segments + 1}, self.options().dtype(dtypes::Int32));
        auto* keys_ptr = reinterpret_cast<key_t*>(keys.mutable_data_ptr<uint8_t>());
        auto* keys_out_ptr = reinterpret_cast<key_t*>(keys_out.mutable_data_ptr<uint8_t>());
        int32_t* offsets_ptr = offsets.mutable_data_ptr<int32_t>();

        make_keys_kernel<scalar_t><<<blocks_for(std::max(n, segments + 1)), kThreads, 0, stream>>>(
            keys_ptr, positions.mutable_data_ptr<int64_t>(), offsets_ptr, self.data_ptr<scalar_t>(),
            static_cast<int32_t>(n), static_cast<int32_t>(size), static_cast<int32_t>(segments));
        RAXIS_CUDA_KERNEL_LAUNCH_CHECK();

        // Only the bits a key uses: 8 passes fewer for uint8 and bool
        const int end_bit = static_cast<int>(sizeof(scalar_t) * 8);
        auto run = [&](void* temp, size_t& temp_bytes) {
            if (descending) {
                return cub::DeviceSegmentedRadixSort::SortPairsDescending(
                    temp, temp_bytes, keys_ptr, keys_out_ptr, positions.data_ptr<int64_t>(),
                    positions_out.mutable_data_ptr<int64_t>(), static_cast<int>(n), static_cast<int>(segments),
                    offsets_ptr, offsets_ptr + 1, 0, end_bit, stream);
            }
            return cub::DeviceSegmentedRadixSort::SortPairs(
                temp, temp_bytes, keys_ptr, keys_out_ptr, positions.data_ptr<int64_t>(),
                positions_out.mutable_data_ptr<int64_t>(), static_cast<int>(n), static_cast<int>(segments),
                offsets_ptr, offsets_ptr + 1, 0, end_bit, stream);
        };
        // cub's two calls: the first only asks for the temporary storage
        size_t temp_bytes = 0;
        RAXIS_CUDA_CHECK(run(nullptr, temp_bytes));
        Tensor temp = empty({static_cast<int64_t>(temp_bytes)}, bytes);
        RAXIS_CUDA_CHECK(run(temp.mutable_data_ptr<uint8_t>(), temp_bytes));

        const int64_t n_out = segments * k;
        gather_sorted_kernel<scalar_t><<<blocks_for(n_out), kThreads, 0, stream>>>(
            sorted_values.mutable_data_ptr<scalar_t>(), sorted_indices.mutable_data_ptr<int64_t>(),
            self.data_ptr<scalar_t>(), positions_out.data_ptr<int64_t>(),
            static_cast<int32_t>(n_out), static_cast<int32_t>(size), static_cast<int32_t>(k));
        RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
    });
    values.transpose(dim, -1).copy_(sorted_values);
    indices.transpose(dim, -1).copy_(sorted_indices);
}

void sort_kernel_cuda(const Tensor& values, const Tensor& indices, const Tensor& self, int64_t dim, bool descending) {
    sort_slices(values, indices, self, dim, self.shape()[dim], descending);
}

// Always sorted: the sort gives that for free
void topk_kernel_cuda(const Tensor& values, const Tensor& indices, const Tensor& self,
                      int64_t k, int64_t dim, bool largest, bool /*sorted*/) {
    sort_slices(values, indices, self, dim, k, /*descending=*/largest);
}

} // anonymous namespace

REGISTER_DISPATCH(sort_stub, &sort_kernel_cuda);
REGISTER_DISPATCH(topk_stub, &topk_kernel_cuda);

} // namespace raxis
