#include <algorithm>
#include <cstdint>
#include <type_traits>

#include <cub/device/device_scan.cuh>
#include <thrust/iterator/reverse_iterator.h>

#include <raxis/core/device_guard.h>
#include <raxis/cuda/cuda_exception.h>
#include <raxis/cuda/cuda_functions.h>
#include <raxis/ops/reduce_ops.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/tensor.h>


// cumsum / cumprod on the GPU.
// From: aten/src/ATen/native/cuda/ScanUtils.cuh (scan_dim,
// tensor_kernel_scan_innermost_dim, tensor_kernel_scan_outer_dim) and
// CumsumKernel.cu, CumprodKernel.cu
//
// Three cases, as PyTorch:
// - A 1-D scan (one row): cub's DeviceScan, the decoupled look-back scan
//   that runs the whole row in one pass over memory. Its tiles combine
//   their prefixes in an order that depends on timing, so float results
//   can change in the last bits between runs (PyTorch's too).
// - Along the last dim: rows of a block, each scanned by a row of threads
//   in chunks of 2 * threads_x elements through shared memory (Sklansky's
//   scan: log2 steps, in step m every element in the upper half of a group
//   of 2^(m+1) adds the last element of the lower half), carrying the
//   chunk's total into the next chunk.
// - Along an outer dim: one thread per column walks it serially; the
//   threads of a warp read neighbouring columns, so the loads coalesce.
//
// reverse (reversed_cumsum, for the backward of cumsum) scans from the end
// of each row: positions along dim are read and written mirrored.

namespace raxis {
namespace {

template <typename T>
struct SumOp {
    __device__ T operator()(T a, T b) const { return a + b; }
};

template <typename T>
struct ProdOp {
    __device__ T operator()(T a, T b) const { return a * b; }
};

template <>
struct SumOp<bool> {
    __device__ bool operator()(bool a, bool b) const { return a || b; }
};

template <>
struct ProdOp<bool> {
    __device__ bool operator()(bool a, bool b) const { return a && b; }
};

template <typename scalar_t, typename op_t>
__global__ void scan_outer_dim_kernel(scalar_t* out, const scalar_t* in, uint32_t num_orows, uint32_t num_irows,
                                      uint32_t row_size, bool reverse, scalar_t init, op_t op) {
    for (uint32_t orow = blockIdx.x; orow < num_orows; orow += gridDim.x) {
        for (uint32_t irow = blockIdx.y * blockDim.x + threadIdx.x; irow < num_irows;
             irow += gridDim.y * blockDim.x) {
            const int64_t base = static_cast<int64_t>(orow) * row_size * num_irows + irow;
            scalar_t acc = init;
            for (uint32_t col = 0; col < row_size; col++) {
                const int64_t offset = base + static_cast<int64_t>(reverse ? row_size - 1 - col : col) * num_irows;
                acc = op(acc, in[offset]);
                out[offset] = acc;
            }
        }
    }
}

// blockDim = (2^log_threads_x, rows per block); shared memory holds
// 2 * blockDim.x values per row
template <typename scalar_t, typename op_t>
__global__ void scan_innermost_dim_kernel(scalar_t* out, const scalar_t* in, uint32_t num_rows, uint32_t row_size,
                                          int log_threads_x, bool reverse, scalar_t init, op_t op) {
    alignas(sizeof(double)) extern __shared__ char shared_bytes[];
    const uint32_t threads_x = 1u << log_threads_x;
    scalar_t* buf = reinterpret_cast<scalar_t*>(shared_bytes) + 2 * threads_x * threadIdx.y;
    const auto at = [&](uint32_t col) { return reverse ? row_size - 1 - col : col; };

    for (uint32_t block_row = blockIdx.x * blockDim.y; block_row < num_rows; block_row += blockDim.y * gridDim.x) {
        const uint32_t row = block_row + threadIdx.y;
        const bool row_exists = row < num_rows;
        const scalar_t* row_in = in + static_cast<int64_t>(row) * row_size;
        scalar_t* row_out = out + static_cast<int64_t>(row) * row_size;
        scalar_t chunk_total = init;

        for (uint32_t chunk = 0; chunk < row_size; chunk += 2 * threads_x) {
            const uint32_t col1 = chunk + threadIdx.x;
            const uint32_t col2 = chunk + threads_x + threadIdx.x;
            if (row_exists) {
                buf[threadIdx.x] = col1 < row_size ? row_in[at(col1)] : init;
                buf[threads_x + threadIdx.x] = col2 < row_size ? row_in[at(col2)] : init;
                // The total of the previous chunks enters through the first value
                if (threadIdx.x == 0) {
                    buf[0] = op(buf[0], chunk_total);
                }
            }
            __syncthreads();
            for (int m = 0; m <= log_threads_x; m++) {
                if (row_exists) {
                    const uint32_t s = 1u << m;
                    const uint32_t a = ((threadIdx.x >> m) << (m + 1)) | s;   // start of the upper half
                    buf[a + threadIdx.x % s] = op(buf[a + threadIdx.x % s], buf[a - 1]);
                }
                __syncthreads();
            }
            if (row_exists) {
                if (col1 < row_size) row_out[at(col1)] = buf[threadIdx.x];
                if (col2 < row_size) row_out[at(col2)] = buf[threads_x + threadIdx.x];
            }
            chunk_total = buf[2 * threads_x - 1];
            __syncthreads();
        }
    }
}

// Threads along a row vs across rows: about the ratio of row_size to
// num_rows, 512 threads in all, between 16 and 512 along the row.
// From: ScanUtils.cuh (get_log_num_threads_x_inner_scan)
int log_threads_x_inner_scan(int64_t num_rows, int64_t row_size) {
    int log_x = 0;
    int log_y = 0;
    while ((int64_t{1} << log_x) < row_size) log_x++;
    while ((int64_t{1} << log_y) < num_rows) log_y++;
    return std::clamp((9 + log_x - log_y) / 2, 4, 9);
}

template <typename scalar_t, typename op_t>
void scan_dim(const Tensor& result, const Tensor& self_, int64_t dim, bool reverse, scalar_t init, op_t op) {
    DeviceGuard guard(self_.device());
    const Tensor self = self_.contiguous();
    const cudaStream_t stream = cuda::current_stream();
    const scalar_t* in = self.data_ptr<scalar_t>();
    scalar_t* out = result.mutable_data_ptr<scalar_t>();
    const int64_t numel = self.numel();
    const int64_t row_size = self.shape()[dim];
    RAXIS_CHECK(numel <= UINT32_MAX, "cumsum / cumprod: tensors over 2^32 elements are not supported on CUDA yet");

    if (numel == row_size) {
        const auto n = static_cast<int>(numel);
        size_t temp_bytes = 0;
        Tensor temp;
        auto run = [&](void* storage) {
            if (reverse) {
                RAXIS_CUDA_CHECK(cub::DeviceScan::InclusiveScan(storage, temp_bytes,
                    thrust::make_reverse_iterator(in + n), thrust::make_reverse_iterator(out + n), op, n, stream));
            } else {
                RAXIS_CUDA_CHECK(cub::DeviceScan::InclusiveScan(storage, temp_bytes, in, out, op, n, stream));
            }
        };
        run(nullptr);
        temp = empty({static_cast<int64_t>(temp_bytes)}, self.options().dtype(dtypes::Uint8));
        run(temp.mutable_data_ptr());
        return;
    }
    const cudaDeviceProp& props = cuda::current_device_properties();
    if (dim == self.ndim() - 1) {
        const int64_t num_rows = numel / row_size;
        const int log_x = log_threads_x_inner_scan(num_rows, row_size);
        const dim3 threads(1u << log_x, 512u >> log_x);
        const dim3 grid(static_cast<unsigned>(std::min<int64_t>(props.maxGridSize[0], (num_rows + threads.y - 1) / threads.y)));
        scan_innermost_dim_kernel<<<grid, threads, 512 * 2 * sizeof(scalar_t), stream>>>(
            out, in, static_cast<uint32_t>(num_rows), static_cast<uint32_t>(row_size), log_x, reverse, init, op);
    } else {
        int64_t num_orows = 1;
        int64_t num_irows = 1;
        for (int64_t d = 0; d < dim; d++) num_orows *= self.shape()[d];
        for (int64_t d = dim + 1; d < self.ndim(); d++) num_irows *= self.shape()[d];
        const dim3 threads(static_cast<unsigned>(std::min<int64_t>(512, num_irows)));
        const dim3 grid(static_cast<unsigned>(std::min<int64_t>(props.maxGridSize[1], num_orows)),
                        static_cast<unsigned>(std::min<int64_t>(props.maxGridSize[1], (num_irows + threads.x - 1) / threads.x)));
        scan_outer_dim_kernel<<<grid, threads, 0, stream>>>(
            out, in, static_cast<uint32_t>(num_orows), static_cast<uint32_t>(num_irows),
            static_cast<uint32_t>(row_size), reverse, init, op);
    }
    RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
}

void cumsum_kernel_cuda(const Tensor& result, const Tensor& self, int64_t dim, bool reverse) {
    result.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        scan_dim<scalar_t>(result, self, dim, reverse, scalar_t(0), SumOp<scalar_t>{});
    });
}

void cumprod_kernel_cuda(const Tensor& result, const Tensor& self, int64_t dim, bool reverse) {
    result.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        scan_dim<scalar_t>(result, self, dim, reverse, scalar_t(1), ProdOp<scalar_t>{});
    });
}

} // anonymous namespace

REGISTER_DISPATCH(cumsum_stub, &cumsum_kernel_cuda);
REGISTER_DISPATCH(cumprod_stub, &cumprod_kernel_cuda);

} // namespace raxis
