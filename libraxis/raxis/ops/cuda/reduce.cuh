#pragma once

#include <algorithm>
#include <array>
#include <cstdint>

#include <cuda_runtime.h>

#include <raxis/core/device_guard.h>
#include <raxis/cuda/cuda_exception.h>
#include <raxis/cuda/cuda_functions.h>
#include <raxis/ops/cuda/offset_calculator.cuh>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/tensor_iterator.h>


// Reductions on the GPU: sum, mean, max, argmax, ... over the reduction
// TensorIterator built by make_reduction, the same one the CPU kernels get.
// From: aten/src/ATen/native/cuda/Reduce.cuh (ReduceConfig, setReduceConfig,
// ReduceOp, gpu_reduce_kernel), simplified: one input and one output, no
// vectorized loads, 32-bit indexing only.
//
// After reorder_dimensions the reduced dims (stride 0 in the output) come
// first, so the iterator is a (inputs_per_output x num_outputs) problem.
// The work is spread over up to three levels, each one ending in a tree of
// combines:
//
//   1. Threads: each thread folds a strided run of inputs into a register,
//      with 4 independent accumulators so consecutive loads do not wait on
//      each other (instruction-level parallelism).
//   2. The block: threads that share an output combine their values, with
//      warp shuffles (registers to registers, no memory) along x and through
//      shared memory along y.
//   3. The grid: when there are few outputs and long reductions, several
//      blocks (gridDim.y) split each output's inputs. Each block writes its
//      partial value to a global staging buffer and bumps a counter; the
//      last block to finish (it sees the counter reach gridDim.y - 1) reads
//      all the partials and writes the result, in the same launch.
//
// The block shape follows the memory layout. When the reduced dim is the
// contiguous one (sum over the last dim), the threads of a warp read
// consecutive inputs of one output and combine them (block_x_reduce). When
// it is not (sum over dim 0 of a row-major matrix), consecutive threads take
// consecutive outputs, which are the neighbouring addresses, and each thread
// walks its column: the loads stay coalesced either way.
//
// An Ops type provides reduce(acc, value, index), combine(acc, acc),
// project(acc) -> output and warp_shfl_down(acc, offset), as PyTorch's.

namespace raxis::cuda {

constexpr int kReduceMaxThreads = 512;
constexpr int kWarpSize = 32;
// Values each thread folds before a reduction is split across warps or blocks
constexpr int kMinValuesPerThread = 16;
constexpr int kMaxValuesPerThread = 256;

inline int64_t div_up(int64_t a, int64_t b) {
    return (a + b - 1) / b;
}

inline int last_pow2(int64_t n) {
    int p = 1;
    while (static_cast<int64_t>(p) * 2 <= n) {
        p *= 2;
    }
    return p;
}

/// Elements of the reduction's output: the product of the kept dims, which
/// reorder_dimensions placed after the reduced ones
inline int64_t num_output_elements(const TensorIterator& iter) {
    int64_t num_outputs = 1;
    for (int d = iter.num_reduce_dims(); d < iter.ndim(); d++) {
        num_outputs *= iter.shape()[d];
    }
    return num_outputs;
}

// From: aten/src/ATen/native/cuda/Reduce.cuh (ReduceConfig)
// Where each thread starts and how far it steps, as multipliers of its
// (threadIdx.x, threadIdx.y, blockIdx.y) for the inputs and (threadIdx.x,
// threadIdx.y) for the outputs: a non-zero input multiplier on an axis means
// the threads along it share outputs, so they must be combined afterwards.
struct ReduceConfig {
    static constexpr int BLOCK_X = 0;
    static constexpr int BLOCK_Y = 1;
    static constexpr int CTA = 2;

    ReduceConfig(int element_size_bytes, int num_outputs, int num_inputs)
        : element_size_bytes(element_size_bytes), num_inputs(num_inputs), num_outputs(num_outputs) {}

    int element_size_bytes;
    int num_inputs;
    int num_outputs;
    int step_input = 1;
    int step_output = 1;
    int ctas_per_output = 1;
    int input_mult[3] = {0, 0, 0};
    int output_mult[2] = {0, 0};
    int block_width = 1;
    int block_height = 1;
    int num_threads = 1;

    // A warp along the dim that maps to consecutive memory (dim0), the rest
    // of the block along the other one
    void set_block_dimension(int64_t dim0, int64_t dim1) {
        const int dim0_pow2 = dim0 < kReduceMaxThreads ? last_pow2(dim0) : kReduceMaxThreads;
        const int dim1_pow2 = dim1 < kReduceMaxThreads ? last_pow2(dim1) : kReduceMaxThreads;
        block_width = std::min(dim0_pow2, kWarpSize);
        block_height = std::min(dim1_pow2, kReduceMaxThreads / block_width);
        block_width = std::min(dim0_pow2, kReduceMaxThreads / block_height);
        num_threads = block_width * block_height;
    }

    int split_input(int parallelism) {
        const int step = step_input;
        step_input *= parallelism;
        return step;
    }

    int split_output(int parallelism) {
        const int step = step_output;
        step_output *= parallelism;
        return step;
    }

    dim3 block() const { return dim3(block_width, block_height); }
    dim3 grid() const { return dim3(static_cast<unsigned>(div_up(num_outputs, step_output)), ctas_per_output); }

    __host__ __device__ bool should_block_x_reduce() const { return input_mult[BLOCK_X] != 0; }
    __host__ __device__ bool should_block_y_reduce() const { return input_mult[BLOCK_Y] != 0; }
    __host__ __device__ bool should_global_reduce() const { return input_mult[CTA] != 0; }

    // One thread per output writes it: the one at x = 0 / y = 0 of the axes
    // that were combined
    __device__ bool should_store(int output_idx) const {
        return output_idx < num_outputs
            && (!should_block_x_reduce() || threadIdx.x == 0)
            && (!should_block_y_reduce() || threadIdx.y == 0);
    }

    __device__ int input_idx() const {
        return threadIdx.x * input_mult[BLOCK_X] + threadIdx.y * input_mult[BLOCK_Y]
            + blockIdx.y * input_mult[CTA];
    }

    __device__ int output_idx() const {
        return threadIdx.x * output_mult[BLOCK_X] + threadIdx.y * output_mult[BLOCK_Y]
            + blockIdx.x * step_output;
    }

    __device__ int shared_memory_offset(int offset) const {
        return threadIdx.x + (threadIdx.y + offset) * blockDim.x;
    }

    // Slot of block (blockIdx.x, cta2) in the staging buffer; without a
    // block_x_reduce every thread along x has its own output, so its own slot
    __device__ int staging_memory_offset(int cta2) const {
        int offset = cta2 + blockIdx.x * gridDim.y;
        if (!should_block_x_reduce()) {
            offset = threadIdx.x + offset * blockDim.x;
        }
        return offset;
    }

    int shared_memory_size() const {
        if (!should_block_y_reduce() && (!should_block_x_reduce() || block_width <= kWarpSize)) {
            return 0;
        }
        return element_size_bytes * num_threads;
    }

    int64_t global_memory_size() const {
        if (!should_global_reduce()) {
            return 0;
        }
        int64_t size = static_cast<int64_t>(element_size_bytes) * num_outputs * ctas_per_output;
        if (!should_block_x_reduce()) {
            size *= block_width;
        }
        return size;
    }

    int semaphore_size() const {
        return should_global_reduce() ? static_cast<int>(sizeof(int)) * static_cast<int>(grid().x) : 0;
    }

    int values_per_thread() const { return static_cast<int>(div_up(num_inputs, step_input)); }
};

template <typename scalar_t, typename ops_t, typename arg_t, typename out_t>
struct ReduceOp {
    // Independent accumulators per thread
    static constexpr int vt0 = 4;

    ops_t ops;
    arg_t ident;
    ReduceConfig config;
    OffsetCalculator<1> input_calc;   // over the reduced dims, input only
    OffsetCalculator<2> output_calc;  // over the kept dims: output, input
    const char* src;
    char* dst;
    void* cta_buf;
    int* semaphores;

    __device__ void run() const {
        extern __shared__ char shared_memory[];
        const int output_idx = config.output_idx();
        const int input_idx = config.input_idx();

        arg_t value = ident;
        if (output_idx < config.num_outputs && input_idx < config.num_inputs) {
            const char* input_slice = src + output_calc.get(output_idx)[1];
            value = thread_reduce(input_slice);
        }
        if (config.should_block_x_reduce()) {
            value = block_x_reduce(value, shared_memory);
        }
        if (config.should_block_y_reduce()) {
            value = block_y_reduce(value, shared_memory);
        }
        if (config.should_global_reduce()) {
            global_reduce(value, shared_memory);
        } else if (config.should_store(output_idx)) {
            store(output_idx, value);
        }
    }

    __device__ void store(int output_idx, arg_t value) const {
        *reinterpret_cast<out_t*>(dst + output_calc.get(output_idx)[0]) = ops.project(value);
    }

    // From: ReduceOp::thread_reduce_impl. vt0 loads in flight, each into
    // its own accumulator, then the accumulators are combined
    __device__ arg_t thread_reduce(const char* data) const {
        int idx = config.input_idx();
        const int end = config.num_inputs;
        const int stride = config.step_input;

        arg_t value_list[vt0];
        #pragma unroll
        for (int i = 0; i < vt0; i++) {
            value_list[i] = ident;
        }
        scalar_t values[vt0];
        while (idx + (vt0 - 1) * stride < end) {
            #pragma unroll
            for (int i = 0; i < vt0; i++) {
                values[i] = load(data, idx + i * stride);
            }
            #pragma unroll
            for (int i = 0; i < vt0; i++) {
                value_list[i] = ops.reduce(value_list[i], values[i], idx + i * stride);
            }
            idx += stride * vt0;
        }
        #pragma unroll
        for (int i = 0; i < vt0; i++) {
            if (idx >= end) {
                break;
            }
            value_list[i] = ops.reduce(value_list[i], load(data, idx), idx);
            idx += stride;
        }
        #pragma unroll
        for (int i = 1; i < vt0; i++) {
            value_list[0] = ops.combine(value_list[0], value_list[i]);
        }
        return value_list[0];
    }

    __device__ scalar_t load(const char* data, int idx) const {
        return *reinterpret_cast<const scalar_t*>(data + input_calc.get(idx)[0]);
    }

    // From: ReduceOp::block_x_reduce. Above a warp, halve through shared
    // memory down to 32 values; then shuffle within the warp. A shuffle
    // reads another lane's register directly, so the last five steps need
    // no shared memory and no __syncthreads
    __device__ arg_t block_x_reduce(arg_t value, char* shared_memory) const {
        int dim_x = blockDim.x;
        auto* shared = reinterpret_cast<arg_t*>(shared_memory);
        if (dim_x > kWarpSize) {
            const int address_base = threadIdx.x + threadIdx.y * blockDim.x;
            shared[address_base] = value;
            for (int offset = dim_x / 2; offset >= kWarpSize; offset >>= 1) {
                __syncthreads();
                if (threadIdx.x < offset && threadIdx.x + offset < blockDim.x) {
                    value = ops.combine(value, shared[address_base + offset]);
                    shared[address_base] = value;
                }
            }
            dim_x = kWarpSize;
        }
        __syncthreads();
        // Halving offsets: lane i adds lane i + 16, then i + 8, ... (the
        // order PyTorch uses on CUDA, for the same rounding)
        for (int offset = dim_x >> 1; offset > 0; offset >>= 1) {
            value = ops.combine(value, ops.warp_shfl_down(value, offset));
        }
        return value;
    }

    // From: ReduceOp::block_y_reduce. A tree over the rows of the block
    __device__ arg_t block_y_reduce(arg_t value, char* shared_memory) const {
        auto* shared = reinterpret_cast<arg_t*>(shared_memory);
        shared[config.shared_memory_offset(0)] = value;
        for (int offset = blockDim.y / 2; offset > 0; offset >>= 1) {
            __syncthreads();
            if (threadIdx.y < offset && threadIdx.y + offset < blockDim.y) {
                value = ops.combine(value, shared[config.shared_memory_offset(offset)]);
                shared[config.shared_memory_offset(0)] = value;
            }
        }
        return value;
    }

    // The counter of output column blockIdx.x: true in the block that
    // finishes last, which then sees every other block's partial
    __device__ bool mark_block_finished() const {
        __shared__ bool is_last_block_done_shared;
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            const int prev_blocks_finished = atomicAdd(&semaphores[blockIdx.x], 1);
            is_last_block_done_shared = (prev_blocks_finished == static_cast<int>(gridDim.y) - 1);
        }
        __syncthreads();
        return is_last_block_done_shared;
    }

    // From: ReduceOp::global_reduce
    __device__ void global_reduce(arg_t value, char* shared_memory) const {
        auto* reduce_buffer = static_cast<arg_t*>(cta_buf);
        const int output_idx = config.output_idx();
        const bool should_store = config.should_store(output_idx);
        if (should_store) {
            reduce_buffer[config.staging_memory_offset(blockIdx.y)] = value;
        }
        // The partials must be visible to the last block before it counts
        // this one as finished
        __threadfence();
        __syncthreads();
        if (!mark_block_finished()) {
            return;
        }
        __threadfence();
        value = ident;
        if (config.should_block_x_reduce()) {
            for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < config.ctas_per_output;
                 i += blockDim.x * blockDim.y) {
                value = ops.combine(value, reduce_buffer[config.staging_memory_offset(i)]);
            }
        } else {
            for (int i = threadIdx.y; i < config.ctas_per_output; i += blockDim.y) {
                value = ops.combine(value, reduce_buffer[config.staging_memory_offset(i)]);
            }
        }
        value = block_y_reduce(value, shared_memory);
        if (config.should_block_x_reduce()) {
            value = block_x_reduce(value, shared_memory);
        }
        if (should_store) {
            store(output_idx, value);
        }
    }
};

template <typename R>
__global__ void __launch_bounds__(kReduceMaxThreads, 4) reduce_kernel(R reduction) {
    reduction.run();
}

// From: aten/src/ATen/native/cuda/Reduce.cuh (setReduceConfig)
template <typename arg_t>
ReduceConfig set_reduce_config(const TensorIterator& iter, int64_t num_outputs, int64_t inputs_per_output) {
    ReduceConfig config(sizeof(arg_t), static_cast<int>(num_outputs), static_cast<int>(inputs_per_output));
    const int num_reduce_dims = iter.num_reduce_dims();
    const int input_index = 1;

    bool reduction_on_fastest_striding_dimension = true;
    int64_t dim0 = 1, dim1 = 1;
    if (iter.ndim() > 0) {
        reduction_on_fastest_striding_dimension = num_reduce_dims == iter.ndim()
            || iter.strides(input_index)[0] < iter.strides(input_index)[num_reduce_dims];
        if (reduction_on_fastest_striding_dimension) {
            dim0 = inputs_per_output;
            dim1 = num_outputs;
        } else {
            dim0 = num_outputs;
            dim1 = inputs_per_output;
        }
    }
    config.set_block_dimension(dim0, dim1);

    if (iter.ndim() == 0 || reduction_on_fastest_striding_dimension) {
        // The lanes of a warp share an output: combined with shuffles
        config.input_mult[ReduceConfig::BLOCK_X] = config.split_input(config.block_width);
    } else {
        config.output_mult[ReduceConfig::BLOCK_X] = config.split_output(config.block_width);
    }

    // Split each output's inputs across the warps of the block when every
    // thread still gets enough of them; otherwise one output per warp
    const int warp_split_threshold = std::min(config.block_height * kMinValuesPerThread, kMaxValuesPerThread);
    if (config.values_per_thread() >= warp_split_threshold) {
        config.input_mult[ReduceConfig::BLOCK_Y] = config.split_input(config.block_height);
    } else {
        config.output_mult[ReduceConfig::BLOCK_Y] = config.split_output(config.block_height);
    }

    // Split across blocks when threads still have long runs and there are
    // too few outputs to fill the GPU
    const cudaDeviceProp& props = current_device_properties();
    const int blocks_per_sm = props.maxThreadsPerMultiProcessor / config.num_threads;
    const int target_grid_size = props.multiProcessorCount * blocks_per_sm;
    const int grid = static_cast<int>(config.grid().x);
    if (config.input_mult[ReduceConfig::BLOCK_Y] != 0 && config.values_per_thread() >= kMaxValuesPerThread
            && grid <= target_grid_size) {
        const int ctas_per_output1 = static_cast<int>(div_up(target_grid_size, grid));
        const int ctas_per_output2 = static_cast<int>(div_up(config.values_per_thread(), kMinValuesPerThread));
        const int ctas_per_output3 = static_cast<int>(div_up(config.values_per_thread(), kMaxValuesPerThread));
        config.ctas_per_output = std::clamp(ctas_per_output1, ctas_per_output3, ctas_per_output2);
        if (config.ctas_per_output > 1) {
            config.input_mult[ReduceConfig::CTA] = config.split_input(config.ctas_per_output);
        }
    }
    return config;
}

/// Reduces input 1 of `iter` into output 0 with `ops`, starting every
/// accumulator at `ident`. scalar_t is the input type, out_t the output's.
/// From: aten/src/ATen/native/cuda/Reduce.cuh (gpu_reduce_kernel)
template <typename scalar_t, typename out_t, typename ops_t, typename arg_t>
void gpu_reduce_kernel(TensorIterator& iter, const ops_t& ops, arg_t ident) {
    RAXIS_INTERNAL_ASSERT(iter.numel() > 0 && iter.num_inputs() == 1 && iter.num_outputs() == 1);
    // PyTorch splits a larger reduction and accumulates the pieces
    RAXIS_CHECK_NOT_IMPLEMENTED(iter.can_use_32bit_indexing(),
        "CUDA reductions over tensors with offsets past 2^31 are not supported yet");
    DeviceGuard guard(iter.device(0));

    const int num_reduce_dims = iter.num_reduce_dims();
    const int ndim = iter.ndim();
    const IntArrayRef shape = iter.shape();
    const int64_t num_outputs = num_output_elements(iter);
    const int64_t inputs_per_output = iter.numel() / num_outputs;
    const ReduceConfig config = set_reduce_config<arg_t>(iter, num_outputs, inputs_per_output);

    // Reduced dims: input strides only; kept dims: output and input
    const int64_t* input_strides[] = {iter.strides(1).data()};
    const int64_t* output_strides[] = {iter.strides(0).data() + num_reduce_dims, iter.strides(1).data() + num_reduce_dims};
    OffsetCalculator<1> input_calc(num_reduce_dims, shape.data(), input_strides);
    OffsetCalculator<2> output_calc(ndim - num_reduce_dims, shape.data() + num_reduce_dims, output_strides);

    // Staging buffer and per-column counters of the cross-block reduction,
    // from the caching allocator; freed after the launch, which is fine on a
    // single stream (the next user of the memory runs after this kernel)
    Tensor buffer, semaphores;
    if (config.should_global_reduce()) {
        const TensorOptions options = TensorOptions(dtypes::Uint8).device(iter.device(0));
        buffer = empty({config.global_memory_size()}, options);
        semaphores = empty({config.semaphore_size()}, options);
        RAXIS_CUDA_CHECK(cudaMemsetAsync(semaphores.mutable_data_ptr(), 0, config.semaphore_size(), current_stream()));
    }

    ReduceOp<scalar_t, ops_t, arg_t, out_t> reduce{
        ops, ident, config, input_calc, output_calc,
        static_cast<const char*>(iter.data_ptr(1)), static_cast<char*>(iter.data_ptr(0)),
        buffer.defined() ? buffer.mutable_data_ptr() : nullptr,
        semaphores.defined() ? static_cast<int*>(semaphores.mutable_data_ptr()) : nullptr};
    reduce_kernel<<<config.grid(), config.block(), config.shared_memory_size(), current_stream()>>>(reduce);
    RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace raxis::cuda
