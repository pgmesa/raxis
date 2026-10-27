#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

#include <cuda_runtime.h>

#include <raxis/core/dtype.h>
#include <raxis/core/device_guard.h>
#include <raxis/cuda/cuda_exception.h>
#include <raxis/cuda/cuda_functions.h>
#include <raxis/ops/cuda/offset_calculator.cuh>
#include <raxis/tensor/tensor_iterator.h>
#include <raxis/util/function_traits.h>


// gpu_kernel: runs an elementwise functor over a TensorIterator on the GPU,
// the CUDA counterpart of cpu_kernel (ops/cpu/loops.h).
// From: aten/src/ATen/native/cuda/Loops.cuh, CUDALoops.cuh, MemoryAccess.cuh
//
// The iterator is the same one the CPU uses: broadcasting, type promotion,
// dims reordered and coalesced on the host. What changes is the loop. On the
// CPU a few threads each walk a long chunk; on the GPU every thread handles
// a handful of elements, so there are two kernels:
//
// - Contiguous operands of the functor's own types (the common case after
//   coalescing): every block takes a contiguous tile, and each thread moves
//   its elements with vector loads and stores (up to 16 bytes, float4 for
//   float), so a warp reads whole cache lines in few instructions.
// - Anything else (broadcast, permuted, mixed dtypes): each thread computes
//   the offsets of its element in every operand with OffsetCalculator, and
//   reads / writes with a cast when the dtype differs from the functor's.
//
// Kernels index with 32-bit integers (faster address math, IntDivider);
// iterators that need larger offsets are split first.

// A lambda that runs on the device but whose type the host can inspect
// (function_traits): needs nvcc's --extended-lambda
#define GPU_LAMBDA __host__ __device__

namespace raxis::cuda {

// 4 warps per block. From: aten/src/ATen/native/cuda/thread_constants.h
constexpr int kNumThreads = 128;

// ============================================================================
// Dynamic casting
// ============================================================================

// Operands whose dtype differs from the functor's types are read and written
// through a switch on the runtime dtype, instead of compiling the functor
// for every combination of input and output dtypes.
// From: c10/core/DynamicCast.h

template <typename dest_t>
__device__ inline dest_t fetch_and_cast(DtypeKind src, const char* ptr) {
    switch (src) {
        case DtypeKind::Bool:    return static_cast<dest_t>(*reinterpret_cast<const bool*>(ptr));
        case DtypeKind::Uint8:   return static_cast<dest_t>(*reinterpret_cast<const uint8_t*>(ptr));
        case DtypeKind::Int32:   return static_cast<dest_t>(*reinterpret_cast<const int32_t*>(ptr));
        case DtypeKind::Int64:   return static_cast<dest_t>(*reinterpret_cast<const int64_t*>(ptr));
        case DtypeKind::Float32: return static_cast<dest_t>(*reinterpret_cast<const float*>(ptr));
        case DtypeKind::Float64: return static_cast<dest_t>(*reinterpret_cast<const double*>(ptr));
        default:                 return dest_t{};
    }
}

template <typename src_t>
__device__ inline void cast_and_store(DtypeKind dest, char* ptr, src_t value) {
    switch (dest) {
        case DtypeKind::Bool:    *reinterpret_cast<bool*>(ptr) = static_cast<bool>(value); break;
        case DtypeKind::Uint8:   *reinterpret_cast<uint8_t*>(ptr) = static_cast<uint8_t>(value); break;
        case DtypeKind::Int32:   *reinterpret_cast<int32_t*>(ptr) = static_cast<int32_t>(value); break;
        case DtypeKind::Int64:   *reinterpret_cast<int64_t*>(ptr) = static_cast<int64_t>(value); break;
        case DtypeKind::Float32: *reinterpret_cast<float*>(ptr) = static_cast<float>(value); break;
        case DtypeKind::Float64: *reinterpret_cast<double*>(ptr) = static_cast<double>(value); break;
        default: break;
    }
}

/// Whether some operand's dtype differs from the functor's signature
template <typename func_t>
bool needs_dynamic_casting(const TensorIterator& iter) {
    using traits = function_traits<func_t>;
    using result_t = typename traits::result_type;
    if (iter.dtype(0) != Dtype::from_type<result_t>()) {
        return true;
    }
    return [&]<size_t... I>(std::index_sequence<I...>) {
        return ((iter.dtype(I + 1) != Dtype::from_type<typename traits::template arg<I>::type>()) || ...);
    }(std::make_index_sequence<traits::arity>{});
}

// ============================================================================
// Vectorized kernel: contiguous operands, no casting
// ============================================================================

template <typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t) * vec_size) aligned_vector {
    scalar_t val[vec_size];
};

/// The widest vector (in elements, up to 4) the address allows
/// From: MemoryAccess.cuh (can_vectorize_up_to)
template <typename scalar_t>
int max_vec_size_for(const char* pointer) {
    auto address = reinterpret_cast<uintptr_t>(pointer);
    if (address % alignof(aligned_vector<scalar_t, 4>) == 0) {
        return 4;
    } else if (address % alignof(aligned_vector<scalar_t, 2>) == 0) {
        return 2;
    }
    return 1;
}

template <typename func_t, typename array_t>
int can_vectorize_up_to(const array_t& data) {
    using traits = function_traits<func_t>;
    int result = max_vec_size_for<typename traits::result_type>(data[0]);
    [&]<size_t... I>(std::index_sequence<I...>) {
        ((result = std::min(result, max_vec_size_for<typename traits::template arg<I>::type>(data[I + 1]))), ...);
    }(std::make_index_sequence<traits::arity>{});
    return result;
}

/// Elements per thread: more for 1-byte operands, so every thread still
/// moves enough bytes to hide the memory latency
/// From: CUDALoops.cuh (elems_per_thread)
template <typename func_t>
constexpr int elems_per_thread() {
    using traits = function_traits<func_t>;
    constexpr size_t io_size = sizeof(typename traits::result_type) +
        []<size_t... I>(std::index_sequence<I...>) {
            return (size_t{0} + ... + sizeof(typename traits::template arg<I>::type));
        }(std::make_index_sequence<traits::arity>{});
    return io_size == 1 ? 16 : 8;
}

/// Loads input I of this thread's elements: element j of vector i goes to
/// args[vec_size * i + j]
template <int I, int vec_size, int kLoops, typename args_t>
__device__ inline void load_vectorized(args_t* args, const char* data, int block_start) {
    using arg_t = std::tuple_element_t<I, args_t>;
    using vec_t = aligned_vector<arg_t, vec_size>;
    const vec_t* from = reinterpret_cast<const vec_t*>(reinterpret_cast<const arg_t*>(data) + block_start);
    #pragma unroll
    for (int i = 0; i < kLoops; i++) {
        vec_t v = from[threadIdx.x + i * kNumThreads];
        #pragma unroll
        for (int j = 0; j < vec_size; j++) {
            std::get<I>(args[vec_size * i + j]) = v.val[j];
        }
    }
}

// Block b handles elements [b * kBlockWork, (b + 1) * kBlockWork). Full
// blocks read and write with vectors: thread t takes vectors t, t + 128, ...
// so consecutive threads touch consecutive addresses (coalesced). The last,
// partial block falls back to one element at a time with a bounds check.
template <int vec_size, typename func_t, typename array_t>
__global__ void __launch_bounds__(kNumThreads)
vectorized_elementwise_kernel(int N, func_t f, array_t data) {
    using traits = function_traits<func_t>;
    using result_t = typename traits::result_type;
    using args_t = typename traits::ArgsTuple;
    constexpr int kArity = static_cast<int>(traits::arity);
    constexpr int kThreadWork = elems_per_thread<func_t>();
    constexpr int kBlockWork = kThreadWork * kNumThreads;
    constexpr int kLoops = kThreadWork / vec_size;
    static_assert(kThreadWork % vec_size == 0);

    const int block_start = kBlockWork * blockIdx.x;
    const int remaining = N - block_start;
    args_t args[kThreadWork];
    result_t results[kThreadWork];

    if (remaining >= kBlockWork) {
        [&]<int... I>(std::integer_sequence<int, I...>) {
            (load_vectorized<I, vec_size, kLoops>(args, data[I + 1], block_start), ...);
        }(std::make_integer_sequence<int, kArity>{});

        #pragma unroll
        for (int k = 0; k < kThreadWork; k++) {
            results[k] = std::apply(f, args[k]);
        }

        using out_vec_t = aligned_vector<result_t, vec_size>;
        out_vec_t* to = reinterpret_cast<out_vec_t*>(reinterpret_cast<result_t*>(data[0]) + block_start);
        #pragma unroll
        for (int i = 0; i < kLoops; i++) {
            out_vec_t v;
            #pragma unroll
            for (int j = 0; j < vec_size; j++) {
                v.val[j] = results[vec_size * i + j];
            }
            to[threadIdx.x + i * kNumThreads] = v;
        }
    } else {
        // Tail: element k of the thread is block_start + threadIdx.x + k * 128
        #pragma unroll
        for (int k = 0; k < kThreadWork; k++) {
            int idx = threadIdx.x + k * kNumThreads;
            if (idx < remaining) {
                [&]<int... I>(std::integer_sequence<int, I...>) {
                    ((std::get<I>(args[k]) =
                        reinterpret_cast<const std::tuple_element_t<I, args_t>*>(data[I + 1])[block_start + idx]), ...);
                }(std::make_integer_sequence<int, kArity>{});
                results[k] = std::apply(f, args[k]);
                reinterpret_cast<result_t*>(data[0])[block_start + idx] = results[k];
            }
        }
    }
}

template <typename func_t, typename array_t>
void launch_vectorized_kernel(int64_t N, const func_t& f, const array_t& data) {
    RAXIS_INTERNAL_ASSERT(N > 0 && N <= std::numeric_limits<int32_t>::max());
    using result_t = typename function_traits<func_t>::result_type;
    constexpr int kBlockWork = elems_per_thread<func_t>() * kNumThreads;
    const int64_t grid = (N + kBlockWork - 1) / kBlockWork;
    // At most 16 bytes per vector (float4, double2), and what the alignment of
    // every pointer allows. PyTorch also goes to 32 bytes on sm_90 / sm_100
    int vec_size = std::min<int>(16 / static_cast<int>(sizeof(result_t)), can_vectorize_up_to<func_t>(data));
    cudaStream_t stream = current_stream();
    switch (vec_size) {
        case 4:
            vectorized_elementwise_kernel<4><<<grid, kNumThreads, 0, stream>>>(static_cast<int>(N), f, data);
            break;
        case 2:
            vectorized_elementwise_kernel<2><<<grid, kNumThreads, 0, stream>>>(static_cast<int>(N), f, data);
            break;
        default:
            vectorized_elementwise_kernel<1><<<grid, kNumThreads, 0, stream>>>(static_cast<int>(N), f, data);
            break;
    }
    RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
}

// ============================================================================
// Generic kernel: any strides, dynamic casting
// ============================================================================

// Thread t of block b handles elements b * nt * vt + t + i * nt, i < vt:
// strided by the block size so a warp still touches neighbouring elements
// From: CUDALoops.cuh (elementwise_kernel, launch_legacy_kernel)
template <int nt, int vt, typename func_t>
__global__ void __launch_bounds__(nt, 4)
elementwise_kernel(int N, func_t f) {
    int idx = nt * vt * blockIdx.x + threadIdx.x;
    #pragma unroll
    for (int i = 0; i < vt; i++) {
        if (idx < N) {
            f(idx);
            idx += nt;
        }
    }
}

template <int nt, int vt, typename func_t>
void launch_legacy_kernel(int64_t N, const func_t& f) {
    RAXIS_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
    if (N == 0) {
        return;
    }
    const int64_t grid = (N + nt * vt - 1) / (nt * vt);
    elementwise_kernel<nt, vt><<<grid, nt, 0, current_stream()>>>(static_cast<int>(N), f);
    RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
}

/// f(input_0, ..., input_n-1), each read at its offset from its base pointer
template <typename func_t, typename traits = function_traits<func_t>>
__device__ typename traits::result_type invoke(const func_t& f, char* const data[], const int32_t offsets[]) {
    return [&]<size_t... I>(std::index_sequence<I...>) {
        return f(*reinterpret_cast<const typename traits::template arg<I>::type*>(data[I] + offsets[I])...);
    }(std::make_index_sequence<traits::arity>{});
}

/// Same, converting each input from its runtime dtype
template <typename func_t, typename traits = function_traits<func_t>>
__device__ typename traits::result_type invoke(
    const func_t& f, char* const data[], const int32_t offsets[], const DtypeKind dtypes[]
) {
    return [&]<size_t... I>(std::index_sequence<I...>) {
        return f(fetch_and_cast<typename traits::template arg<I>::type>(dtypes[I], data[I] + offsets[I])...);
    }(std::make_index_sequence<traits::arity>{});
}

// ============================================================================
// Entry points
// ============================================================================

template <typename func_t>
void gpu_kernel_impl(TensorIterator& iter, const func_t& f) {
    using traits = function_traits<func_t>;
    using result_t = typename traits::result_type;
    constexpr int ntensors = traits::arity + 1;

    RAXIS_INTERNAL_ASSERT(iter.num_inputs() == static_cast<int>(traits::arity));
    RAXIS_INTERNAL_ASSERT(iter.num_outputs() == 1);

    std::array<char*, ntensors> data;
    for (int i = 0; i < ntensors; i++) {
        data[i] = static_cast<char*>(iter.data_ptr(i));
    }
    const int64_t numel = iter.numel();
    auto offset_calc = make_offset_calculator<ntensors>(iter);

    if (!needs_dynamic_casting<func_t>(iter)) {
        if (iter.is_contiguous()) {
            return launch_vectorized_kernel(numel, f, data);
        }
        // Fewer elements per thread for wide types, as PyTorch
        constexpr int unroll = sizeof(result_t) >= 4 ? 2 : 4;
        launch_legacy_kernel<128, unroll>(numel, [=] GPU_LAMBDA(int idx) {
            auto offsets = offset_calc.get(idx);
            *reinterpret_cast<result_t*>(data[0] + offsets[0]) = invoke(f, &data[1], &offsets[1]);
        });
        return;
    }

    // PyTorch has a faster unrolled kernel for contiguous operands that need
    // a cast; here they take the generic one
    std::array<DtypeKind, ntensors> dtypes;
    for (int i = 0; i < ntensors; i++) {
        dtypes[i] = iter.dtype(i).kind();
    }
    launch_legacy_kernel<128, 4>(numel, [=] GPU_LAMBDA(int idx) {
        auto offsets = offset_calc.get(idx);
        result_t result = invoke(f, &data[1], &offsets[1], &dtypes[1]);
        cast_and_store<result_t>(dtypes[0], data[0] + offsets[0], result);
    });
}

/// Runs `f` (a __device__ callable taking the inputs' values, returning the
/// output's) on every element of the iterator, on the operands' device.
/// From: aten/src/ATen/native/cuda/Loops.cuh (gpu_kernel)
template <typename func_t>
void gpu_kernel(TensorIterator& iter, const func_t& f) {
    for (int arg = 0; arg < iter.num_operands(); arg++) {
        RAXIS_INTERNAL_ASSERT(iter.device(arg).is_cuda(),
            "argument ", arg, ": expected a CUDA device but found ", iter.device(arg));
    }
    if (iter.numel() == 0) {
        return;
    }
    if (!iter.can_use_32bit_indexing()) {
        // Halves the iterator along its largest dim until every piece can
        // use 32-bit offsets
        auto first_half = iter.split(iter.get_dim_to_split());
        gpu_kernel(*first_half, f);
        gpu_kernel(iter, f);
        return;
    }
    DeviceGuard guard(iter.device(0));
    gpu_kernel_impl(iter, f);
}

// Binary functors with one operand bound to a CPU scalar
// From: aten/src/ATen/native/cuda/Loops.cuh

template <typename arg1_t, typename arg2_t, typename return_t, typename func_t>
struct AUnaryFunctor {
    using traits = function_traits<func_t>;
    __device__ return_t operator()(arg2_t b) const { return f(a, b); }
    AUnaryFunctor(func_t f_, arg1_t a_) : f(f_), a(a_) {}
    func_t f;
    arg1_t a;
};

template <typename arg1_t, typename arg2_t, typename return_t, typename func_t>
struct BUnaryFunctor {
    __device__ return_t operator()(arg1_t a) const { return f(a, b); }
    BUnaryFunctor(func_t f_, arg2_t b_) : f(f_), b(b_) {}
    func_t f;
    arg2_t b;
};

/// gpu_kernel for a binary functor where either input may be a CPU scalar
/// (`x * 2` wraps the 2 in a 0-dim CPU tensor). Its value is read on the
/// host and passed in the functor's arguments, so the kernel does not read a
/// broadcast operand at all and runs as a unary op.
template <typename func_t>
void gpu_kernel_with_scalars(TensorIterator& iter, const func_t& f) {
    using traits = function_traits<func_t>;
    static_assert(traits::arity == 2, "gpu_kernel_with_scalars only supports two input arguments");
    using arg1_t = typename traits::template arg<0>::type;
    using arg2_t = typename traits::template arg<1>::type;
    using return_t = typename traits::result_type;

    if (iter.is_cpu_scalar(1)) {
        auto a = iter.scalar_value<arg1_t>(1);
        iter.remove_operand(1);
        gpu_kernel(iter, AUnaryFunctor<arg1_t, arg2_t, return_t, func_t>(f, a));
    } else if (iter.is_cpu_scalar(2)) {
        auto b = iter.scalar_value<arg2_t>(2);
        iter.remove_operand(2);
        gpu_kernel(iter, BUnaryFunctor<arg1_t, arg2_t, return_t, func_t>(f, b));
    } else {
        gpu_kernel(iter, f);
    }
}

} // namespace raxis::cuda
