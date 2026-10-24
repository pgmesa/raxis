#include <cstring>
#include <type_traits>

#include <raxis/ops/tensor_advanced_indexing.h>
#include <raxis/ops/tensor_transformations.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/ops/cpu/loops.h>

#include <raxis/core/dtype.h>
#include <raxis/core/scalar.h>
#include <raxis/util/exception.h>
#include <raxis/tensor/tensor_iterator.h>


namespace raxis {

inline namespace CPU_CAPABILITY {

namespace {

// From: aten/src/ATen/native/cpu/IndexKernelUtils.h
//
// x[i0, i1] with index tensors i0, i1 reads, for every element of the
// result, self at the offset i0[k] * stride0 + i1[k] * stride1. The front-end
// (AdvancedIndex) lays this out for a TensorIterator: self is restrided with
// stride 0 over the indexed dims, so the iterator only walks the dims that
// are not indexed, and the index tensors are extra operands broadcast to the
// result shape. The kernel adds the offset given by the index values.
//
// Operands: data[0] = output, data[1] = input, data[2..] = index tensors.
// For index, the output is the result and the input is the restrided self;
// for index_put, the output is the restrided self and the input is the
// values, so the offset applies to data[0] or data[1] depending on the op.

struct Indexer {
    int64_t num_indexers;
    char** indexers;
    const int64_t* indexer_strides;
    const int64_t* original_sizes;
    const int64_t* original_strides;

    // Byte offset of element idx of the loop
    int64_t get(int64_t idx) const {
        int64_t offset = 0;
        for (int64_t j = 0; j < num_indexers; j++) {
            int64_t value = *reinterpret_cast<const int64_t*>(indexers[j] + idx * indexer_strides[j]);
            int64_t size = original_sizes[j];
            RAXIS_CHECK_INDEX(value >= -size && value < size,
                "index ", value, " is out of bounds for dimension ", j, " with size ", size);
            if (value < 0) {
                value += size;
            }
            offset += value * original_strides[j];
        }
        return offset;
    }

    // get() for a single index tensor, without the loop over indexers
    int64_t get_1(int64_t idx) const {
        int64_t value = *reinterpret_cast<const int64_t*>(indexers[0] + idx * indexer_strides[0]);
        const int64_t size = original_sizes[0];
        RAXIS_CHECK_INDEX(value >= -size && value < size,
            "index ", value, " is out of bounds for dimension 0 with size ", size);
        if (value < 0) {
            value += size;
        }
        return value * original_strides[0];
    }
};

// Same grain as PyTorch: each element does little work, so threads start
// earlier than with the default grain
constexpr int64_t kIndexGrainSize = 3000;

// f(dst, src, offset, n, dst_stride, src_stride): n elements that share the
// index offset. Every element shares it when the index operands do not move
// along the inner loop (x[idx] on the rows of a matrix: the whole row comes
// from the same index), and f can then copy a contiguous run at once.
template <typename func_t>
void cpu_index_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride,
                      const func_t& f, bool serial_execution = false) {
    const int ntensor = iter.num_operands();
    auto loop = [&](char** data, const int64_t* strides, int64_t n) {
        Indexer indexer{ntensor - 2, &data[2], &strides[2], index_size.data(), index_stride.data()};
        bool is_constant_index = true;
        for (int arg = 2; arg < ntensor; arg++) {
            is_constant_index &= strides[arg] == 0;
        }
        if (is_constant_index) {
            f(data[0], data[1], indexer.get(0), n, strides[0], strides[1]);
        } else if (indexer.num_indexers == 1) {
            for (int64_t i = 0; i < n; i++) {
                f(data[0] + strides[0] * i, data[1] + strides[1] * i, indexer.get_1(i), 1, 0, 0);
            }
        } else {
            for (int64_t i = 0; i < n; i++) {
                f(data[0] + strides[0] * i, data[1] + strides[1] * i, indexer.get(i), 1, 0, 0);
            }
        }
    };
    if (serial_execution) {
        iter.serial_for_each(loop, {0, iter.numel()});
    } else {
        iter.for_each(loop, kIndexGrainSize);
    }
}

template <typename scalar_t>
void add_into(scalar_t& out, scalar_t value) {
    // A sum of bools is their OR (PyTorch writes += and lets it convert)
    if constexpr (std::is_same_v<scalar_t, bool>) {
        out = out || value;
    } else {
        out += value;
    }
}

} // namespace

void index_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride) {
    iter.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_index_kernel(iter, index_size, index_stride,
            [](char* dst, char* src, int64_t offset, int64_t n, int64_t dst_stride, int64_t src_stride) {
                src += offset;
                if (n > 1 && dst_stride == sizeof(scalar_t) && src_stride == sizeof(scalar_t)) {
                    std::memcpy(dst, src, n * sizeof(scalar_t));
                    return;
                }
                for (int64_t i = 0; i < n; i++) {
                    *reinterpret_cast<scalar_t*>(dst + i * dst_stride) =
                        *reinterpret_cast<const scalar_t*>(src + i * src_stride);
                }
            });
    });
}

// Without accumulate, a repeated index is written by more than one thread and
// any of the values may win, as in PyTorch. With accumulate, the additions
// to a repeated index must not race: PyTorch runs them serially (or with
// atomic float adds, which makes the sum order vary between runs); here they
// run serially, so the result is deterministic, unless the indices cannot
// repeat (they come from a mask), and then threads never share a position.
void index_put_kernel(TensorIterator& iter, IntArrayRef index_size, IntArrayRef index_stride,
                      bool accumulate, bool unique_indices) {
    iter.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        if (accumulate) {
            cpu_index_kernel(iter, index_size, index_stride,
                [](char* dst, char* src, int64_t offset, int64_t n, int64_t dst_stride, int64_t src_stride) {
                    dst += offset;
                    if (dst_stride == sizeof(scalar_t) && src_stride == sizeof(scalar_t)) {
                        // Typed loop over a contiguous run: vectorizable
                        auto* out = reinterpret_cast<scalar_t*>(dst);
                        const auto* in = reinterpret_cast<const scalar_t*>(src);
                        for (int64_t i = 0; i < n; i++) {
                            add_into(out[i], in[i]);
                        }
                        return;
                    }
                    for (int64_t i = 0; i < n; i++) {
                        add_into(
                            *reinterpret_cast<scalar_t*>(dst + i * dst_stride),
                            *reinterpret_cast<const scalar_t*>(src + i * src_stride));
                    }
                }, /*serial_execution=*/!unique_indices);
        } else {
            cpu_index_kernel(iter, index_size, index_stride,
                [](char* dst, char* src, int64_t offset, int64_t n, int64_t dst_stride, int64_t src_stride) {
                    dst += offset;
                    if (n > 1 && dst_stride == sizeof(scalar_t) && src_stride == sizeof(scalar_t)) {
                        std::memcpy(dst, src, n * sizeof(scalar_t));
                        return;
                    }
                    for (int64_t i = 0; i < n; i++) {
                        *reinterpret_cast<scalar_t*>(dst + i * dst_stride) =
                            *reinterpret_cast<const scalar_t*>(src + i * src_stride);
                    }
                });
        }
    });
}

// From: aten/src/ATen/native/cpu/IndexKernel.cpp (cpu_masked_fill_kernel)
// PyTorch writes `if (mask) dst = value`, a conditional store the compiler
// may not vectorize (it cannot add stores the source does not make). Writing
// `dst = mask ? value : dst` stores every element and becomes a vector
// blend when both operands are contiguous.
void masked_fill_kernel(TensorIterator& iter, Scalar value) {
    iter.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        const scalar_t fill = value.to<scalar_t>();
        auto loop = [&](char** data, const int64_t* strides, int64_t n) {
            char* dst = data[0];
            const char* mask = data[1];
            if (strides[0] == sizeof(scalar_t) && strides[1] == sizeof(bool)) {
                auto* out = reinterpret_cast<scalar_t*>(dst);
                const auto* m = reinterpret_cast<const bool*>(mask);
                for (int64_t i = 0; i < n; i++) {
                    out[i] = m[i] ? fill : out[i];
                }
                return;
            }
            for (int64_t i = 0; i < n; i++) {
                if (*reinterpret_cast<const bool*>(mask + i * strides[1])) {
                    *reinterpret_cast<scalar_t*>(dst + i * strides[0]) = fill;
                }
            }
        };
        iter.for_each(loop);
    });
}

// From: aten/src/ATen/native/cpu/IndexKernel.cpp (flip_kernel)
// A copy: the flip is in the output strides set up by ops::flip. The dummy
// input is never read. When the inner dim is not flipped, both operands are
// contiguous there and cpu_kernel takes its vectorized path.
void flip_kernel(TensorIterator& iter) {
    iter.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel(iter, [](scalar_t a, scalar_t /*dummy*/) -> scalar_t { return a; });
    });
}

} // namespace CPU_CAPABILITY

REGISTER_DISPATCH(index_stub, &index_kernel);
REGISTER_DISPATCH(index_put_stub, &index_put_kernel);
REGISTER_DISPATCH(masked_fill_stub, &masked_fill_kernel);
REGISTER_DISPATCH(flip_stub, &flip_kernel);

} // namespace raxis
