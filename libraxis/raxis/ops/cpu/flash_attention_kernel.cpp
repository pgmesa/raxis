#include <raxis/ops/attention.h>
#include <raxis/ops/cpu_blas.h>
#include <raxis/ops/cpu/vec/vec.h>
#include <raxis/ops/cpu/vec/functional_base.h>
#include <raxis/ops/dispatch_stub.h>

#include <raxis/core/dtype.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/tensor.h>
#include <raxis/util/parallel.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>


namespace raxis {

inline namespace CPU_CAPABILITY {

using namespace vec;
using cpublas::TransposeType;

// From: aten/src/ATen/native/cpu/FlashAttentionKernel.cpp
//
// Attention is softmax(q k^T * scale + mask) v. The math version builds the
// [L, S] score matrix per (batch, head), runs a softmax over it and
// multiplies by v: three passes over L * S elements that live in memory.
// Flash attention never builds it. Each task takes a block of q_split
// queries and walks the keys in blocks of kv_split:
//
//   qk  = q_blk @ k_blk^T            (gemm, [q_split, kv_split] in a per-thread buffer)
//   max = max(max, rowmax(qk))       online softmax: running max and sum per row
//   qk  = exp(qk - max)
//   sum = sum * exp(old_max - max) + rowsum(qk)
//   dst = dst * exp(old_max - max) + qk @ v_blk
//
// and divides dst by sum at the end. The rescale by exp(old_max - max)
// fixes what was accumulated with an older max, so the result equals the
// two-pass softmax. The block of scores stays in L1/L2, and with is_causal
// the key blocks after the diagonal are not even computed. The backward
// recomputes the probabilities from the saved logsumexp (max + log(sum))
// instead of storing them.

// out = a * val + b, with b a single broadcast value when its stride is 0
template <bool is_b_stride_zero, typename T>
inline void scale_attn_mask_fusion(const T* a, const T* b, int64_t size, T* out, T val) {
    using Vec = Vectorized<T>;
    const Vec vec_scale(val);
    int64_t i = 0;
    for (; i < size - (size % Vec::size()); i += Vec::size()) {
        const Vec b_vec = is_b_stride_zero ? Vec(b[0]) : Vec::loadu(b + i);
        (Vec::loadu(a + i) * vec_scale + b_vec).store(out + i);
    }
    for (; i < size; ++i) {
        out[i] = a[i] * val + (is_b_stride_zero ? b[0] : b[i]);
    }
}

// out = exp(a - val), then val = sum(out)
template <typename T>
inline void exp_reduce_sum_fusion(const T* a, int64_t size, T* out, T& val) {
    using Vec = Vectorized<T>;
    const Vec vec_max(val);
    Vec vec_sum(T(0));
    const int64_t vec_end = size - (size % Vec::size());
    int64_t i = 0;
    for (; i < vec_end; i += Vec::size()) {
        const Vec e = (Vec::loadu(a + i) - vec_max).exp_u20();
        vec_sum = vec_sum + e;
        e.store(out + i);
    }
    T sum = vec_reduce_all<T>([](Vec x, Vec y) { return x + y; }, vec_sum);
    for (; i < size; ++i) {
        const T e = std::exp(a[i] - val);
        sum += e;
        out[i] = e;
    }
    val = sum;
}

// out = a * scale, then max = max(out). Two vector accumulators per step
// (torch's VectorizedN<T, 2>) break the dependency chain of the max
template <typename T>
inline void mul_reduce_max_fusion(const T* a, T scale, int64_t size, T* out, T& max) {
    using Vec = Vectorized<T>;
    constexpr int64_t kStep = 2 * Vec::size();
    const Vec vec_scale(scale);
    const T neg_inf = -std::numeric_limits<T>::infinity();
    Vec vec_max0(neg_inf);
    Vec vec_max1(neg_inf);
    int64_t i = 0;
    for (; i < size - (size % kStep); i += kStep) {
        const Vec x0 = Vec::loadu(a + i) * vec_scale;
        const Vec x1 = Vec::loadu(a + i + Vec::size()) * vec_scale;
        vec_max0 = maximum(vec_max0, x0);
        vec_max1 = maximum(vec_max1, x1);
        x0.store(out + i);
        x1.store(out + i + Vec::size());
    }
    for (; i < size - (size % Vec::size()); i += Vec::size()) {
        const Vec x = Vec::loadu(a + i) * vec_scale;
        vec_max0 = maximum(vec_max0, x);
        x.store(out + i);
    }
    T tail_max = neg_inf;
    for (; i < size; ++i) {
        const T x = a[i] * scale;
        tail_max = std::max(tail_max, x);
        out[i] = x;
    }
    const T reduced = vec_reduce_all<T>([](Vec x, Vec y) { return maximum(x, y); }, maximum(vec_max0, vec_max1));
    // A NaN in q k^T must reach the output, std::max would drop it
    max = std::isnan(reduced) ? std::numeric_limits<T>::quiet_NaN() : std::max(tail_max, reduced);
}

// Strides of the 4-D mask; 0 for size 1 dims, whatever their stride
struct MaskStrides {
    int64_t b = 0, h = 0, m = 0, n = 0;

    explicit MaskStrides(const Tensor& mask) {
        if (!mask.defined()) {
            return;
        }
        b = mask.shape()[0] > 1 ? mask.stride(0) : 0;
        h = mask.shape()[1] > 1 ? mask.stride(1) : 0;
        m = mask.shape()[2] > 1 ? mask.stride(2) : 0;
        n = mask.shape()[3] > 1 ? mask.stride(3) : 0;
    }
};

// qk[row] = qk[row] * scale + mask row, for the rows of a block
template <typename T>
void add_attn_mask(
    T* qk_data, const T* mask_block, const MaskStrides& ms, int64_t q_block_size, int64_t kv_block_size, T scale
) {
    for (int64_t row = 0; row < q_block_size; ++row) {
        T* qk_row = qk_data + row * kv_block_size;
        const T* mask_row = mask_block + row * ms.m;
        if (ms.n == 0) {
            scale_attn_mask_fusion</*is_b_stride_zero=*/true>(qk_row, mask_row, kv_block_size, qk_row, scale);
        } else {
            scale_attn_mask_fusion</*is_b_stride_zero=*/false>(qk_row, mask_row, kv_block_size, qk_row, scale);
        }
    }
}

// The query block size grows with the sequence: longer blocks amortize
// reading each key block, shorter ones give more tasks to the threads
inline int64_t q_split_size_for(int64_t q_size) {
    return q_size >= 768 ? 256 : q_size >= 192 ? 64 : 32;
}
constexpr int64_t kv_split_size = 512;

// From: cpu_flash_attention (the float / double path; the packing only
// serves the reduced float types)
template <typename scalar_t>
void flash_attention_impl(
    const Tensor& output, const Tensor& logsumexp, const Tensor& query, const Tensor& key,
    const Tensor& value, bool is_causal, const Tensor& attn_mask, double scale
) {
    using Vec = Vectorized<scalar_t>;
    constexpr scalar_t neg_inf = -std::numeric_limits<scalar_t>::infinity();
    const scalar_t scaling_factor = static_cast<scalar_t>(scale);

    const int64_t batch_size = query.shape()[0];
    const int64_t q_size = query.shape()[1];
    const int64_t kv_size = value.shape()[1];
    const int64_t num_head = query.shape()[2];
    const int64_t head_size = query.shape()[3];
    const bool has_attn_mask = attn_mask.defined();

    const int64_t q_stride_b = query.stride(0), q_stride_m = query.stride(1), q_stride_h = query.stride(2);
    const int64_t k_stride_b = key.stride(0), k_stride_n = key.stride(1), k_stride_h = key.stride(2);
    const int64_t v_stride_b = value.stride(0), v_stride_n = value.stride(1), v_stride_h = value.stride(2);
    const int64_t o_stride_b = output.stride(0), o_stride_m = output.stride(1), o_stride_h = output.stride(2);
    const int64_t l_stride_b = logsumexp.stride(0), l_stride_m = logsumexp.stride(1), l_stride_h = logsumexp.stride(2);
    const MaskStrides ms(attn_mask);

    const int64_t q_split = std::min(q_split_size_for(q_size), q_size);
    const int64_t kv_split = std::min(kv_split_size, kv_size);
    const int64_t q_slice = (q_size + q_split - 1) / q_split;

    // Per thread: the [q_split, kv_split] scores, the running max and sum
    // per row and the [q_split, head_size] unnormalized output
    const int num_threads = get_num_threads();
    const int64_t size_per_thread = q_split * kv_split + q_split + q_split + q_split * head_size;
    Tensor buf = empty({num_threads, size_per_thread}, query.options());

    const scalar_t* q_data = query.data_ptr<scalar_t>();
    const scalar_t* k_data = key.data_ptr<scalar_t>();
    const scalar_t* v_data = value.data_ptr<scalar_t>();
    const scalar_t* mask_data = has_attn_mask ? attn_mask.data_ptr<scalar_t>() : nullptr;
    scalar_t* out_data = output.mutable_data_ptr<scalar_t>();
    scalar_t* lse_data = logsumexp.mutable_data_ptr<scalar_t>();
    scalar_t* buf_data = buf.mutable_data_ptr<scalar_t>();

    // One task per (batch, head, query block)
    parallel_for(0, batch_size * num_head * q_slice, 1, [&](int64_t begin, int64_t end) {
        scalar_t* qk_data = buf_data + get_thread_num() * size_per_thread;
        scalar_t* qk_max_data = qk_data + q_split * kv_split;
        scalar_t* qk_sum_data = qk_max_data + q_split;
        scalar_t* dst_data = qk_sum_data + q_split;

        // Task index -> (b, h, q_block) once, then stepped as an odometer
        int64_t q_block = begin % q_slice;
        int64_t h = (begin / q_slice) % num_head;
        int64_t b = begin / (q_slice * num_head);
        for (int64_t task = begin; task < end; ++task) {
            const int64_t m = q_block * q_split;
            const int64_t q_block_size = std::min(q_split, q_size - m);
            std::fill_n(qk_max_data, q_block_size, neg_inf);
            std::fill_n(qk_sum_data, q_block_size, scalar_t(0));
            // With is_causal, query m + row sees the keys up to m + row
            const int64_t num_keys = is_causal ? std::min(m + q_block_size, kv_size) : kv_size;
            const scalar_t* q_block_data = q_data + b * q_stride_b + h * q_stride_h + m * q_stride_m;
            const scalar_t* k_head = k_data + b * k_stride_b + h * k_stride_h;
            const scalar_t* v_head = v_data + b * v_stride_b + h * v_stride_h;

            for (int64_t n = 0; n < num_keys; n += kv_split) {
                const int64_t kv_block_size = std::min(kv_split, kv_size - n);
                // qk = q_blk @ k_blk^T, row-major [q_block_size, kv_block_size].
                // BLAS is column-major: there qk is the [kv, q] matrix k_blk @ q_blk^T
                cpublas::gemm(
                    TransposeType::Transpose, TransposeType::NoTranspose,
                    kv_block_size, q_block_size, head_size,
                    scalar_t(1),
                    k_head + n * k_stride_n, k_stride_n,
                    q_block_data, q_stride_m,
                    scalar_t(0),
                    qk_data, kv_block_size);
                // The block that crosses the diagonal: -inf after each row's last key
                if (is_causal && num_keys - n <= kv_split) {
                    for (int64_t row = 0; row < q_block_size; ++row) {
                        const int64_t last_col = m + row - n;
                        if (last_col + 1 < kv_block_size) {
                            std::fill_n(qk_data + row * kv_block_size + last_col + 1,
                                        kv_block_size - last_col - 1, neg_inf);
                        }
                    }
                }
                if (has_attn_mask) {
                    const scalar_t* mask_block = mask_data + b * ms.b + h * ms.h + m * ms.m + (ms.n == 0 ? 0 : n);
                    add_attn_mask(qk_data, mask_block, ms, q_block_size, kv_block_size, scaling_factor);
                }
                for (int64_t row = 0; row < q_block_size; ++row) {
                    scalar_t* qk_row = qk_data + row * kv_block_size;
                    scalar_t block_max;
                    if (has_attn_mask) {
                        block_max = reduce_all<scalar_t>([](Vec x, Vec y) { return maximum(x, y); },
                                                         qk_row, kv_block_size);
                    } else {
                        // The scale is applied here, fused with the max
                        mul_reduce_max_fusion(qk_row, scaling_factor, kv_block_size, qk_row, block_max);
                    }
                    const scalar_t new_max = std::max(qk_max_data[row], block_max);
                    if (new_max == neg_inf) {
                        // Every key so far masked out: avoid exp(-inf - -inf) = NaN
                        std::fill_n(qk_row, kv_block_size, scalar_t(0));
                        continue;
                    }
                    scalar_t block_sum = new_max;
                    exp_reduce_sum_fusion(qk_row, kv_block_size, qk_row, block_sum);
                    const scalar_t correction = std::exp(qk_max_data[row] - new_max);
                    qk_sum_data[row] = block_sum + correction * qk_sum_data[row];
                    qk_max_data[row] = new_max;
                    if (n > 0) {
                        scalar_t* dst_row = dst_data + row * head_size;
                        map([correction](Vec x) { return x * Vec(correction); }, dst_row, dst_row, head_size);
                    }
                }
                // dst (+)= qk @ v_blk, row-major [q_block_size, head_size]
                cpublas::gemm(
                    TransposeType::NoTranspose, TransposeType::NoTranspose,
                    head_size, q_block_size, kv_block_size,
                    scalar_t(1),
                    v_head + n * v_stride_n, v_stride_n,
                    qk_data, kv_block_size,
                    n == 0 ? scalar_t(0) : scalar_t(1),
                    dst_data, head_size);
            }

            scalar_t* out_block = out_data + b * o_stride_b + h * o_stride_h + m * o_stride_m;
            scalar_t* lse_block = lse_data + b * l_stride_b + h * l_stride_h + m * l_stride_m;
            for (int64_t row = 0; row < q_block_size; ++row) {
                // A row with every key masked out has sum 0 and max -inf:
                // its output is 0 (as the math path's safe softmax) and its
                // logsumexp stays finite
                const scalar_t max = qk_max_data[row] == neg_inf ? scalar_t(0) : qk_max_data[row];
                const scalar_t sum = qk_sum_data[row] == 0 ? scalar_t(1) : qk_sum_data[row];
                const scalar_t sum_reciprocal = 1 / sum;
                map([sum_reciprocal](Vec x) { return x * Vec(sum_reciprocal); },
                    out_block + row * o_stride_m, dst_data + row * head_size, head_size);
                lse_block[row * l_stride_m] = max + std::log(sum);
            }

            if (++q_block == q_slice) {
                q_block = 0;
                if (++h == num_head) {
                    h = 0;
                    ++b;
                }
            }
        }
    });
}

// From: cpu_flash_attention_backward
//
// With P = softmax(S), S = q k^T * scale + mask, and dO the output gradient:
//   dV = P^T dO
//   dP = dO V^T
//   dS = P * (dP - rowsum(dO * O))       softmax backward; rowsum(P * dP) = rowsum(dO * O)
//   dQ = dS K * scale,  dK = dS^T Q * scale
// P is rebuilt block by block from logsumexp, never stored. One task per
// (batch, head) owns its rows of dK and dV, so the key blocks are added
// into without races.
template <typename scalar_t>
void flash_attention_backward_impl(
    const Tensor& grad_q, const Tensor& grad_k, const Tensor& grad_v, const Tensor& grad_out,
    const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& out,
    const Tensor& logsumexp, bool is_causal, const Tensor& attn_mask, double scale
) {
    using Vec = Vectorized<scalar_t>;
    const scalar_t scaling_factor = static_cast<scalar_t>(scale);

    const int64_t batch_size = query.shape()[0];
    const int64_t q_size = query.shape()[1];
    const int64_t kv_size = value.shape()[1];
    const int64_t num_head = query.shape()[2];
    const int64_t head_size = query.shape()[3];
    const bool has_attn_mask = attn_mask.defined();

    const int64_t q_stride_b = query.stride(0), q_stride_m = query.stride(1), q_stride_h = query.stride(2);
    const int64_t k_stride_b = key.stride(0), k_stride_n = key.stride(1), k_stride_h = key.stride(2);
    const int64_t v_stride_b = value.stride(0), v_stride_n = value.stride(1), v_stride_h = value.stride(2);
    const int64_t o_stride_b = out.stride(0), o_stride_m = out.stride(1), o_stride_h = out.stride(2);
    const int64_t l_stride_b = logsumexp.stride(0), l_stride_m = logsumexp.stride(1), l_stride_h = logsumexp.stride(2);
    const int64_t go_stride_b = grad_out.stride(0), go_stride_m = grad_out.stride(1), go_stride_h = grad_out.stride(2);
    const int64_t gq_stride_b = grad_q.stride(0), gq_stride_m = grad_q.stride(1), gq_stride_h = grad_q.stride(2);
    const int64_t gk_stride_b = grad_k.stride(0), gk_stride_n = grad_k.stride(1), gk_stride_h = grad_k.stride(2);
    const int64_t gv_stride_b = grad_v.stride(0), gv_stride_n = grad_v.stride(1), gv_stride_h = grad_v.stride(2);
    const MaskStrides ms(attn_mask);

    const int64_t q_split = std::min(q_split_size_for(q_size), q_size);
    const int64_t kv_split = std::min(kv_split_size, kv_size);

    // Per thread: the probabilities and their gradient for one block, and
    // rowsum(dO * O) of the query block
    const int num_threads = get_num_threads();
    const int64_t size_per_thread = 2 * q_split * kv_split + q_split;
    Tensor buf = empty({num_threads, size_per_thread}, query.options());

    scalar_t* grad_q_data = grad_q.mutable_data_ptr<scalar_t>();
    scalar_t* grad_k_data = grad_k.mutable_data_ptr<scalar_t>();
    scalar_t* grad_v_data = grad_v.mutable_data_ptr<scalar_t>();
    const scalar_t* grad_out_data = grad_out.data_ptr<scalar_t>();
    const scalar_t* q_data = query.data_ptr<scalar_t>();
    const scalar_t* k_data = key.data_ptr<scalar_t>();
    const scalar_t* v_data = value.data_ptr<scalar_t>();
    const scalar_t* out_data = out.data_ptr<scalar_t>();
    const scalar_t* lse_data = logsumexp.data_ptr<scalar_t>();
    const scalar_t* mask_data = has_attn_mask ? attn_mask.data_ptr<scalar_t>() : nullptr;
    scalar_t* buf_data = buf.mutable_data_ptr<scalar_t>();

    parallel_for(0, batch_size * num_head, 1, [&](int64_t begin, int64_t end) {
        scalar_t* attn_data = buf_data + get_thread_num() * size_per_thread;
        scalar_t* grad_attn_data = attn_data + q_split * kv_split;
        scalar_t* dsum_data = grad_attn_data + q_split * kv_split;

        for (int64_t task = begin; task < end; ++task) {
            const int64_t b = task / num_head;
            const int64_t h = task % num_head;
            const scalar_t* q_head = q_data + b * q_stride_b + h * q_stride_h;
            const scalar_t* k_head = k_data + b * k_stride_b + h * k_stride_h;
            const scalar_t* v_head = v_data + b * v_stride_b + h * v_stride_h;
            const scalar_t* out_head = out_data + b * o_stride_b + h * o_stride_h;
            const scalar_t* go_head = grad_out_data + b * go_stride_b + h * go_stride_h;
            const scalar_t* lse_head = lse_data + b * l_stride_b + h * l_stride_h;
            scalar_t* gq_head = grad_q_data + b * gq_stride_b + h * gq_stride_h;
            scalar_t* gk_head = grad_k_data + b * gk_stride_b + h * gk_stride_h;
            scalar_t* gv_head = grad_v_data + b * gv_stride_b + h * gv_stride_h;

            for (int64_t m = 0; m < q_size; m += q_split) {
                const int64_t q_block_size = std::min(q_split, q_size - m);
                for (int64_t row = 0; row < q_block_size; ++row) {
                    dsum_data[row] = map2_reduce_all<scalar_t>(
                        [](Vec x, Vec y) { return x * y; }, [](Vec x, Vec y) { return x + y; },
                        go_head + (m + row) * go_stride_m, out_head + (m + row) * o_stride_m, head_size);
                }
                const int64_t num_keys = is_causal ? std::min(m + q_block_size, kv_size) : kv_size;
                for (int64_t n = 0; n < num_keys; n += kv_split) {
                    const int64_t kv_block_size = std::min(kv_split, kv_size - n);
                    // attn = scale * q_blk @ k_blk^T
                    cpublas::gemm(
                        TransposeType::Transpose, TransposeType::NoTranspose,
                        kv_block_size, q_block_size, head_size,
                        scaling_factor,
                        k_head + n * k_stride_n, k_stride_n,
                        q_head + m * q_stride_m, q_stride_m,
                        scalar_t(0),
                        attn_data, kv_block_size);
                    if (has_attn_mask) {
                        const scalar_t* mask_block = mask_data + b * ms.b + h * ms.h + m * ms.m + (ms.n == 0 ? 0 : n);
                        add_attn_mask(attn_data, mask_block, ms, q_block_size, kv_block_size, scalar_t(1));
                    }
                    // attn = exp(attn - logsumexp): the softmax probabilities
                    for (int64_t row = 0; row < q_block_size; ++row) {
                        const scalar_t normalizer = lse_head[(m + row) * l_stride_m];
                        scalar_t* attn_row = attn_data + row * kv_block_size;
                        map([normalizer](Vec x) { return (x - Vec(normalizer)).exp(); }, attn_row, attn_row, kv_block_size);
                    }
                    if (is_causal && num_keys - n <= kv_split) {
                        for (int64_t row = 0; row < q_block_size; ++row) {
                            const int64_t last_col = m + row - n;
                            if (last_col + 1 < kv_block_size) {
                                std::fill_n(attn_data + row * kv_block_size + last_col + 1,
                                            kv_block_size - last_col - 1, scalar_t(0));
                            }
                        }
                    }
                    // grad_v += attn^T @ grad_out
                    cpublas::gemm(
                        TransposeType::NoTranspose, TransposeType::Transpose,
                        head_size, kv_block_size, q_block_size,
                        scalar_t(1),
                        go_head + m * go_stride_m, go_stride_m,
                        attn_data, kv_block_size,
                        scalar_t(1),
                        gv_head + n * gv_stride_n, gv_stride_n);
                    // grad_attn = grad_out @ v_blk^T
                    cpublas::gemm(
                        TransposeType::Transpose, TransposeType::NoTranspose,
                        kv_block_size, q_block_size, head_size,
                        scalar_t(1),
                        v_head + n * v_stride_n, v_stride_n,
                        go_head + m * go_stride_m, go_stride_m,
                        scalar_t(0),
                        grad_attn_data, kv_block_size);
                    // grad_attn = attn * (grad_attn - dsum)
                    for (int64_t row = 0; row < q_block_size; ++row) {
                        const scalar_t d = dsum_data[row];
                        scalar_t* grad_attn_row = grad_attn_data + row * kv_block_size;
                        map2([d](Vec attn, Vec grad_attn) { return attn * (grad_attn - Vec(d)); },
                             grad_attn_row, attn_data + row * kv_block_size, grad_attn_row, kv_block_size);
                    }
                    // grad_q += scale * grad_attn @ k_blk
                    cpublas::gemm(
                        TransposeType::NoTranspose, TransposeType::NoTranspose,
                        head_size, q_block_size, kv_block_size,
                        scaling_factor,
                        k_head + n * k_stride_n, k_stride_n,
                        grad_attn_data, kv_block_size,
                        scalar_t(1),
                        gq_head + m * gq_stride_m, gq_stride_m);
                    // grad_k += scale * grad_attn^T @ q_blk
                    cpublas::gemm(
                        TransposeType::NoTranspose, TransposeType::Transpose,
                        head_size, kv_block_size, q_block_size,
                        scaling_factor,
                        q_head + m * q_stride_m, q_stride_m,
                        grad_attn_data, kv_block_size,
                        scalar_t(1),
                        gk_head + n * gk_stride_n, gk_stride_n);
                }
            }
        }
    });
}

void flash_attention_kernel(
    const Tensor& output, const Tensor& logsumexp, const Tensor& query, const Tensor& key,
    const Tensor& value, bool is_causal, const Tensor& attn_mask, double scale
) {
    query.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        flash_attention_impl<scalar_t>(output, logsumexp, query, key, value, is_causal, attn_mask, scale);
    });
}

void flash_attention_backward_kernel(
    const Tensor& grad_q, const Tensor& grad_k, const Tensor& grad_v, const Tensor& grad_out,
    const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& out,
    const Tensor& logsumexp, bool is_causal, const Tensor& attn_mask, double scale
) {
    query.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        flash_attention_backward_impl<scalar_t>(
            grad_q, grad_k, grad_v, grad_out, query, key, value, out, logsumexp, is_causal, attn_mask, scale);
    });
}

} // namespace CPU_CAPABILITY

ALSO_REGISTER_AVX512_DISPATCH(flash_attention_stub, &flash_attention_kernel);
ALSO_REGISTER_AVX512_DISPATCH(flash_attention_backward_stub, &flash_attention_backward_kernel);

} // namespace raxis
