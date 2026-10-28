#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include <raxis/core/device_guard.h>
#include <raxis/core/philox.h>
#include <raxis/cuda/cuda_exception.h>
#include <raxis/cuda/cuda_functions.h>
#include <raxis/ops/attention.h>
#include <raxis/ops/binary_ops.h>
#include <raxis/ops/reduce_ops.h>
#include <raxis/tensor/tensor.h>


// Fused attention on the GPU: softmax(Q K^T * scale + bias) with dropout,
// times V, without storing the [L, S] scores.
//
// For float32 PyTorch runs its memory-efficient attention here
// (aten/src/ATen/native/transformers/cuda/mem_eff_attention, xformers'
// CUTLASS kernels; FlashAttention-2 only takes fp16 / bf16). That code is
// thousands of lines of CUTLASS templates, so this is a rewrite of the same
// algorithm, FlashAttention-2's loop (Dao, 2023), in plain CUDA:
//
// Forward: a block owns 64 queries and walks the keys in tiles of 64. For
// each tile it computes the scores, updates every row's running max m and
// running sum l of exp(score - m) (the online softmax: when m grows, the
// sum and the output accumulated so far are rescaled by exp(m_old - m_new)),
// and adds P V to the output. At the end the output is divided by l and
// logsumexp = m + log(l) is stored: the backward rebuilds P = exp(S - lse)
// from it instead of storing P.
//
// Backward: with D_i = sum_e dO[i][e] O[i][e],
//   dV = P^T dO,  dP = dO V^T,  dS = P * (dP - D),  dQ = dS K,  dK = dS^T Q
// (times scale for dQ and dK). FlashAttention-2 runs one block per key
// tile and adds each block's part of dQ with atomics, so the sum order (and
// the result) changes between runs. Here a second kernel, one block per
// query tile, computes dQ on its own: it recomputes P and dP (two more tile
// products), but the gradients are the same on every run.
//
// Dropout: element (b, h, i, j) is kept when its Philox word is below
// p_keep * 2^32. The words are a function of the position only (word c of
// block (row, j / 64) * 16 + j % 16, c = j % 64 / 16), so the backward draws
// the same mask without storing it, and a thread that owns columns
// tx, tx + 16, tx + 32, tx + 48 of a tile gets its 4 words from one block.
//
// Tiles live in shared memory with rows of 64 + 4 floats: 16-byte aligned
// for float4 loads, and 8 threads reading float4s of 8 different rows hit
// 8 different groups of 4 banks. Scores are in base 2 (scale * log2(e)
// folded in) so the exponentials are exp2. Head dims up to 64 (nanoGPT and
// GPT-2 use 64); SDPA takes the math path above that.

namespace raxis {
namespace {

constexpr int kBlockM = 64;         // queries per tile
constexpr int kBlockN = 64;         // keys per tile
constexpr int kHeadDim = 64;        // largest head dim, the tiles are padded to it
constexpr int kThreads = 256;       // 16 x 16 threads, each owns 4 x 4 elements of a tile
constexpr int kLd = kHeadDim + 4;   // row stride of a tile in shared memory
constexpr int kTile = kBlockM * kLd;
constexpr float kLog2e = 1.4426950408889634f;
constexpr float kLn2 = 0.6931471805599453f;
// INFINITY is 1e300 * 1e300 in MSVC's headers, which nvcc warns about
constexpr float kInf = std::numeric_limits<float>::infinity();

static_assert(kBlockM == kBlockN && kBlockN == kHeadDim, "every tile has the same shape");

// A [B, H, rows, E] tensor with unit last stride
template <typename T>
struct Operand {
    T* data;
    int64_t stride_b, stride_h, stride_l;

    __device__ T* row(int b, int h, int i) const {
        return data + b * stride_b + h * stride_h + i * stride_l;
    }
};

struct Params {
    Operand<const float> q, k, v, grad_out;
    Operand<float> out, grad_q, grad_k, grad_v;
    float* lse;            // [B, H, L], natural log
    const float* delta;    // [B, H, L]
    const float* bias;     // [B or 1, H or 1, L, S] or null
    int64_t bias_stride_b, bias_stride_h, bias_stride_l, bias_stride_s;
    int B, H, L, S, E;
    float scale;
    float scale_log2;      // scale * log2(e)
    bool causal;
    bool dropout;
    uint32_t keep_threshold;
    float dropout_scale;   // 1 / p_keep
    uint64_t seed, offset;
    int col_tiles;         // ceil(S / 64), for the Philox counters
    bool aligned;          // every row of q, k, v and grad_out starts at a 16-byte boundary
};

// Thread (ty, tx) of the 16 x 16 grid. In a score tile it owns rows
// ty + 16 r and columns tx + 16 c (r, c < 4); in an output tile, rows
// ty + 16 r and columns 4 tx + c.
__device__ inline int tile_tx() { return threadIdx.x % 16; }
__device__ inline int tile_ty() { return threadIdx.x / 16; }

// rows [row0, row0 + 64) of t into a tile, zero past `rows` and past E.
// With 16-byte aligned rows each thread moves float4s (4 per tile instead
// of 16 loads)
__device__ void load_tile(float* tile, const Operand<const float>& t, int b, int h, int row0, int rows, int E,
                          bool aligned) {
    if (aligned) {
        constexpr int kVecs = kHeadDim / 4;
        for (int idx = threadIdx.x; idx < kBlockM * kVecs; idx += kThreads) {
            const int r = idx / kVecs;
            const int e = idx % kVecs * 4;
            float4 x = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (row0 + r < rows && e < E) {
                x = *reinterpret_cast<const float4*>(t.row(b, h, row0 + r) + e);
            }
            *reinterpret_cast<float4*>(&tile[r * kLd + e]) = x;
        }
        return;
    }
    for (int idx = threadIdx.x; idx < kBlockM * kHeadDim; idx += kThreads) {
        const int r = idx / kHeadDim;
        const int e = idx % kHeadDim;
        float x = 0.0f;
        if (row0 + r < rows && e < E) {
            x = t.row(b, h, row0 + r)[e];
        }
        tile[r * kLd + e] = x;
    }
}

// acc[r][c] += sum_e A[ty + 16 r][e] * B[tx + 16 c][e]: A B^T, the scores
__device__ void tile_abt(float (&acc)[4][4], const float* A, const float* B) {
    const int tx = tile_tx();
    const int ty = tile_ty();
    #pragma unroll 4
    for (int e = 0; e < kHeadDim; e += 4) {
        float4 a[4];
        float4 b[4];
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            a[r] = *reinterpret_cast<const float4*>(&A[(ty + 16 * r) * kLd + e]);
        }
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            b[c] = *reinterpret_cast<const float4*>(&B[(tx + 16 * c) * kLd + e]);
        }
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                acc[r][c] = fmaf(a[r].x, b[c].x, acc[r][c]);
                acc[r][c] = fmaf(a[r].y, b[c].y, acc[r][c]);
                acc[r][c] = fmaf(a[r].z, b[c].z, acc[r][c]);
                acc[r][c] = fmaf(a[r].w, b[c].w, acc[r][c]);
            }
        }
    }
}

// acc[r][c] += sum_k A[ty + 16 r][k] * B[k][4 tx + c]: A B, the outputs
__device__ void tile_ab(float (&acc)[4][4], const float* A, const float* B) {
    const int tx = tile_tx();
    const int ty = tile_ty();
    #pragma unroll 2
    for (int k = 0; k < kBlockN; k += 4) {
        float4 a[4];
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            a[r] = *reinterpret_cast<const float4*>(&A[(ty + 16 * r) * kLd + k]);
        }
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            const float4 b = *reinterpret_cast<const float4*>(&B[(k + u) * kLd + 4 * tx]);
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                const float a_ru = u == 0 ? a[r].x : u == 1 ? a[r].y : u == 2 ? a[r].z : a[r].w;
                acc[r][0] = fmaf(a_ru, b.x, acc[r][0]);
                acc[r][1] = fmaf(a_ru, b.y, acc[r][1]);
                acc[r][2] = fmaf(a_ru, b.z, acc[r][2]);
                acc[r][3] = fmaf(a_ru, b.w, acc[r][3]);
            }
        }
    }
}

// Reductions over the 16 threads that share a row (lanes that differ in
// their low 4 bits)
__device__ inline float row_max(float x) {
    #pragma unroll
    for (int offset = 8; offset > 0; offset /= 2) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset));
    }
    return x;
}

__device__ inline float row_sum(float x) {
    #pragma unroll
    for (int offset = 8; offset > 0; offset /= 2) {
        x += __shfl_xor_sync(0xffffffff, x, offset);
    }
    return x;
}

// Scores of the tile at (q0, k0) in base 2: scale, bias, and -inf where
// masked (causal, or a key past S)
__device__ void scale_and_mask(float (&s)[4][4], const Params& p, int b, int h, int q0, int k0) {
    const int tx = tile_tx();
    const int ty = tile_ty();
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        const int i = q0 + ty + 16 * r;
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            const int j = k0 + tx + 16 * c;
            float x = s[r][c] * p.scale_log2;
            if (p.bias != nullptr && i < p.L && j < p.S) {
                x += p.bias[b * p.bias_stride_b + h * p.bias_stride_h + i * p.bias_stride_l
                            + j * p.bias_stride_s] * kLog2e;
            }
            if (j >= p.S || (p.causal && j > i)) {
                x = -kInf;
            }
            s[r][c] = x;
        }
    }
}

// keep[r][c] for the tile at (q0, k0), from one Philox block per row
__device__ void dropout_mask(bool (&keep)[4][4], const Params& p, int bh, int q0, int k0) {
    const int tx = tile_tx();
    const int ty = tile_ty();
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        const int64_t i = q0 + ty + 16 * r;
        const uint64_t block = (static_cast<uint64_t>(bh * static_cast<int64_t>(p.L) + i) * p.col_tiles
                                + k0 / kBlockN) * 16 + tx;
        const PhiloxBlock words = philox4x32_10(p.offset + block, p.seed);
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            keep[r][c] = words[c] < p.keep_threshold;
        }
    }
}

__global__ void __launch_bounds__(kThreads) attention_forward_kernel(Params p) {
    extern __shared__ float4 smem4[];
    float* smem = reinterpret_cast<float*>(smem4);
    float* q_tile = smem;
    float* k_tile = smem + kTile;   // also P, once the scores are computed
    float* v_tile = smem + 2 * kTile;
    const int tx = tile_tx();
    const int ty = tile_ty();
    const int bh = blockIdx.y;
    const int b = bh / p.H;
    const int h = bh % p.H;
    const int q0 = blockIdx.x * kBlockM;

    load_tile(q_tile, p.q, b, h, q0, p.L, p.E, p.aligned);
    float o[4][4] = {};
    float m[4];
    float l[4] = {};
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        m[r] = -kInf;
    }
    // Causal: keys after the tile's last query are all masked
    const int kv_end = p.causal ? ::min(p.S, q0 + kBlockM) : p.S;
    for (int k0 = 0; k0 < kv_end; k0 += kBlockN) {
        __syncthreads();
        load_tile(k_tile, p.k, b, h, k0, p.S, p.E, p.aligned);
        load_tile(v_tile, p.v, b, h, k0, p.S, p.E, p.aligned);
        __syncthreads();

        float s[4][4] = {};
        tile_abt(s, q_tile, k_tile);
        scale_and_mask(s, p, b, h, q0, k0);
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            const float m_new = fmaxf(m[r], row_max(fmaxf(fmaxf(s[r][0], s[r][1]), fmaxf(s[r][2], s[r][3]))));
            // A row masked so far has m = -inf: exp2(x - (-inf)) would be
            // NaN, so it subtracts 0 instead (everything is still 0)
            const float base = m_new == -kInf ? 0.0f : m_new;
            const float alpha = exp2f(m[r] - base);
            float sum = 0.0f;
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                s[r][c] = exp2f(s[r][c] - base);
                sum += s[r][c];
            }
            l[r] = l[r] * alpha + row_sum(sum);
            m[r] = m_new;
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                o[r][c] *= alpha;
            }
        }
        // The normalizer l counts every key; dropout only removes terms of P V
        if (p.dropout) {
            bool keep[4][4];
            dropout_mask(keep, p, bh, q0, k0);
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    s[r][c] = keep[r][c] ? s[r][c] * p.dropout_scale : 0.0f;
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                k_tile[(ty + 16 * r) * kLd + tx + 16 * c] = s[r][c];
            }
        }
        __syncthreads();
        tile_ab(o, k_tile, v_tile);
    }

    #pragma unroll
    for (int r = 0; r < 4; r++) {
        const int i = q0 + ty + 16 * r;
        if (i >= p.L) {
            continue;
        }
        // A row with every key masked gets zeros, as the math path's safe softmax
        const float inv = l[r] > 0.0f ? 1.0f / l[r] : 0.0f;
        float* out = p.out.row(b, h, i);
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            if (4 * tx + c < p.E) {
                out[4 * tx + c] = o[r][c] * inv;
            }
        }
        if (tx == 0) {
            p.lse[static_cast<int64_t>(bh) * p.L + i] = l[r] > 0.0f ? (m[r] + log2f(l[r])) * kLn2 : kInf;
        }
    }
}

// P = exp(S - lse) and dS = P * (dP - D) of the tile at (q0, k0); the
// dropped-out P (what multiplied V in the forward) goes to p_dropped
__device__ void backward_tile(
    float (&prob)[4][4], float (&ds)[4][4], const Params& p, const float* q_tile, const float* k_tile,
    const float* grad_out_tile, const float* v_tile, const float (&lse)[4], const float (&delta)[4],
    int b, int h, int q0, int k0
) {
    float s[4][4] = {};
    tile_abt(s, q_tile, k_tile);
    scale_and_mask(s, p, b, h, q0, k0);
    float dp[4][4] = {};
    tile_abt(dp, grad_out_tile, v_tile);
    bool keep[4][4];
    if (p.dropout) {
        dropout_mask(keep, p, b * p.H + h, q0, k0);
    }
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            // Masked scores are -inf; rows past L have lse = +inf: P = 0
            const float pr = exp2f(s[r][c] - lse[r]);
            float g = dp[r][c];
            float pd = pr;
            if (p.dropout) {
                g = keep[r][c] ? g * p.dropout_scale : 0.0f;
                pd = keep[r][c] ? pr * p.dropout_scale : 0.0f;
            }
            ds[r][c] = pr * (g - delta[r]);
            prob[r][c] = pd;
        }
    }
}

// lse (in base 2) and D of the rows of the tile at q0 this thread owns
__device__ void load_row_stats(float (&lse)[4], float (&delta)[4], const Params& p, int bh, int q0) {
    const int ty = tile_ty();
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        const int i = q0 + ty + 16 * r;
        const int64_t idx = static_cast<int64_t>(bh) * p.L + i;
        lse[r] = i < p.L ? p.lse[idx] * kLog2e : kInf;
        delta[r] = i < p.L ? p.delta[idx] : 0.0f;
    }
}

__device__ void store_tile(const Operand<float>& dst, const float (&acc)[4][4], int b, int h, int row0, int rows,
                           int E, float factor) {
    const int tx = tile_tx();
    const int ty = tile_ty();
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        const int i = row0 + ty + 16 * r;
        if (i >= rows) {
            continue;
        }
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            if (4 * tx + c < E) {
                dst.row(b, h, i)[4 * tx + c] = acc[r][c] * factor;
            }
        }
    }
}

// dK and dV of one key tile: a loop over the query tiles that see it
__global__ void __launch_bounds__(kThreads) attention_backward_kv_kernel(Params p) {
    extern __shared__ float4 smem4[];
    float* smem = reinterpret_cast<float*>(smem4);
    float* k_tile = smem;
    float* v_tile = smem + kTile;
    float* q_tile = smem + 2 * kTile;
    float* grad_out_tile = smem + 3 * kTile;
    float* t_tile = smem + 4 * kTile;   // P^T, then dS^T
    const int tx = tile_tx();
    const int ty = tile_ty();
    const int bh = blockIdx.y;
    const int b = bh / p.H;
    const int h = bh % p.H;
    const int k0 = blockIdx.x * kBlockN;

    load_tile(k_tile, p.k, b, h, k0, p.S, p.E, p.aligned);
    load_tile(v_tile, p.v, b, h, k0, p.S, p.E, p.aligned);
    float dk[4][4] = {};
    float dv[4][4] = {};
    // Causal: queries before the tile's first key see none of it
    for (int q0 = p.causal ? k0 / kBlockM * kBlockM : 0; q0 < p.L; q0 += kBlockM) {
        __syncthreads();
        load_tile(q_tile, p.q, b, h, q0, p.L, p.E, p.aligned);
        load_tile(grad_out_tile, p.grad_out, b, h, q0, p.L, p.E, p.aligned);
        float lse[4];
        float delta[4];
        load_row_stats(lse, delta, p, bh, q0);
        __syncthreads();

        float prob[4][4];
        float ds[4][4];
        backward_tile(prob, ds, p, q_tile, k_tile, grad_out_tile, v_tile, lse, delta, b, h, q0, k0);
        // dV[j] += sum_i P[i][j] dO[i]: P transposed into shared memory
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                t_tile[(tx + 16 * c) * kLd + ty + 16 * r] = prob[r][c];
            }
        }
        __syncthreads();
        tile_ab(dv, t_tile, grad_out_tile);
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                t_tile[(tx + 16 * c) * kLd + ty + 16 * r] = ds[r][c];
            }
        }
        __syncthreads();
        tile_ab(dk, t_tile, q_tile);
    }
    store_tile(p.grad_k, dk, b, h, k0, p.S, p.E, p.scale);
    store_tile(p.grad_v, dv, b, h, k0, p.S, p.E, 1.0f);
}

// dQ of one query tile: a loop over the key tiles it sees
__global__ void __launch_bounds__(kThreads) attention_backward_q_kernel(Params p) {
    extern __shared__ float4 smem4[];
    float* smem = reinterpret_cast<float*>(smem4);
    float* q_tile = smem;
    float* grad_out_tile = smem + kTile;
    float* k_tile = smem + 2 * kTile;
    float* v_tile = smem + 3 * kTile;   // also dS, once dP is computed
    const int tx = tile_tx();
    const int ty = tile_ty();
    const int bh = blockIdx.y;
    const int b = bh / p.H;
    const int h = bh % p.H;
    const int q0 = blockIdx.x * kBlockM;

    load_tile(q_tile, p.q, b, h, q0, p.L, p.E, p.aligned);
    load_tile(grad_out_tile, p.grad_out, b, h, q0, p.L, p.E, p.aligned);
    float lse[4];
    float delta[4];
    load_row_stats(lse, delta, p, bh, q0);
    float dq[4][4] = {};
    const int kv_end = p.causal ? ::min(p.S, q0 + kBlockM) : p.S;
    for (int k0 = 0; k0 < kv_end; k0 += kBlockN) {
        __syncthreads();
        load_tile(k_tile, p.k, b, h, k0, p.S, p.E, p.aligned);
        load_tile(v_tile, p.v, b, h, k0, p.S, p.E, p.aligned);
        __syncthreads();

        float prob[4][4];
        float ds[4][4];
        backward_tile(prob, ds, p, q_tile, k_tile, grad_out_tile, v_tile, lse, delta, b, h, q0, k0);
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            #pragma unroll
            for (int c = 0; c < 4; c++) {
                v_tile[(ty + 16 * r) * kLd + tx + 16 * c] = ds[r][c];
            }
        }
        __syncthreads();
        tile_ab(dq, v_tile, k_tile);
    }
    store_tile(p.grad_q, dq, b, h, q0, p.L, p.E, p.scale);
}

Operand<const float> operand(const Tensor& t) {
    return {t.data_ptr<float>(), t.stride(0), t.stride(1), t.stride(2)};
}

Operand<float> mutable_operand(const Tensor& t) {
    return {t.mutable_data_ptr<float>(), t.stride(0), t.stride(1), t.stride(2)};
}

Params make_params(const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& attn_bias,
                   double dropout_p, bool is_causal, double scale, uint64_t seed, uint64_t offset) {
    Params p{};
    p.q = operand(query);
    p.k = operand(key);
    p.v = operand(value);
    p.B = static_cast<int>(query.shape()[0]);
    p.H = static_cast<int>(query.shape()[1]);
    p.L = static_cast<int>(query.shape()[2]);
    p.S = static_cast<int>(key.shape()[2]);
    p.E = static_cast<int>(query.shape()[3]);
    p.scale = static_cast<float>(scale);
    p.scale_log2 = static_cast<float>(scale) * kLog2e;
    p.causal = is_causal;
    if (attn_bias.defined()) {
        p.bias = attn_bias.data_ptr<float>();
        // A mask shared by the batch or the heads has size 1 there
        p.bias_stride_b = attn_bias.shape()[0] == 1 ? 0 : attn_bias.stride(0);
        p.bias_stride_h = attn_bias.shape()[1] == 1 ? 0 : attn_bias.stride(1);
        p.bias_stride_l = attn_bias.stride(2);
        p.bias_stride_s = attn_bias.stride(3);
    }
    p.dropout = dropout_p > 0.0;
    const double p_keep = 1.0 - dropout_p;
    p.keep_threshold = static_cast<uint32_t>(std::min(std::ldexp(p_keep, 32), 4294967295.0));
    p.dropout_scale = p_keep == 0.0 ? 0.0f : static_cast<float>(1.0 / p_keep);
    p.seed = seed;
    p.offset = offset;
    p.col_tiles = (p.S + kBlockN - 1) / kBlockN;
    RAXIS_CHECK(p.E <= kHeadDim, "efficient attention on CUDA supports head dims up to ", kHeadDim);
    auto aligned = [&](const Tensor& t) {
        return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0 && t.stride(0) % 4 == 0
               && t.stride(1) % 4 == 0 && t.stride(2) % 4 == 0;
    };
    p.aligned = p.E % 4 == 0 && aligned(query) && aligned(key) && aligned(value);
    RAXIS_CHECK(static_cast<int64_t>(p.B) * p.H <= 65535, "efficient attention: batch * heads over 65535");
    return p;
}

template <typename Kernel>
void launch(Kernel kernel, int tiles, const Params& p, int smem_tiles) {
    const int smem_bytes = smem_tiles * kTile * static_cast<int>(sizeof(float));
    RAXIS_CHECK_NOT_IMPLEMENTED(
        static_cast<size_t>(smem_bytes) <= cuda::current_device_properties().sharedMemPerBlockOptin,
        "efficient attention needs ", smem_bytes, " bytes of shared memory per block (compute capability 8.0+)");
    RAXIS_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    const dim3 grid(static_cast<unsigned>(tiles), static_cast<unsigned>(p.B * p.H));
    kernel<<<grid, kThreads, smem_bytes, cuda::current_stream()>>>(p);
    RAXIS_CUDA_KERNEL_LAUNCH_CHECK();
}

void efficient_attention_kernel_cuda(
    const Tensor& output, const Tensor& logsumexp, const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& attn_bias, double dropout_p, bool is_causal, double scale, uint64_t seed, uint64_t offset
) {
    DeviceGuard guard(query.device());
    Params p = make_params(query, key, value, attn_bias, dropout_p, is_causal, scale, seed, offset);
    p.out = mutable_operand(output);
    p.lse = logsumexp.mutable_data_ptr<float>();
    launch(attention_forward_kernel, (p.L + kBlockM - 1) / kBlockM, p, 3);
}

void efficient_attention_backward_kernel_cuda(
    const Tensor& grad_query, const Tensor& grad_key, const Tensor& grad_value, const Tensor& grad_output,
    const Tensor& query, const Tensor& key, const Tensor& value, const Tensor& output, const Tensor& logsumexp,
    const Tensor& attn_bias, double dropout_p, bool is_causal, double scale, uint64_t seed, uint64_t offset
) {
    DeviceGuard guard(query.device());
    // D_i = dO_i . O_i: the row term of the softmax backward
    const Tensor delta = ops::sum(ops::mul(grad_output, output), {-1}).contiguous();
    Params p = make_params(query, key, value, attn_bias, dropout_p, is_causal, scale, seed, offset);
    p.grad_out = operand(grad_output);
    p.aligned = p.aligned && reinterpret_cast<uintptr_t>(grad_output.data_ptr()) % 16 == 0
                && grad_output.stride(0) % 4 == 0 && grad_output.stride(1) % 4 == 0
                && grad_output.stride(2) % 4 == 0;
    p.lse = logsumexp.mutable_data_ptr<float>();
    p.delta = delta.data_ptr<float>();
    p.grad_q = mutable_operand(grad_query);
    p.grad_k = mutable_operand(grad_key);
    p.grad_v = mutable_operand(grad_value);
    launch(attention_backward_kv_kernel, (p.S + kBlockN - 1) / kBlockN, p, 5);
    launch(attention_backward_q_kernel, (p.L + kBlockM - 1) / kBlockM, p, 4);
}

} // anonymous namespace

REGISTER_DISPATCH(efficient_attention_stub, &efficient_attention_kernel_cuda);
REGISTER_DISPATCH(efficient_attention_backward_stub, &efficient_attention_backward_kernel_cuda);

} // namespace raxis
