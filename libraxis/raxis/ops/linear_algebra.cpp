#include <raxis/ops/linear_algebra.h>

#include <algorithm>
#include <cstdint>
#include <numeric>

#include <raxis/ops/cpu_blas.h>
#include <raxis/ops/device_check.h>

#include <raxis/util/exception.h>
#include <raxis/util/parallel.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/tensor_iterator.h>


namespace raxis {

namespace {

using cpublas::TransposeType;


// Core GEMM dispatch: result = beta * self + alpha * (m1 @ m2)
//
// Analyses strides of result, m1, m2 to pick transpose flags that let BLAS
// consume the data without copying. Uses the identity
//     C = A*B  <=>  C^T = B^T * A^T
// to handle row-major results.
static void addmm_impl_cpu_(
    Tensor& result, 
    const Tensor& self,
    Tensor m1, 
    Tensor m2,
    const Scalar& beta, 
    const Scalar& alpha
) {
    check_cpu_only("addmm", result, self, m1, m2);
    RAXIS_INTERNAL_ASSERT(self.ndim() == 2 && m1.ndim() == 2 && m2.ndim() == 2);
    RAXIS_CHECK(
        m1.dtype() == m2.dtype(),
        "expected m1 and m2 to have the same dtype, but got: ",
        m1.dtype(), " != ", m2.dtype());

    const auto self_sizes = self.shape();
    auto m1_strides = m1.strides();
    auto m1_sizes = m1.shape();
    auto m2_strides = m2.strides();
    auto m2_sizes = m2.shape();

    RAXIS_CHECK(
        self_sizes[0] == m1_sizes[0] && self_sizes[1] == m2_sizes[1],
        "input shape is incompatible with matrix multiplication (",
        m1_sizes[0], "x", m1_sizes[1], " @ ",
        m2_sizes[0], "x", m2_sizes[1], " != ",
        self_sizes[0], "x", self_sizes[1], ")");

    RAXIS_CHECK(
        m1_sizes[1] == m2_sizes[0],
        "mat1 and mat2 shapes cannot be multiplied (",
        m1_sizes[0], "x", m1_sizes[1], " and ",
        m2_sizes[0], "x", m2_sizes[1], ")");

    if (!result.defined()) {
        result = empty({self_sizes[0], self_sizes[1]}, self.options());
    }

    const auto result_strides = result.strides();
    const auto result_sizes = result.shape();

    if (result.numel() == 0) return;

    if (m1_sizes[1] == 0) {
        if (beta.to<double>() == 0.0) {
            result.zero_();
        } else {
            if (!self.is_same(result)) result.copy_(self);
            result.mul_(beta);
        }
        return;
    }

    if (beta.to<double>() != 0.0 && !self.is_same(result)) {
        result.copy_(self);
    }

    // -- Determine layout of C (result) --
    bool transpose_c = false;
    Tensor c;

    if (
        result_strides[0] == 1 &&
        (result_sizes[1] == 1 || result_strides[1] >= std::max<int64_t>(1, result_sizes[0]))
    ) {
        transpose_c = false;
        c = result;
    } else if (result_strides[1] == 1 &&
               (result_sizes[0] == 1 || result_strides[0] >= std::max<int64_t>(1, result_sizes[1])))
    {
        std::swap(m1, m2);
        std::swap(m1_sizes, m2_sizes);
        std::swap(m1_strides, m2_strides);
        transpose_c = true;
        c = result;
    } else {
        transpose_c = false;
        c = empty({result_sizes[0], result_sizes[1]}, result.options());
        if (beta.to<double>() != 0.0) c.copy_(result);
    }

    const int64_t m = result_sizes[transpose_c ? 1 : 0];
    const int64_t n = result_sizes[transpose_c ? 0 : 1];
    const int64_t k = m1_sizes[transpose_c ? 0 : 1];

    // -- Determine transpose flag for A (m1) --
    bool transpose_a = false;
    Tensor a;
    if (
        m1_strides[transpose_c ? 1 : 0] == 1 &&
        m1_strides[transpose_c ? 0 : 1] >= std::max<int64_t>(1, m)
    ) {
        transpose_a = false;
        a = m1;
    } else if (
        m1_strides[transpose_c ? 0 : 1] == 1 &&
        m1_strides[transpose_c ? 1 : 0] >= std::max<int64_t>(1, k)
    ) {
        transpose_a = true;
        a = m1;
    } else {
        transpose_a = !transpose_c;
        a = m1.contiguous();
    }

    // -- Determine transpose flag for B (m2) --
    bool transpose_b = false;
    Tensor b;
    if (
        m2_strides[transpose_c ? 1 : 0] == 1 &&
        m2_strides[transpose_c ? 0 : 1] >= std::max<int64_t>(1, k)
    ) {
        transpose_b = false;
        b = m2;
    } else if (
        m2_strides[transpose_c ? 0 : 1] == 1 &&
        m2_strides[transpose_c ? 1 : 0] >= std::max<int64_t>(1, n)
    ) {
        transpose_b = true;
        b = m2;
    } else {
        transpose_b = !transpose_c;
        b = m2.contiguous();
    }

    const int64_t lda = std::max(transpose_a ? k : m, a.strides()[(transpose_a == transpose_c) ? 1 : 0]);
    const int64_t ldb = std::max(transpose_b ? n : k, b.strides()[(transpose_b == transpose_c) ? 1 : 0]);
    const int64_t ldc = std::max(m, c.strides()[transpose_c ? 0 : 1]);

    auto ta = transpose_a ? TransposeType::Transpose : TransposeType::NoTranspose;
    auto tb = transpose_b ? TransposeType::Transpose : TransposeType::NoTranspose;

    result.dtype().dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpublas::gemm(
            ta, tb, m, n, k,
            alpha.to<scalar_t>(),
            a.data_ptr<scalar_t>(), lda,
            b.data_ptr<scalar_t>(), ldb,
            beta.to<scalar_t>(),
            c.mutable_data_ptr<scalar_t>(), ldc
        );
    });

    if (!c.is_same(result)) {
        result.copy_(c);
    }
}

// From: aten/src/ATen/native/LinearAlgebra.cpp (baddbmm_cpu_kernel)
// Tiny matrices: a BLAS call costs more than the product itself, so a plain
// triple loop, parallel over the batch
template <typename scalar_t, bool is_bmm>
static void baddbmm_cpu_kernel(
    const Tensor& result, const Tensor& self, const Tensor& mat2,
    const Scalar& beta_, const Scalar& alpha_
) {
    const int64_t bs = result.shape()[0];
    const int64_t is = result.shape()[1];
    const int64_t js = result.shape()[2];
    const int64_t ks = self.shape()[2];
    const scalar_t alpha = alpha_.to<scalar_t>();
    const scalar_t beta = beta_.to<scalar_t>();

    scalar_t* r = result.mutable_data_ptr<scalar_t>();
    const scalar_t* s = self.data_ptr<scalar_t>();
    const scalar_t* m = mat2.data_ptr<scalar_t>();
    IntArrayRef rst = result.strides();
    IntArrayRef sst = self.strides();
    IntArrayRef mst = mat2.strides();

    int64_t grain_size = std::max<int64_t>(detail::GRAIN_SIZE / (is * js * ks), 1);
    parallel_for(0, bs, grain_size, [&](int64_t b_begin, int64_t b_end) {
        for (int64_t b = b_begin; b < b_end; ++b) {
            for (int64_t i = 0; i < is; ++i) {
                scalar_t* r2 = r + b * rst[0] + i * rst[1];
                const scalar_t* s2 = s + b * sst[0] + i * sst[1];
                for (int64_t j = 0; j < js; ++j) {
                    const scalar_t* m2 = m + b * mst[0] + j * mst[2];
                    scalar_t acc = 0;
                    for (int64_t k = 0; k < ks; ++k) {
                        acc += s2[k * sst[2]] * m2[k * mst[1]];
                    }
                    scalar_t& out = r2[j * rst[2]];
                    if constexpr (is_bmm) {
                        out = acc;
                    } else {
                        // beta == 0 ignores the old value, also a NaN
                        out = beta == scalar_t(0) ? alpha * acc : out * beta + alpha * acc;
                    }
                }
            }
        }
    });
}

// From: aten/src/ATen/native/LinearAlgebra.cpp (baddbmm_with_gemm_)
// One batched gemm for the whole batch. As in addmm_impl_cpu_, the
// row-major result is computed as its column-major transpose:
// C^T = mat2^T * mat1^T, so mat2 is BLAS's A and mat1 its B.
static void baddbmm_with_gemm_(
    const Tensor& result, const Tensor& mat1, const Tensor& mat2,
    const Scalar& beta_, const Scalar& alpha_
) {
    RAXIS_INTERNAL_ASSERT(result.is_contiguous());
    IntArrayRef result_sizes = result.shape();
    IntArrayRef result_strides = result.strides();
    IntArrayRef mat1_strides = mat1.strides();
    IntArrayRef mat2_strides = mat2.strides();
    IntArrayRef mat1_sizes = mat1.shape();
    IntArrayRef mat2_sizes = mat2.shape();

    auto is_transposed = [](IntArrayRef strides, IntArrayRef sizes) {
        return strides[1] == 1 && strides[2] >= sizes[1];
    };
    const bool transpose_a = is_transposed(mat2_strides, mat2_sizes);
    const bool transpose_b = is_transposed(mat1_strides, mat1_sizes);

    const int64_t batch_size = mat1_sizes[0];
    const int64_t m = result_sizes[2];
    const int64_t n = result_sizes[1];
    const int64_t k = mat2_sizes[1];

    const int64_t lda = mat2_strides[transpose_a ? 2 : 1];
    const int64_t ldb = mat1_strides[transpose_b ? 2 : 1];
    const int64_t ldc = result_strides[1];

    result.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpublas::gemm_batched_with_stride(
            transpose_a ? TransposeType::Transpose : TransposeType::NoTranspose,
            transpose_b ? TransposeType::Transpose : TransposeType::NoTranspose,
            batch_size, m, n, k, alpha_.to<scalar_t>(),
            mat2.data_ptr<scalar_t>(), lda, mat2_strides[0],
            mat1.data_ptr<scalar_t>(), ldb, mat1_strides[0],
            beta_.to<scalar_t>(),
            result.mutable_data_ptr<scalar_t>(), ldc, result_strides[0]);
    });
}

// From: aten/src/ATen/native/LinearAlgebra.cpp (bmm_out_or_baddbmm_)
// Three strategies by size and layout: a naive kernel for tiny matrices, one
// batched gemm when every batch item is a plain or transposed matrix, and
// otherwise one addmm per batch item
static void bmm_impl_(
    Tensor& self_or_result,
    const Tensor& batch1, 
    const Tensor& batch2,
    const Scalar& beta, 
    const Scalar& alpha,
    bool is_bmm_out
) {
    check_cpu_only("bmm", self_or_result, batch1, batch2);
    const int64_t bs = batch1.shape()[0];
    const int64_t contraction_size = batch1.shape()[2];
    const int64_t res_rows = batch1.shape()[1];
    const int64_t res_cols = batch2.shape()[2];

    if (self_or_result.numel() == 0) return;

    if (contraction_size == 0) {
        if (is_bmm_out || beta.to<double>() == 0.0) {
            self_or_result.zero_();
        } else {
            self_or_result.mul_(beta);
        }
        return;
    }

    // Strides do not matter on a size 1 dim
    auto batch_items_contiguous_or_transposed = [](const Tensor& t) {
        IntArrayRef sizes = t.shape();
        IntArrayRef strides = t.strides();
        return (strides[2] == 1 && (sizes[1] == 1 || strides[1] >= sizes[2]))
            || (strides[1] == 1 && (sizes[2] == 1 || strides[2] >= sizes[1]));
    };

    if (contraction_size * res_rows * res_cols < 400) {
        self_or_result.dtype().dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
            if (is_bmm_out) {
                baddbmm_cpu_kernel<scalar_t, true>(self_or_result, batch1, batch2, beta, alpha);
            } else {
                baddbmm_cpu_kernel<scalar_t, false>(self_or_result, batch1, batch2, beta, alpha);
            }
        });
        return;
    }

    if (self_or_result.dtype().is_floating()
            && batch_items_contiguous_or_transposed(batch1)
            && batch_items_contiguous_or_transposed(batch2)
            && self_or_result.is_contiguous()) {
        baddbmm_with_gemm_(self_or_result, batch1, batch2, is_bmm_out ? 0 : beta, alpha);
        return;
    }

    if (is_bmm_out) {
        for (int64_t b = 0; b < bs; ++b) {
            auto r = self_or_result.select(0, b);
            addmm_impl_cpu_(r, r, batch1.select(0, b), batch2.select(0, b), 0, 1);
        }
    } else {
        for (int64_t b = 0; b < bs; ++b) {
            auto r = self_or_result.select(0, b);
            addmm_impl_cpu_(r, r, batch1.select(0, b), batch2.select(0, b), beta, alpha);
        }
    }
}

static void addbmm_impl_(
    Tensor& result, 
    const Tensor& self,
    const Tensor& batch1, 
    const Tensor& batch2,
    const Scalar& beta, 
    const Scalar& alpha
) {
    RAXIS_CHECK(batch1.ndim() == 3, "batch1 must be a 3D tensor");
    RAXIS_CHECK(batch2.ndim() == 3, "batch2 must be a 3D tensor");
    RAXIS_CHECK(batch1.shape()[0] == batch2.shape()[0],
        "batch1 and batch2 must have same number of batches, got ",
        batch1.shape()[0], " and ", batch2.shape()[0]);
    RAXIS_CHECK(batch1.shape()[2] == batch2.shape()[1],
        "incompatible matrix sizes for bmm (",
        batch1.shape()[1], "x", batch1.shape()[2], " and ",
        batch2.shape()[1], "x", batch2.shape()[2], ")");
 
    const int64_t dim1 = batch1.shape()[1];
    const int64_t dim2 = batch2.shape()[2];
    RAXIS_CHECK(self.shape()[0] == dim1 && self.shape()[1] == dim2,
        "self tensor does not match matmul output shape");
 
    if (!result.defined()) {
        result = empty({dim1, dim2}, self.options());
    }
 
    if (beta.to<double>() != 0.0 && !self.is_same(result)) {
        result.copy_(self);
    }
 
    const int64_t num_batches = batch1.shape()[0];
 
    if (num_batches == 0) {
        if (beta.to<double>() != 0.0)
            result.mul_(beta);
        else
            result.zero_();
        return;
    }
 
    Scalar adjusted_beta = beta;
    for (int64_t b = 0; b < num_batches; ++b) {
        addmm_impl_cpu_(result, result, batch1.select(0, b), batch2.select(0, b),
                         adjusted_beta, alpha);
        adjusted_beta = 1;
    }
}

// Returns true when an ND x 2D (or 2D x ND) matmul can be folded into a
// single mm call by reshaping the larger tensor to 2D. This avoids the
// overhead of bmm and potentially a large intermediate in the backward pass.
//
// Mirrors PyTorch's should_fold with autograd considerations:
//   - If the small tensor requires_grad, always fold (avoids large
//     intermediate in the backward of expand+bmm).
//   - Otherwise, only fold if the larger tensor's leading dims are
//     contiguous (so reshape is a view, not a copy).
static bool should_fold(const Tensor& tensor1, const Tensor& tensor2) {
    const auto t1_larger = tensor1.ndim() >= tensor2.ndim();

    // Conceptually: t1 is the larger tensor, t2 is the smaller.
    // If tensor2 is larger, we'd transpose before folding.
    const auto dim_t1 = t1_larger ? tensor1.ndim() : tensor2.ndim();
    const auto dim_t2 = t1_larger ? tensor2.ndim() : tensor1.ndim();

    if (!(dim_t1 >= 3 && dim_t2 <= 2)) return false;

    // Autograd: if the small tensor requires_grad, always fold to avoid
    // materializing a large expanded tensor in the backward pass.
    // TODO(autograd): once GradMode is implemented, also check
    //   !GradMode::is_enabled() here.
    bool t2_requires_grad = t1_larger ? tensor2.requires_grad()
                                      : tensor1.requires_grad();
    if (t2_requires_grad) return true;

    // 2D x ND case: folding would require transposing the result and then
    // calling contiguous, which is a copy. Not worth it.
    if (tensor1.ndim() == 2) return false;

    if ((t1_larger ? tensor1 : tensor2).numel() == 0) return true;

    // Check that the first N-1 dims of t1 are contiguous, so reshape is a
    // view (no copy).
    const auto& t1 = t1_larger ? tensor1 : tensor2;
    const auto t1_shape   = t1.shape();
    const auto t1_strides = t1.strides();
    for (int64_t i = 0; i < dim_t1 - 2; ++i) {
        if (t1_strides[i] != t1_strides[i + 1] * t1_shape[i + 1])
            return false;
    }
    return true;
}

// The top-level matmul router. Examines dimensionality and dispatches to the
// most efficient path.
//
//   1D x 1D  ->  dot
//   2D x 1D  ->  mv
//   1D x 2D  ->  mm(t1.unsqueeze(0), t2).squeeze(0)
//   2D x 2D  ->  mm
//   ND x 1D/2D or 2D/1D x ND  ->  fold into mm if possible (should_fold)
//   ND x ND  ->  broadcast batch dims, bmm
static Tensor matmul_impl_(const Tensor& tensor1, const Tensor& tensor2) {
    const auto dim1 = tensor1.ndim();
    const auto dim2 = tensor2.ndim();

    RAXIS_CHECK(dim1 != 0 && dim2 != 0,
        "both arguments to matmul need to be at least 1D, but they are ",
        dim1, "D and ", dim2, "D");

    // -- 1D x 1D -> dot --
    if (dim1 == 1 && dim2 == 1) {
        return dot(tensor1, tensor2);
    }

    // -- 2D x 1D -> mv --
    if (dim1 == 2 && dim2 == 1) {
        return mv(tensor1, tensor2);
    }

    // -- 1D x 2D --
    if (dim1 == 1 && dim2 == 2) {
        return mm(tensor1.unsqueeze(0), tensor2).squeeze(0);
    }

    // -- 2D x 2D -> mm --
    if (dim1 == 2 && dim2 == 2) {
        return mm(tensor1, tensor2);
    }

    // -- Fold path: ND x 2D, ND x 1D, 2D x ND, 1D x ND --
    if (should_fold(tensor1, tensor2)) {
        const auto transpose = dim2 > dim1;

        // t1 is the larger tensor (>= 3D). If tensor2 is larger, we
        // conceptually transpose the problem: (A @ B) = (B^T @ A^T)^T
        // t2 is the smaller tensor (1D or 2D).
        const auto& t1_ref = transpose ? tensor2 : tensor1;
        const auto& t2_ref = transpose ? tensor1 : tensor2;

        // If transposing and t1_ref was originally the RHS, we need its
        // transpose so that the matmul dimensions align.
        Tensor t1 = transpose ? t1_ref.mT() : t1_ref;
        Tensor t2 = !transpose ? t2_ref
                   : (dim1 == 2 ? tensor1.transpose(0, 1) : tensor1);

        const auto sizes_1 = t1.shape();
        int64_t folded_dim1 = 1;
        for (int64_t i = 0; i < t1.ndim() - 1; ++i)
            folded_dim1 *= sizes_1[i];

        DimVector output_shape(sizes_1.begin(), sizes_1.end() - 1);
        const bool t2_is_matrix = t2.ndim() == 2;
        if (t2_is_matrix) output_shape.push_back(t2.shape()[1]);

        auto t1_folded = t1.reshape({folded_dim1, sizes_1[t1.ndim() - 1]});

        if (t2_is_matrix) {
            auto output = mm(t1_folded, t2).reshape(output_shape);
            return transpose ? output.mT().contiguous() : output;
        } else {
            return mm(t1_folded, t2.unsqueeze(1)).squeeze(1).reshape(output_shape);
        }
    }

    // -- General batched path: ND x ND -> broadcast + bmm --
    {
        const int64_t n = dim1 > 1 ? tensor1.shape()[dim1 - 2] : 1;
        const int64_t m1 = tensor1.shape()[dim1 - 1];
        const int64_t m2 = dim2 > 1 ? tensor2.shape()[dim2 - 2] : tensor2.shape()[0];
        const int64_t p = dim2 > 1 ? tensor2.shape()[dim2 - 1] : 1;

        RAXIS_CHECK(m1 == m2,
            "matmul: mat1 and mat2 shapes cannot be multiplied (",
            n, "x", m1, " and ", m2, "x", p, ")");

        IntArrayRef batch1(tensor1.shape().data(), std::max<int64_t>(dim1 - 2, 0));
        IntArrayRef batch2(tensor2.shape().data(), std::max<int64_t>(dim2 - 2, 0));

        // Autograd optimization: if 3D x 3D with mismatched batch and one
        // side has batch=1 + requires_grad, squeeze it so should_fold kicks
        // in on the recursive call, avoiding large intermediates in backward.
        if (dim1 == 3 && dim2 == 3 && batch1[0] != batch2[0]) {
            if (batch1[0] == 1 && tensor1.requires_grad())
                return matmul_impl_(tensor1.squeeze(0), tensor2);
            if (batch2[0] == 1 && tensor2.requires_grad())
                return matmul_impl_(tensor1, tensor2.squeeze(0));
        }

        DimVector output_shape = broadcast_shapes({batch1, batch2});
        int64_t expand_batch_product = 1;
        for (auto s : output_shape) expand_batch_product *= s;

        // Expand and flatten batch dims for tensor1
        DimVector t1_expand(output_shape.begin(), output_shape.end());
        t1_expand.push_back(n);
        t1_expand.push_back(m1);
        auto tensor1_expanded = tensor1.broadcast_to(t1_expand)
                                       .reshape({expand_batch_product, n, m1});

        // Expand and flatten batch dims for tensor2
        bool vector_rhs = (dim2 == 1);
        DimVector t2_expand(output_shape.begin(), output_shape.end());
        if (vector_rhs) {
            t2_expand.push_back(m2);
        } else {
            t2_expand.push_back(m2);
            t2_expand.push_back(p);
        }
        auto tensor2_expanded = tensor2.broadcast_to(t2_expand);
        if (vector_rhs) {
            tensor2_expanded = tensor2_expanded.reshape({expand_batch_product, m2}).unsqueeze(2);
        } else {
            tensor2_expanded = tensor2_expanded.reshape({expand_batch_product, m2, p});
        }

        if (dim1 > 1) output_shape.push_back(n);
        if (dim2 > 1) output_shape.push_back(p);

        auto bmm_result = bmm(tensor1_expanded, tensor2_expanded);
        if (vector_rhs) bmm_result = bmm_result.squeeze(-1);
        return bmm_result.reshape(output_shape);
    }
}

} // namespace


Tensor matmul(const Tensor& tensor1, const Tensor& tensor2) {
    return matmul_impl_(tensor1, tensor2);
}

// The 2-D case with bias is a single addmm: BLAS adds the bias as the C term
// of the gemm (beta = 1) instead of a separate broadcast add.
// From: aten/src/ATen/native/Linear.cpp
Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias) {
    if (input.ndim() == 2 && bias.defined()) {
        return addmm(bias, input, weight.transpose(0, 1));
    }
    Tensor output = matmul(input, weight.transpose(0, 1));
    if (bias.defined()) {
        output = output + bias;
    }
    return output;
}
    
// ---------------------------------------------------------------------------
// mm
// ---------------------------------------------------------------------------

Tensor& mm_out(const Tensor& mat1, const Tensor& mat2, Tensor& out) {
    RAXIS_CHECK(mat1.ndim() == 2, "mm: mat1 must be a matrix, got ", mat1.ndim(), "-D tensor");
    RAXIS_CHECK(mat2.ndim() == 2, "mm: mat2 must be a matrix, got ", mat2.ndim(), "-D tensor");

    if (!out.defined()) {
        out = empty({mat1.shape()[0], mat2.shape()[1]}, mat1.options());
    }
    addmm_impl_cpu_(out, out, mat1, mat2, 0, 1);
    return out;
}

Tensor& addmm_out(
    const Tensor& self, 
    const Tensor& mat1, 
    const Tensor& mat2,
    Scalar beta, 
    Scalar alpha, 
    Tensor& out
) {
    RAXIS_CHECK(mat1.ndim() == 2, "addmm: mat1 must be a matrix, got ", mat1.ndim(), "-D tensor");
    RAXIS_CHECK(mat2.ndim() == 2, "addmm: mat2 must be a matrix, got ", mat2.ndim(), "-D tensor");

    Tensor self_expanded = self.broadcast_to({mat1.shape()[0], mat2.shape()[1]});
    if (!out.defined()) {
        out = empty({mat1.shape()[0], mat2.shape()[1]}, mat1.options());
    }
    addmm_impl_cpu_(out, self_expanded, mat1, mat2, beta, alpha);
    return out;
}

Tensor& bmm_out(const Tensor& self, const Tensor& mat2, Tensor& out) {
    RAXIS_CHECK(self.ndim() == 3, "bmm: expected 3D tensor, got ", self.ndim(), "-D tensor");
    RAXIS_CHECK(mat2.ndim() == 3, "bmm: expected 3D tensor, got ", mat2.ndim(), "-D tensor");
    RAXIS_CHECK(self.shape()[0] == mat2.shape()[0],
        "bmm: batch sizes must match, got ", self.shape()[0], " and ", mat2.shape()[0]);

    if (!out.defined()) {
        out = empty({self.shape()[0], self.shape()[1], mat2.shape()[2]}, self.options());
    }
    bmm_impl_(out, self, mat2, 0, 1, true);
    return out;
}
 
Tensor& addbmm_out(
    const Tensor& self, 
    const Tensor& batch1, 
    const Tensor& batch2,
    Scalar beta, 
    Scalar alpha, 
    Tensor& out
) {
    Tensor self_expanded = self.broadcast_to({batch1.shape()[1], batch2.shape()[2]});
    addbmm_impl_(out, self_expanded, batch1, batch2, beta, alpha);
    return out;
}

namespace ops {

Tensor mm(const Tensor& mat1, const Tensor& mat2) {
    Tensor result;
    return mm_out(mat1, mat2, result);
}

Tensor addmm(
    const Tensor& self, 
    const Tensor& mat1, 
    const Tensor& mat2,
    Scalar beta, 
    Scalar alpha
) {
    Tensor result;
    return addmm_out(self, mat1, mat2, beta, alpha, result);
}


Tensor bmm(const Tensor& self, const Tensor& mat2) {
    Tensor result;
    return bmm_out(self, mat2, result);
}

Tensor addbmm(
    const Tensor& self, 
    const Tensor& batch1, 
    const Tensor& batch2,
    Scalar beta, 
    Scalar alpha
) {
    Tensor result;
    return addbmm_out(self, batch1, batch2, beta, alpha, result);
}
    
} // namespace ops

// ---------------------------------------------------------------------------
// mv
// ---------------------------------------------------------------------------
static constexpr bool lda_cond(int64_t m, int64_t n, int64_t lda) {
    return n == 1 || lda >= std::max<int64_t>(1, m);
}
 
Tensor& addmv_out(
    const Tensor& self, 
    const Tensor& mat, 
    const Tensor& vec,
    Scalar beta, 
    Scalar alpha, 
    Tensor& out
) {
    check_cpu_only("addmv", self, mat, vec, out);
    RAXIS_CHECK(mat.ndim() == 2 && vec.ndim() == 1,
        "matrix + vector expected, got ", mat.ndim(), "D and ", vec.ndim(), "D tensors");
    RAXIS_CHECK(mat.shape()[1] == vec.shape()[0],
        "size mismatch, got mat (", mat.shape()[0], "x", mat.shape()[1],
        "), vec (", vec.shape()[0], ")");
 
    Tensor self_expanded = self.broadcast_to({mat.shape()[0]});
 
    if (!out.defined()) {
        out = empty({mat.shape()[0]}, mat.options());
    }
 
    auto betaval = beta.to<double>();
 
    if (mat.numel() == 0) {
        if (betaval == 0.0)
            out.zero_();
        else
            out.mul_(beta);
        return out;
    }
 
    if (!out.is_same(self_expanded) && betaval != 0.0) {
        out.copy_(self_expanded);
    }
 
    if (out.numel() == 0) return out;
 
    auto r_stride = out.stride(0);
 
    out.dtype().dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        auto b = beta.to<scalar_t>();
        auto a = alpha.to<scalar_t>();
 
        if (mat.stride(0) == 1 && lda_cond(mat.shape()[0], mat.shape()[1], mat.stride(1))) {
            cpublas::gemv<scalar_t>(
                TransposeType::NoTranspose, mat.shape()[0], mat.shape()[1], a,
                mat.data_ptr<scalar_t>(), mat.stride(1),
                vec.data_ptr<scalar_t>(), vec.stride(0),
                b, out.mutable_data_ptr<scalar_t>(), r_stride
            );
        } else if (mat.stride(1) == 1 && lda_cond(mat.shape()[1], mat.shape()[0], mat.stride(0))) {
            cpublas::gemv<scalar_t>(
                TransposeType::Transpose, mat.shape()[1], mat.shape()[0], a,
                mat.data_ptr<scalar_t>(), mat.stride(0),
                vec.data_ptr<scalar_t>(), vec.stride(0),
                b, out.mutable_data_ptr<scalar_t>(), r_stride
            );
        } else {
            Tensor cmat = mat.contiguous();
            cpublas::gemv<scalar_t>(
                TransposeType::Transpose, mat.shape()[1], mat.shape()[0], a,
                cmat.data_ptr<scalar_t>(), cmat.stride(0),
                vec.data_ptr<scalar_t>(), vec.stride(0),
                b, out.mutable_data_ptr<scalar_t>(), r_stride
            );
        }
    });
 
    return out;
}
 
Tensor& mv_out(const Tensor& mat, const Tensor& vec, Tensor& out) {
    if (!out.defined()) {
        out = empty({mat.shape()[0]}, mat.options());
    }
    return addmv_out(out, mat, vec, 0, 1, out);
}

namespace ops {

Tensor addmv(
    const Tensor& self, 
    const Tensor& mat, 
    const Tensor& vec,
    Scalar beta, 
    Scalar alpha
) {
    Tensor result;
    return addmv_out(self, mat, vec, beta, alpha, result);
}

Tensor mv(const Tensor& mat, const Tensor& vec) {
    Tensor result;
    return mv_out(mat, vec, result);
}
 
} // namespace ops

// ---------------------------------------------------------------------------
// dot
// ---------------------------------------------------------------------------
namespace ops {

Tensor dot(const Tensor& self, const Tensor& other) {
    check_cpu_only("dot", self, other);
    RAXIS_CHECK(self.ndim() == 1 && other.ndim() == 1,
        "1D tensors expected, but got ", self.ndim(), "D and ", other.ndim(), "D tensors");
    RAXIS_CHECK(self.dtype() == other.dtype(),
        "dot: expected both vectors to have same dtype, but found ",
        self.dtype(), " and ", other.dtype());
    RAXIS_CHECK(self.numel() == other.numel(),
        "inconsistent tensor size, expected tensor [", self.numel(),
        "] and src [", other.numel(), "] to have the same number of elements");
 
    Tensor result = empty({}, self.options());
 
    self.dtype().dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        result.fill_(cpublas::dot_impl<scalar_t>(
            self.numel(),
            self.data_ptr<scalar_t>(), self.stride(0),
            other.data_ptr<scalar_t>(), other.stride(0)
        ));
    });
 
    return result;
}

}

} // namespace raxis