
#include <raxis/ops/linear_algebra.h>

#include <raxis/tensor/tensor.h>

#include <raxis/autograd/variable.h>
#include <raxis/autograd/functions/linear_algebra_backward.h>


namespace raxis {

Tensor mm(const Tensor& mat1, const Tensor& mat2) {
    Tensor result = ops::mm(mat1, mat2);
    if (autograd::requires_autograd(mat1, mat2)) {
        auto grad_fn = std::make_shared<autograd::MmBackward0>(mat1, mat2);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, mat1, mat2);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor mv(const Tensor& mat, const Tensor& vec) {
    Tensor result = ops::mv(mat, vec);
    if (autograd::requires_autograd(mat, vec)) {
        auto grad_fn = std::make_shared<autograd::MvBackward0>(mat, vec);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, mat, vec);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor dot(const Tensor& self, const Tensor& other) {
    Tensor result = ops::dot(self, other);
    if (autograd::requires_autograd(self, other)) {
        auto grad_fn = std::make_shared<autograd::DotBackward0>(self, other);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self, other);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor bmm(const Tensor& self, const Tensor& mat2) {
    Tensor result = ops::bmm(self, mat2);
    if (autograd::requires_autograd(self, mat2)) {
        auto grad_fn = std::make_shared<autograd::BmmBackward0>(self, mat2);
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self, mat2);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor addmm(const Tensor& self, const Tensor& mat1, const Tensor& mat2,
             Scalar beta, Scalar alpha) {
    Tensor result = ops::addmm(self, mat1, mat2, beta, alpha);
    if (autograd::requires_autograd(self) ||
        autograd::requires_autograd(mat1) ||
        autograd::requires_autograd(mat2)) {
        auto grad_fn = std::make_shared<autograd::AddmmBackward0>(
            self, mat1, mat2, beta, alpha
        );
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_next_edges(grad_fn, mat1);
        autograd::set_next_edges(grad_fn, mat2);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor addmv(const Tensor& self, const Tensor& mat, const Tensor& vec,
             Scalar beta, Scalar alpha) {
    Tensor result = ops::addmv(self, mat, vec, beta, alpha);
    if (autograd::requires_autograd(self) ||
        autograd::requires_autograd(mat) ||
        autograd::requires_autograd(vec)) {
        auto grad_fn = std::make_shared<autograd::AddmvBackward0>(
            self, mat, vec, beta, alpha
        );
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_next_edges(grad_fn, mat);
        autograd::set_next_edges(grad_fn, vec);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

Tensor addbmm(const Tensor& self, const Tensor& batch1, const Tensor& batch2,
              Scalar beta, Scalar alpha) {
    Tensor result = ops::addbmm(self, batch1, batch2, beta, alpha);
    if (autograd::requires_autograd(self) ||
        autograd::requires_autograd(batch1) ||
        autograd::requires_autograd(batch2)) {
        auto grad_fn = std::make_shared<autograd::AddbmmBackward0>(
            self, batch1, batch2, beta, alpha
        );
        grad_fn->set_num_inputs(1);
        autograd::set_next_edges(grad_fn, self);
        autograd::set_next_edges(grad_fn, batch1);
        autograd::set_next_edges(grad_fn, batch2);
        autograd::set_autograd_meta(result, grad_fn);
    }
    return result;
}

} // namespace raxis