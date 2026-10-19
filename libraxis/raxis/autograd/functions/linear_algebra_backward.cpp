
#include <raxis/autograd/functions/linear_algebra_backward.h>

#include <raxis/autograd/variable.h>


namespace raxis::autograd {

MmBackward0::MmBackward0(const Tensor& mat1, const Tensor& mat2)
    : mat1_requires_grad_(mat1.requires_grad()),
      mat2_requires_grad_(mat2.requires_grad())
{
    if (mat1_requires_grad_) mat2_.save(mat2);
    if (mat2_requires_grad_) mat1_.save(mat1);
}

std::string MmBackward0::name() const { return "MmBackward0"; }

TensorList MmBackward0::apply(const TensorList& grads) {
    const Tensor& grad = grads[0];
    Tensor grad_mat1, grad_mat2;

    if (mat1_requires_grad_)
        grad_mat1 = grad.mm(mat2_.unpack().transpose(0, 1));
    if (mat2_requires_grad_)
        grad_mat2 = mat1_.unpack().transpose(0, 1).mm(grad);

    return {grad_mat1, grad_mat2};
}


MvBackward0::MvBackward0(const Tensor& mat, const Tensor& vec)
    : mat_requires_grad_(mat.requires_grad()),
      vec_requires_grad_(vec.requires_grad())
{
    if (mat_requires_grad_) vec_.save(vec);
    if (vec_requires_grad_) mat_.save(mat);
}

std::string MvBackward0::name() const { return "MvBackward0"; }

TensorList MvBackward0::apply(const TensorList& grads) {
    const Tensor& grad = grads[0];
    Tensor grad_mat, grad_vec;

    if (mat_requires_grad_)
        grad_mat = grad.unsqueeze(1).mm(vec_.unpack().unsqueeze(0));
    if (vec_requires_grad_)
        grad_vec = mat_.unpack().transpose(0, 1).mv(grad);

    return {grad_mat, grad_vec};
}


DotBackward0::DotBackward0(const Tensor& self, const Tensor& other)
    : self_requires_grad_(self.requires_grad()),
      other_requires_grad_(other.requires_grad())
{
    if (self_requires_grad_) other_.save(other);
    if (other_requires_grad_) self_.save(self);
}

std::string DotBackward0::name() const { return "DotBackward0"; }

TensorList DotBackward0::apply(const TensorList& grads) {
    const Tensor& grad = grads[0];
    Tensor grad_self, grad_other;

    if (self_requires_grad_)
        grad_self = grad * other_.unpack();
    if (other_requires_grad_)
        grad_other = grad * self_.unpack();

    return {grad_self, grad_other};
}


BmmBackward0::BmmBackward0(const Tensor& self, const Tensor& mat2)
    : self_requires_grad_(self.requires_grad()),
      mat2_requires_grad_(mat2.requires_grad())
{
    if (self_requires_grad_) mat2_.save(mat2);
    if (mat2_requires_grad_) self_.save(self);
}

std::string BmmBackward0::name() const { return "BmmBackward0"; }

TensorList BmmBackward0::apply(const TensorList& grads) {
    const Tensor& grad = grads[0];
    Tensor grad_self, grad_mat2;

    if (self_requires_grad_)
        grad_self = grad.bmm(mat2_.unpack().transpose(1, 2));
    if (mat2_requires_grad_)
        grad_mat2 = self_.unpack().transpose(1, 2).bmm(grad);

    return {grad_self, grad_mat2};
}


AddmmBackward0::AddmmBackward0(
    const Tensor& self, const Tensor& mat1, const Tensor& mat2,
    Scalar beta, Scalar alpha)
    : self_requires_grad_(self.requires_grad()),
      mat1_requires_grad_(mat1.requires_grad()),
      mat2_requires_grad_(mat2.requires_grad()),
      beta_(beta), alpha_(alpha)
{
    if (self_requires_grad_) self_shape_ = DimVector(self.shape());
    if (mat1_requires_grad_) mat2_.save(mat2);
    if (mat2_requires_grad_) mat1_.save(mat1);
}

std::string AddmmBackward0::name() const { return "AddmmBackward0"; }

TensorList AddmmBackward0::apply(const TensorList& grads) {
    const Tensor& grad = grads[0];
    Tensor grad_self, grad_mat1, grad_mat2;

    if (self_requires_grad_)
        grad_self = unbroadcast(grad * beta_, self_shape_);
    if (mat1_requires_grad_)
        grad_mat1 = grad.mm(mat2_.unpack().transpose(0, 1)) * alpha_;
    if (mat2_requires_grad_)
        grad_mat2 = mat1_.unpack().transpose(0, 1).mm(grad) * alpha_;

    return {grad_self, grad_mat1, grad_mat2};
}


AddmvBackward0::AddmvBackward0(
    const Tensor& self, const Tensor& mat, const Tensor& vec,
    Scalar beta, Scalar alpha)
    : self_requires_grad_(self.requires_grad()),
      mat_requires_grad_(mat.requires_grad()),
      vec_requires_grad_(vec.requires_grad()),
      beta_(beta), alpha_(alpha)
{
    if (self_requires_grad_) self_shape_ = DimVector(self.shape());
    if (mat_requires_grad_) vec_.save(vec);
    if (vec_requires_grad_) mat_.save(mat);
}

std::string AddmvBackward0::name() const { return "AddmvBackward0"; }

TensorList AddmvBackward0::apply(const TensorList& grads) {
    const Tensor& grad = grads[0];
    Tensor grad_self, grad_mat, grad_vec;

    if (self_requires_grad_)
        grad_self = grad * beta_;
    if (mat_requires_grad_)
        grad_mat = grad.unsqueeze(1).mm(vec_.unpack().unsqueeze(0)) * alpha_;
    if (vec_requires_grad_)
        grad_vec = mat_.unpack().transpose(0, 1).mv(grad) * alpha_;

    return {grad_self, grad_mat, grad_vec};
}


AddbmmBackward0::AddbmmBackward0(
    const Tensor& self, const Tensor& batch1, const Tensor& batch2,
    Scalar beta, Scalar alpha)
    : self_requires_grad_(self.requires_grad()),
      batch1_requires_grad_(batch1.requires_grad()),
      batch2_requires_grad_(batch2.requires_grad()),
      beta_(beta), alpha_(alpha)
{
    if (self_requires_grad_) self_shape_ = DimVector(self.shape());
    if (batch1_requires_grad_) batch2_.save(batch2);
    if (batch2_requires_grad_) batch1_.save(batch1);
    batch_size_ = batch1.shape()[0];
}

std::string AddbmmBackward0::name() const { return "AddbmmBackward0"; }

TensorList AddbmmBackward0::apply(const TensorList& grads) {
    const Tensor& grad = grads[0];
    Tensor grad_self, grad_batch1, grad_batch2;

    if (self_requires_grad_)
        grad_self = unbroadcast(grad * beta_, self_shape_);

    if (batch1_requires_grad_) {
        // grad is [M, N], batch2 is [B, K, N]
        // d/d(batch1_i) = alpha * grad @ batch2_i^T for each i
        // -> broadcast grad to [B, M, N], then bmm with batch2^T
        Tensor grad_expanded = grad.unsqueeze(0).broadcast_to(
            {batch_size_, grad.shape()[0], grad.shape()[1]});
        grad_batch1 = grad_expanded.bmm(batch2_.unpack().transpose(1, 2)) * alpha_;
    }

    if (batch2_requires_grad_) {
        Tensor grad_expanded = grad.unsqueeze(0).broadcast_to(
            {batch_size_, grad.shape()[0], grad.shape()[1]});
        grad_batch2 = batch1_.unpack().transpose(1, 2).bmm(grad_expanded) * alpha_;
    }

    return {grad_self, grad_batch1, grad_batch2};
}

} // namespace raxis::autograd