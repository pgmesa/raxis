#include <raxis/ops/foreach_ops.h>

#include <cmath>
#include <limits>
#include <optional>

#include <raxis/autograd/grad_mode.h>

#include <raxis/util/exception.h>

#include <raxis/tensor/factories.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

namespace {

// From: aten/src/ATen/native/ForeachUtils.h (check_fast_path_restrictions)
bool can_use_fast_path(ArrayRef<Tensor> tensors) {
    if (tensors.empty() || tensors[0].device().type() != DeviceType::CUDA) {
        return false;
    }
    const Tensor& first = tensors[0];
    for (const Tensor& t : tensors) {
        if (t.device() != first.device() || t.dtype() != first.dtype() || !t.is_contiguous()) {
            return false;
        }
    }
    return first.dtype().is_floating();
}

std::optional<ForeachNormType> fused_norm_type(double ord) {
    if (ord == 0) return ForeachNormType::L0;
    if (ord == 1) return ForeachNormType::L1;
    if (ord == 2) return ForeachNormType::L2;
    if (ord == std::numeric_limits<double>::infinity()) return ForeachNormType::LInf;
    return std::nullopt;
}

// torch.linalg.vector_norm of t flattened, composed of elementwise ops and a reduction
Tensor vector_norm(const Tensor& t, double ord) {
    if (ord == std::numeric_limits<double>::infinity()) {
        return t.abs().amax();
    }
    if (ord == -std::numeric_limits<double>::infinity()) {
        return t.abs().amin();
    }
    if (ord == 0) {
        return t.ne(0).sum().to(t.dtype());
    }
    if (ord == 1) {
        return t.abs().sum();
    }
    if (ord == 2) {
        return t.mul(t).sum().sqrt();
    }
    return t.abs().pow(ord).sum().pow(1.0 / ord);
}

} // anonymous namespace

std::vector<Tensor> foreach_norm(ArrayRef<Tensor> tensors, double ord) {
    for (const Tensor& t : tensors) {
        RAXIS_CHECK(t.defined() && t.dtype().is_floating(),
            "_foreach_norm: expected floating point tensors, got ", t.dtype());
        RAXIS_CHECK(std::isfinite(ord) || t.numel() > 0,
            "_foreach_norm cannot compute the infinity norm on an empty tensor because the operation does not "
            "have an identity");
    }
    autograd::GradModeGuard no_grad(false);
    const std::optional<ForeachNormType> norm = fused_norm_type(ord);
    std::vector<Tensor> result;
    result.reserve(tensors.size());
    if (!norm || !can_use_fast_path(tensors)) {
        for (const Tensor& t : tensors) {
            result.push_back(vector_norm(t, ord));
        }
        return result;
    }
    const Tensor out = empty({static_cast<int64_t>(tensors.size())}, tensors[0].options());
    foreach_norm_stub(DeviceType::CUDA, tensors, *norm, out);
    for (size_t i = 0; i < tensors.size(); i++) {
        result.push_back(out.select(0, static_cast<int64_t>(i)));
    }
    return result;
}

void foreach_mul_(ArrayRef<Tensor> tensors, const Tensor& scalar) {
    RAXIS_CHECK(scalar.numel() == 1, "_foreach_mul_: expected a one-element scalar tensor, got ", scalar.numel(),
        " elements");
    autograd::GradModeGuard no_grad(false);
    const bool fused = can_use_fast_path(tensors) && scalar.device() == tensors[0].device()
        && scalar.dtype() == tensors[0].dtype();
    if (!fused) {
        for (const Tensor& t : tensors) {
            t.mul_(scalar);
        }
        return;
    }
    foreach_mul_stub(DeviceType::CUDA, tensors, scalar.reshape({}));
    // In place without autograd (as fused_adam_): copies saved of them are stale now
    for (const Tensor& t : tensors) {
        t.unsafe_get_tensor_impl()->storage()->bump_version();
    }
}

DEFINE_DISPATCH(foreach_norm_stub);
DEFINE_DISPATCH(foreach_mul_stub);
REGISTER_NO_CPU_DISPATCH(foreach_norm_stub)
REGISTER_NO_CPU_DISPATCH(foreach_mul_stub)

} // namespace raxis
