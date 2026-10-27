#pragma once

#include <raxis/tensor/tensor.h>
#include <raxis/util/exception.h>


namespace raxis {

/// For ops whose only implementation reads the data on the host (BLAS calls,
/// loops written in the op itself) instead of going through a DispatchStub.
/// Given a CUDA tensor they would dereference device pointers on the CPU and
/// crash, so they raise NotImplementedError as a stub without a CUDA kernel
/// does. PyTorch's dispatcher raises the same when an op has no kernel for
/// the tensor's device ("Could not run 'aten::mm' with arguments from the
/// 'CUDA' backend"). Undefined (optional) tensors are skipped.
template <typename... Tensors>
void check_cpu_only(const char* op_name, const Tensors&... tensors) {
    auto check = [op_name](const Tensor& t) {
        RAXIS_CHECK_NOT_IMPLEMENTED(!t.defined() || t.is_cpu(),
            op_name, ": no ", t.device(), " implementation yet (raxis only runs it on the CPU)");
    };
    (check(tensors), ...);
}

} // namespace raxis
