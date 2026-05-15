
#include <raxis/ops/cat_ops.h>
#include <raxis/ops/cpu/vec/vec.h>
#include <raxis/ops/dispatch_stub.h>

#include <raxis/core/dtype.h>
#include <raxis/tensor/tensor.h>
#include <raxis/util/small_vector.h>


namespace raxis {

namespace {

using namespace vec;

struct InputMeta {
    const void* data_ptr;
    int64_t inner_size;
};

template <typename scalar_t>
void cat_serial_kernel_impl(
    const Tensor& result,
    ArrayRef<Tensor> tensors,
    int64_t dim
) {
    int64_t inner = result.strides()[dim];
    int64_t outer = result.numel() / (result.shape()[dim] * inner);

    int64_t ninputs = static_cast<int64_t>(tensors.size());
    SmallVector<InputMeta, 8> inputs;
    inputs.reserve(ninputs);
    for (const Tensor& t : tensors) {
        if (t.ndim() == 1 && t.shape()[0] == 0) {
            continue;
        }
        inputs.push_back({t.data_ptr(), t.shape()[dim] * inner});
    }
    ninputs = static_cast<int64_t>(inputs.size());

    using Vec = Vectorized<scalar_t>;
    scalar_t* result_ptr = result.mutable_data_ptr<scalar_t>();

    for (int64_t i = 0; i < outer; ++i) {
        for (int64_t j = 0; j < ninputs; ++j) {
            int64_t local_inner = inputs[j].inner_size;
            const scalar_t* input_ptr =
                static_cast<const scalar_t*>(inputs[j].data_ptr) + i * local_inner;
            int64_t d = 0;
            for (; d < local_inner - (local_inner % Vec::size()); d += Vec::size()) {
                Vec in_vec = Vec::loadu(input_ptr + d);
                in_vec.store(result_ptr + d);
            }
            for (; d < local_inner; ++d) {
                result_ptr[d] = input_ptr[d];
            }
            result_ptr += local_inner;
        }
    }
}

void cat_serial_kernel(const Tensor& result, ArrayRef<Tensor> tensors, int64_t dim) {
    result.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cat_serial_kernel_impl<scalar_t>(result, tensors, dim);
    });
}

template <typename scalar_t>
void stack_serial_kernel_impl(
    const Tensor& result,
    ArrayRef<Tensor> tensors,
    int64_t dim
) {
    int64_t outer = result.numel() / (result.shape()[dim] * result.strides()[dim]);

    int64_t ninputs = static_cast<int64_t>(tensors.size());
    SmallVector<InputMeta, 8> inputs;
    inputs.reserve(ninputs);
    for (const Tensor& t : tensors) {
        int64_t input_inner = (dim < t.ndim()) ? t.shape()[dim] * t.strides()[dim]
                                               : 1;
        inputs.push_back({t.data_ptr(), input_inner});
    }

    using Vec = Vectorized<scalar_t>;
    scalar_t* result_ptr = result.mutable_data_ptr<scalar_t>();

    for (int64_t i = 0; i < outer; ++i) {
        for (int64_t j = 0; j < ninputs; ++j) {
            int64_t local_inner = inputs[j].inner_size;
            const scalar_t* input_ptr =
                static_cast<const scalar_t*>(inputs[j].data_ptr) + i * local_inner;
            int64_t d = 0;
            for (; d < local_inner - (local_inner % Vec::size()); d += Vec::size()) {
                Vec in_vec = Vec::loadu(input_ptr + d);
                in_vec.store(result_ptr + d);
            }
            for (; d < local_inner; ++d) {
                result_ptr[d] = input_ptr[d];
            }
            result_ptr += local_inner;
        }
    }
}

void stack_serial_kernel(const Tensor& result, ArrayRef<Tensor> tensors, int64_t dim) {
    result.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        stack_serial_kernel_impl<scalar_t>(result, tensors, dim);
    });
}

} // namespace

REGISTER_DISPATCH(cat_serial_stub, &cat_serial_kernel);
REGISTER_DISPATCH(stack_serial_stub, &stack_serial_kernel);

} // namespace raxis