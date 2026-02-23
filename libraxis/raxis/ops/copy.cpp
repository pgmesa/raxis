
#include <raxis/ops/copy.h>

#include <raxis/ops/dispatch_stub.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/factories.h>
#include <raxis/tensor/tensor_iterator.h>


namespace raxis {

Tensor& copy(const Tensor& a, Tensor& out) {
    auto iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .add_output(out)
        .add_input(a)
        .build();
    
    copy_stub(iter.device_type(), iter);
    out = iter.get_output(0);
    return out;
}

Tensor clone(const Tensor& a) {
    Tensor out;
    copy(a, out);
    return out;
}

Tensor to(const Tensor& a, Dtype dtype) {
    if (a.dtype() == dtype) return a;

    Tensor out = empty(a.shape(), dtype);
    copy(a, out);
    return out;
}

DEFINE_DISPATCH(copy_stub);

}