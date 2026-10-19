
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

namespace ops {

Tensor clone(const Tensor& self, MemoryFormat memory_format) {
    Tensor out = empty_like(self, {}, memory_format);
    copy(self, out);
    return out;
}

Tensor to_copy(const Tensor& self, Dtype dtype) {
    Tensor out = empty_like(self, dtype);
    copy(self, out);
    return out;
}

} // namespace ops

DEFINE_DISPATCH(copy_stub);

}