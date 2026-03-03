
#include <raxis/ops/dispatch_stub.h>
#include <raxis/ops/cpu/loops.h>
#include <raxis/ops/copy.h>

#include <raxis/core/dtype.h>
#include <raxis/tensor/tensor_iterator.h>


namespace raxis {

namespace {

void direct_copy_kernel(TensorIterator& iter) {
    Dtype dtype = iter.dtype(0);

    dtype.dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>){
        cpu_kernel(iter, [](scalar_t x) -> scalar_t {
            return x;
        });
    });
}

void copy_kernel(TensorIterator& iter) {
    Dtype dtype_out = iter.dtype(0);
    Dtype dtype_inp = iter.dtype(1);
    
    if (dtype_inp == dtype_out) {
        direct_copy_kernel(iter);
    } else {
        dtype_inp.dispatch_all_types([&]<typename src_T>(std::type_identity<src_T>) {
            dtype_out.dispatch_all_types([&]<typename dest_T>(std::type_identity<dest_T>) {
                cpu_kernel(iter, [](src_T x) -> dest_T {
                    return static_cast<dest_T>(x);
                });
            });
        });
    }
}

} // namespace

REGISTER_CPU_DISPATCH(copy_stub, &copy_kernel);
    
} // namespace raxis
