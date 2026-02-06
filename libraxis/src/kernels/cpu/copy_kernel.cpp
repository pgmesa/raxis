
#include <raxis/kernels/cpu/copy_kernel.hpp>

#include <raxis/dtype.hpp>
#include <raxis/core/array_iterator.hpp>
#include <raxis/kernels/cpu/loops.hpp>


namespace raxis::kernels::cpu {

void direct_copy_kernel(ArrayIterator& iter) {
    Dtype dtype = iter.dtype(0);

    dtype.dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>){
        cpu_kernel(iter, [](scalar_t x) -> scalar_t {
            return x;
        });
    });
}

void copy_kernel(ArrayIterator& iter) {
    Dtype dtype_inp = iter.dtype(0);
    Dtype dtype_out = iter.dtype(1);
    
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
    
} // namespace raxis::kernels::cpu
