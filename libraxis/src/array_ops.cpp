
#include <raxis/array_ops.hpp>

#include <iostream>

#include <fmt/format.h>
#include <raxis/array.hpp>
#include <raxis/factories.hpp>
#include <raxis/core/shape.hpp>
#include <raxis/format/array_fmt.hpp>
#include <raxis/core/array_iterator.hpp>
#include <raxis/kernels/cpu/binary_ops.hpp>


namespace raxis {

    NDArray add(const NDArray& a, const NDArray& b) {
        IntArray result_shape = broadcast_shapes(a.shape(), b.shape());

        // Execute add
        NDArray out = empty(result_shape, a.dtype());

        auto iter = ArrayIteratorConfig()
            .add_output(out)
            .add_input(a)
            .add_input(b)
            .build();

        iter.print_state();
        //kernels::cpu::add_kernel(iter);
        
        // const void* a_data = a.data_ptr();
        // const void* b_data = b.data_ptr();
        // void* out_data = out.mutable_data_ptr();

        // if (collapsed_shape.size() == 1) {
        //     kernels::add(a_data, b_data, out_data, out.numel(), out.dtype());
        // } else {
            
            
        //     // cpu_kernel(iter, [](float a, float b) {
        //     //     return a + b;
        //     // });
        // }

        return out;
    }
    
}