
#include <raxis/array_ops.hpp>

#include <iostream>

#include <fmt/format.h>
#include <raxis/array.hpp>
#include <raxis/factories.hpp>
#include <raxis/core/shape.hpp>
#include <raxis/kernels/elementwise.hpp>
#include <raxis/format/array_fmt.hpp>
#include <raxis/core/array_iterator.hpp>


namespace raxis {

    NDArray add(const NDArray& a, const NDArray& b) {
        IntArray result_shape = broadcast_shapes(a.shape(), b.shape());
        IntArray sa = broadcast_strides(a.shape(), a.strides(), result_shape);
        IntArray sb = broadcast_strides(b.shape(), b.strides(), result_shape);
        IntArray sout = compute_strides(result_shape);
        
        std::cout << "After Broadcasting\n";
        std::cout << fmt::format("result_shape: {}\n", result_shape);
        std::cout << fmt::format("strides_a: {}\n", sa);
        std::cout << fmt::format("strides_b: {}\n", sb);
        std::cout << fmt::format("strides_out: {}\n", sout);
        
        IntArray collapsed_shape(result_shape);
        collapse_dim(collapsed_shape, sa, sb, sout);

        std::cout << "After Collapsing\n";
        std::cout << fmt::format("collapsed_shape: {}\n", collapsed_shape);
        std::cout << fmt::format("strides_a: {}\n", sa);
        std::cout << fmt::format("strides_b: {}\n", sb);
        std::cout << fmt::format("strides_out: {}\n", sout);

        // Execute add
        NDArray out = empty(result_shape, a.dtype());
        
        const void* a_data = a.data_ptr();
        const void* b_data = b.data_ptr();
        void* out_data = out.mutable_data_ptr();

        if (collapsed_shape.size() == 1) {
            kernels::add(a_data, b_data, out_data, out.numel(), out.dtype());
        } else {
            auto iter = ArrayIteratorConfig()
                .add_output(out)
                .add_input(a)
                .add_input(b)
                .build();
            
            // cpu_kernel(iter, [](float a, float b) {
            //     return a + b;
            // });
        }

        return out;
    }
    
}