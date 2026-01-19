
#include <raxis/array_ops.hpp>

#include <iostream>

#include <raxis/array.hpp>
#include <raxis/factories.hpp>
#include <raxis/core/shape.hpp>
#include <raxis/kernels/elementwise.hpp>


namespace raxis {

    NDArray add(const NDArray& a, const NDArray& b) {
        IntArray result_shape = broadcast_shapes(a.shape(), b.shape());
        IntArray sa = broadcast_strides(a.shape(), a.strides(), result_shape);
        IntArray sb = broadcast_strides(b.shape(), b.strides(), result_shape);
        IntArray sout = compute_strides(result_shape);
        
        std::cout << "After Broadcasting\n";
        std::cout << std::format("result_shape: {}\n", result_shape);
        std::cout << std::format("strides_a: {}\n", sa);
        std::cout << std::format("strides_b: {}\n", sb);
        std::cout << std::format("strides_out: {}\n", sout);

        collapse_dim(result_shape, sa, sb, sout);

        std::cout << "After Collapsing\n";
        std::cout << std::format("result_shape: {}\n", result_shape);
        std::cout << std::format("strides_a: {}\n", sa);
        std::cout << std::format("strides_b: {}\n", sb);
        std::cout << std::format("strides_out: {}\n", sout);

        // Execute add
        NDArray out = empty(result_shape, a.dtype());
        
        const void* a_data = a.data_ptr();
        const void* b_data = b.data_ptr();
        void* out_data = out.mutable_data_ptr();

        if (result_shape.size() == 1) {
            kernels::add(a_data, b_data, out_data, out.numel(), out.dtype());
        }

        return out;
    }
    
}