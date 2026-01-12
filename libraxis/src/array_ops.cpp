
#include <raxis/array_ops.hpp>

#include <iostream>

#include <raxis/array.hpp>
#include <raxis/array_utils.hpp>
#include <raxis/kernels/elementwise.hpp>


namespace raxis {

    NDArray add(const NDArray a, const NDArray b) {
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
        Dtype out_dtype = a.dtype();
        size_t itemsize = out_dtype.itemsize();
        size_t out_numel = 1;
        for (Int dim_size: result_shape) {
            out_numel *= dim_size;
        }
        
        void* a_data = a.data();
        void* b_data = b.data();
        void* out_data = std::malloc(out_numel * itemsize);

        if (result_shape.size() == 1) {
            kernels::add(a_data, b_data, out_data, out_numel, out_dtype);
        }

        return NDArray(out_data, result_shape, out_dtype);
    }
    
}