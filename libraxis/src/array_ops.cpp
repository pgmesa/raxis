
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

    // TODO: Let ArrayIterator allocate the output tensor, initialize
    // it as undefined. NDArray out;
    IntArray result_shape = broadcast_shapes(a.shape(), b.shape());
    NDArray out = empty(result_shape, a.dtype());

    auto iter = ArrayIteratorConfig()
        .add_output(out)
        .add_input(a)
        .add_input(b)
        .build();

    // iter.print_state();
    kernels::cpu::add_kernel(iter);
    return out;
}
    
}