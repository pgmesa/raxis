#pragma once

#include <raxis/ops/cpu/vec/vec.h>


namespace raxis::vec {

// Apply a unary vectorized operation to a contiguous buffer
template <typename scalar_t, typename Op,
    std::enable_if_t<
        std::is_invocable_v<Op, Vectorized<scalar_t>>, int> = 0>
inline void map(
    const Op& vec_fun,
    scalar_t* output_data,
    const scalar_t* input_data,
    int64_t size
) {
    using Vec = Vectorized<scalar_t>;
    int64_t d = 0;
    for (; d < size - (size % Vec::size()); d += Vec::size()) {
        Vec output_vec = vec_fun(Vec::loadu(input_data + d));
        output_vec.store(output_data + d);
    }
    if (size - d > 0) {
        Vec output_vec = vec_fun(Vec::loadu(input_data + d, size - d));
        output_vec.store(output_data + d, size - d);
    }
}
 
// Apply a binary vectorized operation to two contiguous buffers
template <typename scalar_t, typename Op,
    std::enable_if_t<
        std::is_invocable_v<Op, Vectorized<scalar_t>, Vectorized<scalar_t>>, int> = 0>
inline void map2(
    const Op& vec_fun,
    scalar_t* output_data,
    const scalar_t* input_data,
    const scalar_t* input_data2,
    int64_t size
) {
    using Vec = Vectorized<scalar_t>;
    int64_t d = 0;
    for (; d < size - (size % Vec::size()); d += Vec::size()) {
        Vec data_vec = Vec::loadu(input_data + d);
        Vec data_vec2 = Vec::loadu(input_data2 + d);
        Vec output_vec = vec_fun(data_vec, data_vec2);
        output_vec.store(output_data + d);
    }
    if (size - d > 0) {
        Vec data_vec = Vec::loadu(input_data + d, size - d);
        Vec data_vec2 = Vec::loadu(input_data2 + d, size - d);
        Vec output_vec = vec_fun(data_vec, data_vec2);
        output_vec.store(output_data + d, size - d);
    }
}

// Apply a ternary vectorized operation to three contiguous buffers
// From: aten/src/ATen/cpu/vec/functional_base.h
template <typename scalar_t, typename Op,
    std::enable_if_t<
        std::is_invocable_v<Op, Vectorized<scalar_t>, Vectorized<scalar_t>, Vectorized<scalar_t>>, int> = 0>
inline void map3(
    const Op& vec_fun,
    scalar_t* output_data,
    const scalar_t* input_data1,
    const scalar_t* input_data2,
    const scalar_t* input_data3,
    int64_t size
) {
    using Vec = Vectorized<scalar_t>;
    int64_t d = 0;
    for (; d < size - (size % Vec::size()); d += Vec::size()) {
        Vec output_vec = vec_fun(
            Vec::loadu(input_data1 + d), Vec::loadu(input_data2 + d), Vec::loadu(input_data3 + d));
        output_vec.store(output_data + d);
    }
    if (size - d > 0) {
        Vec output_vec = vec_fun(
            Vec::loadu(input_data1 + d, size - d),
            Vec::loadu(input_data2 + d, size - d),
            Vec::loadu(input_data3 + d, size - d));
        output_vec.store(output_data + d, size - d);
    }
}

// Apply a quaternary vectorized operation to four contiguous buffers
template <typename scalar_t, typename Op,
    std::enable_if_t<
        std::is_invocable_v<Op, Vectorized<scalar_t>, Vectorized<scalar_t>,
                            Vectorized<scalar_t>, Vectorized<scalar_t>>, int> = 0>
inline void map4(
    const Op& vec_fun,
    scalar_t* output_data,
    const scalar_t* input_data1,
    const scalar_t* input_data2,
    const scalar_t* input_data3,
    const scalar_t* input_data4,
    int64_t size
) {
    using Vec = Vectorized<scalar_t>;
    int64_t d = 0;
    for (; d < size - (size % Vec::size()); d += Vec::size()) {
        Vec output_vec = vec_fun(
            Vec::loadu(input_data1 + d), Vec::loadu(input_data2 + d),
            Vec::loadu(input_data3 + d), Vec::loadu(input_data4 + d));
        output_vec.store(output_data + d);
    }
    if (size - d > 0) {
        Vec output_vec = vec_fun(
            Vec::loadu(input_data1 + d, size - d), Vec::loadu(input_data2 + d, size - d),
            Vec::loadu(input_data3 + d, size - d), Vec::loadu(input_data4 + d, size - d));
        output_vec.store(output_data + d, size - d);
    }
}

template <typename scalar_t, typename Op>
inline scalar_t vec_reduce_all(
    const Op& vec_fun,
    Vectorized<scalar_t> acc_vec,
    int64_t size
) {
    using Vec = Vectorized<scalar_t>;
    scalar_t acc_arr[Vec::size()];
    acc_vec.store(acc_arr);
    for (int64_t i = 1; i < size; ++i) {
        scalar_t acc_arr_next[Vec::size()] = {0};
        acc_arr_next[0] = acc_arr[i];
        Vec acc_vec_next = Vec::loadu(acc_arr_next);
        acc_vec = vec_fun(acc_vec, acc_vec_next);
    }
    acc_vec.store(acc_arr);
    return acc_arr[0];
}

template <typename scalar_t, typename Op>
inline scalar_t vec_reduce_all(
    const Op& vec_fun,
    const Vectorized<scalar_t>& acc_vec
) {
    return vec_reduce_all(vec_fun, acc_vec, Vectorized<scalar_t>::size());
}

template <typename scalar_t, typename Op>
inline scalar_t reduce_all(
    const Op& vec_fun,
    const scalar_t* data,
    int64_t size
) {
    using Vec = Vectorized<scalar_t>;
    if (size < Vec::size()) {
        return vec_reduce_all(vec_fun, Vec::loadu(data, size), size);
    }
    int64_t d = Vec::size();
    Vec acc_vec = Vec::loadu(data);
    for (; d < size - (size % Vec::size()); d += Vec::size()) {
        acc_vec = vec_fun(acc_vec, Vec::loadu(data + d));
    }
    if (size - d > 0) {
        Vec data_vec = Vec::loadu(data + d, size - d);
        acc_vec = Vec::set(acc_vec, vec_fun(acc_vec, data_vec), size - d);
    }
    return vec_reduce_all(vec_fun, acc_vec);
}

// From: aten/src/ATen/cpu/vec/functional_base.h
// reduce(map(data[i], data2[i])) in one pass, e.g. a dot product
template <typename scalar_t, typename MapOp, typename ReduceOp>
inline scalar_t map2_reduce_all(
    const MapOp& map_fun,
    const ReduceOp& red_fun,
    const scalar_t* data,
    const scalar_t* data2,
    int64_t size
) {
    using Vec = Vectorized<scalar_t>;
    if (size < Vec::size()) {
        Vec data_vec = map_fun(Vec::loadu(data, size), Vec::loadu(data2, size));
        return vec_reduce_all(red_fun, data_vec, size);
    }
    int64_t d = Vec::size();
    Vec acc_vec = map_fun(Vec::loadu(data), Vec::loadu(data2));
    for (; d < size - (size % Vec::size()); d += Vec::size()) {
        acc_vec = red_fun(acc_vec, map_fun(Vec::loadu(data + d), Vec::loadu(data2 + d)));
    }
    if (size - d > 0) {
        Vec data_vec = map_fun(Vec::loadu(data + d, size - d), Vec::loadu(data2 + d, size - d));
        acc_vec = Vec::set(acc_vec, red_fun(acc_vec, data_vec), size - d);
    }
    return vec_reduce_all(red_fun, acc_vec);
}

// From: aten/src/ATen/cpu/vec/functional_base.h
// reduce(map(data[i], data2[i], data3[i])) in one pass
template <typename scalar_t, typename MapOp, typename ReduceOp>
inline scalar_t map3_reduce_all(
    const MapOp& map_fun,
    const ReduceOp& red_fun,
    const scalar_t* data,
    const scalar_t* data2,
    const scalar_t* data3,
    int64_t size
) {
    using Vec = Vectorized<scalar_t>;
    if (size < Vec::size()) {
        Vec data_vec = map_fun(Vec::loadu(data, size), Vec::loadu(data2, size), Vec::loadu(data3, size));
        return vec_reduce_all(red_fun, data_vec, size);
    }
    int64_t d = Vec::size();
    Vec acc_vec = map_fun(Vec::loadu(data), Vec::loadu(data2), Vec::loadu(data3));
    for (; d < size - (size % Vec::size()); d += Vec::size()) {
        acc_vec = red_fun(acc_vec, map_fun(Vec::loadu(data + d), Vec::loadu(data2 + d), Vec::loadu(data3 + d)));
    }
    if (size - d > 0) {
        Vec data_vec = map_fun(
            Vec::loadu(data + d, size - d), Vec::loadu(data2 + d, size - d), Vec::loadu(data3 + d, size - d));
        acc_vec = Vec::set(acc_vec, red_fun(acc_vec, data_vec), size - d);
    }
    return vec_reduce_all(red_fun, acc_vec);
}

} // namespace raxis::vec