#pragma once

#include <raxis/ops/cpu/vec/vec.h>


namespace raxis::vec {

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

} // namespace raxis::vec