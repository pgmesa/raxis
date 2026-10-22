
#include <raxis/ops/tensor_compare.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/ops/cpu/loops.h>
#include <raxis/ops/cpu/vec/vec.h>

#include <raxis/core/dtype.h>
#include <raxis/core/scalar.h>
#include <raxis/ops/numeric_utils.h>
#include <raxis/tensor/tensor.h>
#include <raxis/tensor/tensor_iterator.h>

#include <algorithm>
#include <limits>


namespace raxis {

inline namespace CPU_CAPABILITY {

// From: aten/src/ATen/native/cpu/TensorCompareKernel.cpp
void where_kernel(TensorIterator& iter) {
    iter.dtype(0).dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel(iter, [](bool cond, scalar_t self, scalar_t other) -> scalar_t {
            return cond ? self : other;
        });
    });
}

// From: aten/src/ATen/native/cpu/TensorCompareKernel.cpp
// min(max(a, lo), hi): with lo > hi every element becomes hi, as in torch.
// A NaN element stays NaN without extra work: std::max/min return their first
// argument when the comparison is false, and vec::clamp puts `a` second in
// maxps/minps, which return the second operand when either one is NaN
void clamp_scalar_kernel(TensorIterator& iter, Scalar min_, Scalar max_) {
    iter.common_dtype().dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        const auto min = min_.to<scalar_t>();
        const auto max = max_.to<scalar_t>();
        const Vectorized<scalar_t> min_vec(min);
        const Vectorized<scalar_t> max_vec(max);
        cpu_kernel_vec(iter,
            [=](scalar_t a) -> scalar_t {
                return std::min(std::max(a, min), max);
            },
            [=](Vectorized<scalar_t> a) {
                return vec::clamp(a, min_vec, max_vec);
            }
        );
    });
}

void clamp_min_scalar_kernel(TensorIterator& iter, Scalar min_) {
    iter.common_dtype().dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        const auto min = min_.to<scalar_t>();
        const Vectorized<scalar_t> min_vec(min);
        cpu_kernel_vec(iter,
            [=](scalar_t a) -> scalar_t {
                return std::max(a, min);
            },
            [=](Vectorized<scalar_t> a) {
                return vec::clamp_min(a, min_vec);
            }
        );
    });
}

void clamp_max_scalar_kernel(TensorIterator& iter, Scalar max_) {
    iter.common_dtype().dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        const auto max = max_.to<scalar_t>();
        const Vectorized<scalar_t> max_vec(max);
        cpu_kernel_vec(iter,
            [=](scalar_t a) -> scalar_t {
                return std::min(a, max);
            },
            [=](Vectorized<scalar_t> a) {
                return vec::clamp_max(a, max_vec);
            }
        );
    });
}

// From: aten/src/ATen/native/cpu/TensorCompareKernel.cpp (clamp_kernel_impl)
// min(max(a, lo), hi) propagates a NaN in a; a NaN bound gives NaN too
void clamp_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel_vec(iter,
            [](scalar_t a, scalar_t min, scalar_t max) -> scalar_t {
                if (_isnan(min) || _isnan(max)) {
                    return std::numeric_limits<scalar_t>::quiet_NaN();
                }
                return std::min(std::max(a, min), max);
            },
            [](Vectorized<scalar_t> a, Vectorized<scalar_t> min, Vectorized<scalar_t> max) {
                return vec::minimum(vec::maximum(a, min), max);
            }
        );
    });
}

// From: aten/src/ATen/native/cpu/TensorCompareKernel.cpp (compare_base_kernel)
// Not a reduction: an elementwise iteration over the output positions (self
// with `dim` squashed to 1), where `f` walks the whole dim of self for each
// one. Output positions are independent, so they run in parallel (grain 1:
// every position does a full scan along dim).
template <typename scalar_t, typename func_t>
void compare_base_kernel(
    const Tensor& values, const Tensor& indices, const Tensor& self, int64_t dim, const func_t& f
) {
    int64_t self_dim_stride = self.strides()[dim];
    int64_t self_dim_size = self.shape()[dim];

    auto iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .declare_static_shape(self.shape(), {dim})
        .add_output(values)
        .add_output(indices)
        .add_input(self)
        .build();

    auto loop = [&](char** data, const int64_t* strides, int64_t n) {
        char* values_bytes = data[0];
        char* indices_bytes = data[1];
        const char* self_bytes = data[2];
        for (int64_t i = 0; i < n; i++) {
            f(reinterpret_cast<scalar_t*>(values_bytes), reinterpret_cast<int64_t*>(indices_bytes),
              reinterpret_cast<const scalar_t*>(self_bytes), self_dim_stride, self_dim_size);
            values_bytes += strides[0];
            indices_bytes += strides[1];
            self_bytes += strides[2];
        }
    };
    iter.for_each(loop, /*grain_size=*/1);
}

// !(value <= best) is true when value > best and also when value is NaN, so
// NaN wins (and stops the scan: nothing beats it). Ties keep the first index.
void max_dim_kernel(const Tensor& values, const Tensor& indices, const Tensor& self, int64_t dim) {
    self.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        compare_base_kernel<scalar_t>(values, indices, self, dim,
            [](scalar_t* value_out, int64_t* index_out, const scalar_t* data, int64_t stride, int64_t size) {
                scalar_t best = data[0];
                int64_t index = 0;
                for (int64_t i = 0; i < size; i++) {
                    scalar_t value = data[i * stride];
                    if (!(value <= best)) {
                        best = value;
                        index = i;
                        if (_isnan(value)) break;
                    }
                }
                *value_out = best;
                *index_out = index;
            });
    });
}

void min_dim_kernel(const Tensor& values, const Tensor& indices, const Tensor& self, int64_t dim) {
    self.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        compare_base_kernel<scalar_t>(values, indices, self, dim,
            [](scalar_t* value_out, int64_t* index_out, const scalar_t* data, int64_t stride, int64_t size) {
                scalar_t best = data[0];
                int64_t index = 0;
                for (int64_t i = 0; i < size; i++) {
                    scalar_t value = data[i * stride];
                    if (!(value >= best)) {
                        best = value;
                        index = i;
                        if (_isnan(value)) break;
                    }
                }
                *value_out = best;
                *index_out = index;
            });
    });
}

} // namespace CPU_CAPABILITY

REGISTER_DISPATCH(max_dim_stub, &max_dim_kernel);
REGISTER_DISPATCH(min_dim_stub, &min_dim_kernel);
REGISTER_DISPATCH(where_stub, &where_kernel);
REGISTER_DISPATCH(clamp_scalar_stub, &clamp_scalar_kernel);
REGISTER_DISPATCH(clamp_min_scalar_stub, &clamp_min_scalar_kernel);
REGISTER_DISPATCH(clamp_max_scalar_stub, &clamp_max_scalar_kernel);
REGISTER_DISPATCH(clamp_stub, &clamp_kernel);

} // namespace raxis
