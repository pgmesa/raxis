
#include <algorithm>
#include <cmath>
#include <limits>

#include <raxis/ops/cpu/vec/vec.h>
#include <raxis/ops/cpu/loops.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/ops/binary_ops.h>

#include <raxis/core/dtype.h>
#include <raxis/tensor/tensor_iterator.h>


namespace raxis {

namespace {

using namespace vec;

void add_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel_vec(iter,
            [](scalar_t a, scalar_t b) -> scalar_t {
                return a + b;
            },
            [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
                return a + b;
            }
        );
    });
}

void sub_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel_vec(iter,
            [](scalar_t a, scalar_t b) -> scalar_t {
                return a - b;
            },
            [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
                return a - b;
            }
        );
    });
}

void mul_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        if constexpr (std::is_same_v<scalar_t, bool>) {
            cpu_kernel(iter, [](bool a, bool b) -> bool {
                return a && b;
            });
        } else {
            cpu_kernel_vec(iter,
                [](scalar_t a, scalar_t b) -> scalar_t {
                    return a * b;
                },
                [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
                    return a * b;
                }
            );
        }
    });
}

void div_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel_vec(iter,
            [](scalar_t a, scalar_t b) -> scalar_t {
                return a / b;
            },
            [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
                return a / b;
            }
        );
    });
}

void gt_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel(iter, [](scalar_t a, scalar_t b) -> bool { return a > b; });
    });
}
 
void ge_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel(iter, [](scalar_t a, scalar_t b) -> bool { return a >= b; });
    });
}
 
void lt_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel(iter, [](scalar_t a, scalar_t b) -> bool { return a < b; });
    });
}
 
void le_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel(iter, [](scalar_t a, scalar_t b) -> bool { return a <= b; });
    });
}
 
void eq_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel(iter, [](scalar_t a, scalar_t b) -> bool { return a == b; });
    });
}
 
void ne_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel(iter, [](scalar_t a, scalar_t b) -> bool { return a != b; });
    });
}

void maximum_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        if constexpr (std::is_same_v<scalar_t, bool>) {
            cpu_kernel(iter, [](bool a, bool b) -> bool { return a || b; });
        } else if constexpr (std::is_floating_point_v<scalar_t>) {
            cpu_kernel_vec(iter,
                [](scalar_t a, scalar_t b) -> scalar_t {
                    if (a != a || b != b) {
                        return std::numeric_limits<scalar_t>::quiet_NaN();
                    }
                    return std::max(a, b);
                },
                [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
                    return vec::maximum(a, b);
                }
            );
        } else {
            cpu_kernel_vec(iter,
                [](scalar_t a, scalar_t b) -> scalar_t {
                    return std::max(a, b);
                },
                [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
                    return vec::maximum(a, b);
                }
            );
        }
    });
}

void minimum_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        if constexpr (std::is_same_v<scalar_t, bool>) {
            cpu_kernel(iter, [](bool a, bool b) -> bool { return a && b; });
        } else if constexpr (std::is_floating_point_v<scalar_t>) {
            cpu_kernel_vec(iter,
                [](scalar_t a, scalar_t b) -> scalar_t {
                    if (a != a || b != b) {
                        return std::numeric_limits<scalar_t>::quiet_NaN();
                    }
                    return std::min(a, b);
                },
                [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
                    return vec::minimum(a, b);
                }
            );
        } else {
            cpu_kernel_vec(iter,
                [](scalar_t a, scalar_t b) -> scalar_t {
                    return std::min(a, b);
                },
                [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
                    return vec::minimum(a, b);
                }
            );
        }
    });
}
 
void logical_and_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel(iter, [](scalar_t a, scalar_t b) -> bool {
            return static_cast<bool>(a) && static_cast<bool>(b);
        });
    });
}
 
void logical_or_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel(iter, [](scalar_t a, scalar_t b) -> bool {
            return static_cast<bool>(a) || static_cast<bool>(b);
        });
    });
}

} // namespace

REGISTER_DISPATCH(add_stub, &add_kernel);
REGISTER_DISPATCH(sub_stub, &sub_kernel);
REGISTER_DISPATCH(mul_stub, &mul_kernel);
REGISTER_DISPATCH(div_stub, &div_kernel);
REGISTER_DISPATCH(gt_stub, &gt_kernel);
REGISTER_DISPATCH(ge_stub, &ge_kernel);
REGISTER_DISPATCH(lt_stub, &lt_kernel);
REGISTER_DISPATCH(le_stub, &le_kernel);
REGISTER_DISPATCH(eq_stub, &eq_kernel);
REGISTER_DISPATCH(ne_stub, &ne_kernel);
REGISTER_DISPATCH(maximum_stub, &maximum_kernel);
REGISTER_DISPATCH(minimum_stub, &minimum_kernel);
REGISTER_DISPATCH(logical_and_stub, &logical_and_kernel);
REGISTER_DISPATCH(logical_or_stub, &logical_or_kernel);

} // namesapce raxis