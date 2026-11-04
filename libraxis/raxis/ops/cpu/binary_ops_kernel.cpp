
#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <type_traits>

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

// --- Integer division, remainder and fmod ---
// From: aten/src/ATen/native/cpu/BinaryOpsKernel.cpp, c10/util/generic_math.h
// There is no SIMD integer division, so the integer cases are scalar loops.
// An integer division by zero raises (torch's ZeroDivisionError); for
// floats it gives what IEEE says (inf or nan). INT_MIN / -1 overflows in
// C++, so those cases are written out.

template<typename scalar_t>
bool signs_differ(scalar_t a, scalar_t b) {
    if constexpr (std::is_unsigned_v<scalar_t>) {
        return false;
    } else {
        return (a < 0) != (b < 0);
    }
}

// Python's a // b for floats (NOTE: [Floor Division in Python] in torch):
// (a - fmod(a, b)) / b keeps a == (a // b) * b + a % b exact, where
// floor(a / b) can be off by one after the rounding of a / b
template<typename scalar_t>
scalar_t div_floor_floating(scalar_t a, scalar_t b) {
    if (b == 0) [[unlikely]] {
        return a / b;
    }
    const scalar_t mod = std::fmod(a, b);
    scalar_t div = (a - mod) / b;
    if (mod != 0 && (b < 0) != (mod < 0)) {
        div -= scalar_t(1);
    }
    if (div == 0) {
        return std::copysign(scalar_t(0), a / b);
    }
    scalar_t floordiv = std::floor(div);
    if (div - floordiv > scalar_t(0.5)) {
        floordiv += scalar_t(1);
    }
    return floordiv;
}

// The same steps on lanes: every branch becomes a blend
template<typename scalar_t>
Vectorized<scalar_t> div_floor_floating_vec(const Vectorized<scalar_t>& a, const Vectorized<scalar_t>& b) {
    using vec_t = Vectorized<scalar_t>;
    const vec_t zero(0);
    const vec_t one(1);
    const vec_t inf(std::numeric_limits<scalar_t>::infinity());
    // -0.0 is only the sign bit: x & -0.0 is copysign(0, x)
    const vec_t sign_bit(scalar_t(-0.0));
    const vec_t basic_div = a / b;
    const vec_t mod = a.fmod(b);
    // A finite a over a tiny b overflows a / b; Sleef's fmod is then not
    // exact enough, and a itself is the better numerator (as torch)
    const vec_t numerator = vec_t::blendv(a - mod, a, (basic_div.abs() == inf) & (a.abs() != inf));
    vec_t div = numerator / b;
    div = vec_t::blendv(div, div - one, (mod != zero) & ((b < zero) ^ (mod < zero)));
    vec_t floordiv = div.floor();
    floordiv = vec_t::blendv(floordiv, floordiv + one, (div - floordiv) > vec_t(scalar_t(0.5)));
    floordiv = vec_t::blendv(floordiv, basic_div & sign_bit, div == zero);
    return vec_t::blendv(floordiv, basic_div, b == zero);
}

void div_trunc_kernel(TensorIterator& iter) {
    const Dtype dtype = iter.common_dtype();
    if (dtype.is_integral()) {
        dtype.dispatch_integral_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
            cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
                RAXIS_CHECK(b != 0, "ZeroDivisionError");
                if constexpr (std::is_signed_v<scalar_t>) {
                    if (a == std::numeric_limits<scalar_t>::min() && b == scalar_t(-1)) {
                        return a;
                    }
                }
                return a / b;
            });
        });
        return;
    }
    dtype.dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel_vec(iter,
            [](scalar_t a, scalar_t b) -> scalar_t { return std::trunc(a / b); },
            [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return (a / b).trunc(); }
        );
    });
}

void div_floor_kernel(TensorIterator& iter) {
    const Dtype dtype = iter.common_dtype();
    if (dtype == dtypes::Uint8) {
        // Both signs are positive: floor and truncation are the same
        div_trunc_kernel(iter);
        return;
    }
    if (dtype.is_integral()) {
        dtype.dispatch_integral_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
            cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
                RAXIS_CHECK(b != 0, "ZeroDivisionError");
                if constexpr (std::is_signed_v<scalar_t>) {
                    if (a == std::numeric_limits<scalar_t>::min() && b == scalar_t(-1)) {
                        return a;
                    }
                }
                // Truncation rounds toward zero; with a remainder of the
                // other sign than b, floor is one less. One idiv gives both
                // quot and rem, and the correction is a subtraction of 0 or
                // 1 instead of a branch: with mixed signs a branch is
                // mispredicted half of the time, which cost more than the
                // division (torch branches: 3.2 ms per million int64, 1.3
                // for the truncation alone)
                const scalar_t quot = a / b;
                const scalar_t rem = a % b;
                return quot - static_cast<scalar_t>((rem != 0) & signs_differ(rem, b));
            });
        });
        return;
    }
    dtype.dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel_vec(iter,
            [](scalar_t a, scalar_t b) -> scalar_t { return div_floor_floating(a, b); },
            [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return div_floor_floating_vec(a, b); }
        );
    });
}

// Python's %: the result takes the sign of the divisor. C's % and fmod take
// the dividend's, so a nonzero result of the other sign gets b added
void remainder_kernel(TensorIterator& iter) {
    const Dtype dtype = iter.common_dtype();
    if (dtype.is_integral()) {
        dtype.dispatch_integral_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
            cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
                RAXIS_CHECK(b != 0, "ZeroDivisionError");
                if constexpr (std::is_signed_v<scalar_t>) {
                    if (a == std::numeric_limits<scalar_t>::min() && b == scalar_t(-1)) {
                        return 0;
                    }
                }
                // Branchless, as in div_floor_kernel: b is added through a
                // mask of all ones or all zeros
                const scalar_t r = a % b;
                const auto mask = static_cast<scalar_t>(-static_cast<scalar_t>((r != 0) & signs_differ(r, b)));
                return static_cast<scalar_t>(r + (b & mask));
            });
        });
        return;
    }
    dtype.dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel_vec(iter,
            [](scalar_t a, scalar_t b) -> scalar_t {
                scalar_t mod = std::fmod(a, b);
                if (mod != 0 && (b < 0) != (mod < 0)) {
                    mod += b;
                }
                return mod;
            },
            [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) {
                const Vectorized<scalar_t> zero(0);
                const auto mod = a.fmod(b);
                const auto mask = (mod != zero) & ((b < zero) ^ (mod < zero));
                return Vectorized<scalar_t>::blendv(mod, mod + b, mask);
            }
        );
    });
}

void fmod_kernel(TensorIterator& iter) {
    const Dtype dtype = iter.common_dtype();
    if (dtype.is_integral()) {
        dtype.dispatch_integral_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
            cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t {
                RAXIS_CHECK(b != 0, "ZeroDivisionError");
                if constexpr (std::is_signed_v<scalar_t>) {
                    if (a == std::numeric_limits<scalar_t>::min() && b == scalar_t(-1)) {
                        return 0;
                    }
                }
                return a % b;
            });
        });
        return;
    }
    dtype.dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel_vec(iter,
            [](scalar_t a, scalar_t b) -> scalar_t { return std::fmod(a, b); },
            [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return a.fmod(b); }
        );
    });
}

// --- Bitwise ---
// From: aten/src/ATen/native/cpu/BinaryOpsKernel.cpp
// bool has no ^ in C++ (it promotes to int), so the bool cases are written
// as logical ops

void bitwise_and_kernel(TensorIterator& iter) {
    if (iter.common_dtype().is_bool()) {
        cpu_kernel(iter, [](bool a, bool b) -> bool { return a && b; });
        return;
    }
    iter.common_dtype().dispatch_integral_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel_vec(iter,
            [](scalar_t a, scalar_t b) -> scalar_t { return a & b; },
            [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return a & b; }
        );
    });
}

void bitwise_or_kernel(TensorIterator& iter) {
    if (iter.common_dtype().is_bool()) {
        cpu_kernel(iter, [](bool a, bool b) -> bool { return a || b; });
        return;
    }
    iter.common_dtype().dispatch_integral_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel_vec(iter,
            [](scalar_t a, scalar_t b) -> scalar_t { return a | b; },
            [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return a | b; }
        );
    });
}

void bitwise_xor_kernel(TensorIterator& iter) {
    if (iter.common_dtype().is_bool()) {
        cpu_kernel(iter, [](bool a, bool b) -> bool { return a != b; });
        return;
    }
    iter.common_dtype().dispatch_integral_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        cpu_kernel_vec(iter,
            [](scalar_t a, scalar_t b) -> scalar_t { return a ^ b; },
            [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return a ^ b; }
        );
    });
}

// In C++ a shift by a negative amount or by the bit width or more is
// undefined; torch defines it: << gives 0 and >> fills with the sign bit
template<typename scalar_t>
scalar_t lshift(scalar_t a, scalar_t b) {
    constexpr scalar_t max_shift = sizeof(scalar_t) * CHAR_BIT;
    if (static_cast<std::make_signed_t<scalar_t>>(b) < 0 || b >= max_shift) {
        return 0;
    }
    // Shifting the unsigned bits: a negative a << b is undefined before
    // C++20, and the overflow wraps as torch's does
    return static_cast<scalar_t>(static_cast<std::make_unsigned_t<scalar_t>>(a) << b);
}

template<typename scalar_t>
scalar_t rshift(scalar_t a, scalar_t b) {
    // A signed value keeps its sign bit: shifts of width - 1 or more give
    // 0 or -1; an unsigned one ends at 0
    constexpr scalar_t max_shift = sizeof(scalar_t) * CHAR_BIT - std::is_signed_v<scalar_t>;
    if (static_cast<std::make_signed_t<scalar_t>>(b) < 0 || b >= max_shift) {
        if constexpr (std::is_signed_v<scalar_t>) {
            return a < 0 ? scalar_t(-1) : scalar_t(0);
        } else {
            return 0;
        }
    }
    return static_cast<scalar_t>(a >> b);
}

// AVX2 has per-lane shifts of int32 and int64 (vpsllv / vpsrav, see
// vec256_int.h); uint8 and the other ISAs take the scalar loop. torch keeps
// these kernels on AVX2 too
template<typename scalar_t>
constexpr bool has_vector_shifts =
#if defined(CPU_CAPABILITY_AVX2)
    std::is_same_v<scalar_t, int32_t> || std::is_same_v<scalar_t, int64_t>;
#else
    false;
#endif

void lshift_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_integral_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        if constexpr (has_vector_shifts<scalar_t>) {
            cpu_kernel_vec(iter,
                [](scalar_t a, scalar_t b) -> scalar_t { return lshift(a, b); },
                [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return a << b; }
            );
        } else {
            cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t { return lshift(a, b); });
        }
    });
}

void rshift_kernel(TensorIterator& iter) {
    iter.common_dtype().dispatch_integral_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        if constexpr (has_vector_shifts<scalar_t>) {
            cpu_kernel_vec(iter,
                [](scalar_t a, scalar_t b) -> scalar_t { return rshift(a, b); },
                [](Vectorized<scalar_t> a, Vectorized<scalar_t> b) { return a >> b; }
            );
        } else {
            cpu_kernel(iter, [](scalar_t a, scalar_t b) -> scalar_t { return rshift(a, b); });
        }
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
REGISTER_DISPATCH(div_trunc_stub, &div_trunc_kernel);
REGISTER_DISPATCH(div_floor_stub, &div_floor_kernel);
REGISTER_DISPATCH(remainder_stub, &remainder_kernel);
REGISTER_DISPATCH(fmod_stub, &fmod_kernel);
REGISTER_DISPATCH(bitwise_and_stub, &bitwise_and_kernel);
REGISTER_DISPATCH(bitwise_or_stub, &bitwise_or_kernel);
REGISTER_DISPATCH(bitwise_xor_stub, &bitwise_xor_kernel);
REGISTER_DISPATCH(lshift_stub, &lshift_kernel);
REGISTER_DISPATCH(rshift_stub, &rshift_kernel);

} // namesapce raxis