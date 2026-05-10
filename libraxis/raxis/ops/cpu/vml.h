#pragma once

// VML (Vectorized Math Library) interface.
//
// Provides functions like vml::vexp, vml::vlog, etc. that apply a math
// operation to a contiguous buffer of scalars.
//
// Default: uses vec::map with SIMD (Sleef-backed Vectorized<T> methods).
// When MKL is available: template specializations redirect float/double
// to Intel MKL VML routines (vsExp, vdExp, vsLn, vdLn, ...).

#include <raxis/ops/cpu/vec/functional_base.h>
#include <raxis/ops/cpu/vec/vec.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

#if RAXIS_BUILD_WITH_MKL
#include <mkl.h>
#endif


namespace raxis::vml {

inline namespace CPU_CAPABILITY {

using namespace vec;

// ---------------------------------------------------------------------------
// Default implementations via vec::map (SIMD / Sleef)
// ---------------------------------------------------------------------------

#define RAXIS_IMPLEMENT_VML(op)                                             \
    template <typename scalar_t>                                            \
    inline void v##op(scalar_t* out, const scalar_t* in, int64_t size) {    \
        vec::map(                                                           \
            [](Vectorized<scalar_t> x) { return x.op(); },                 \
            out, in, size);                                                 \
    }

RAXIS_IMPLEMENT_VML(abs)
RAXIS_IMPLEMENT_VML(ceil)
RAXIS_IMPLEMENT_VML(exp)
RAXIS_IMPLEMENT_VML(expm1)
RAXIS_IMPLEMENT_VML(floor)
RAXIS_IMPLEMENT_VML(log)
RAXIS_IMPLEMENT_VML(log2)
RAXIS_IMPLEMENT_VML(log10)
RAXIS_IMPLEMENT_VML(log1p)
RAXIS_IMPLEMENT_VML(neg)
RAXIS_IMPLEMENT_VML(reciprocal)
RAXIS_IMPLEMENT_VML(round)
RAXIS_IMPLEMENT_VML(rsqrt)
RAXIS_IMPLEMENT_VML(sqrt)
RAXIS_IMPLEMENT_VML(trunc)
RAXIS_IMPLEMENT_VML(sin)
RAXIS_IMPLEMENT_VML(cos)
RAXIS_IMPLEMENT_VML(tan)
RAXIS_IMPLEMENT_VML(asin)
RAXIS_IMPLEMENT_VML(acos)
RAXIS_IMPLEMENT_VML(atan)
RAXIS_IMPLEMENT_VML(sinh)
RAXIS_IMPLEMENT_VML(cosh)
RAXIS_IMPLEMENT_VML(tanh)
RAXIS_IMPLEMENT_VML(erf)
RAXIS_IMPLEMENT_VML(erfc)

#undef RAXIS_IMPLEMENT_VML

// ---------------------------------------------------------------------------
// MKL specializations (float and double)
// ---------------------------------------------------------------------------

#if RAXIS_BUILD_WITH_MKL

static_assert(
    std::is_same_v<MKL_INT, int32_t> || std::is_same_v<MKL_INT, int64_t>,
    "MKL_INT is assumed to be int32_t or int64_t");

// Generates float and double specializations that call MKL VML functions.
// Uses VML_HA (high accuracy) to match std::exp/std::log precision.
// VML_FTZDAZ_OFF preserves denormals; VML_ERRMODE_IGNORE skips error callbacks.
#define RAXIS_IMPLEMENT_VML_MKL_STUB(op, mklop, type, mkltype)                 \
    template <>                                                                 \
    inline void v##op(type* out, const type* in, int64_t size) {                \
        constexpr auto mkl_max = std::numeric_limits<MKL_INT>::max();           \
        if (size <= static_cast<int64_t>(mkl_max)) {                            \
            vm##mkltype##mklop(                                                 \
                size, in, out, VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE);   \
        } else {                                                                \
            int64_t ind = 0;                                                    \
            int64_t chunks = size / mkl_max;                                    \
            int64_t rest = size % mkl_max;                                      \
            for (; ind < chunks; ind++) {                                       \
                vm##mkltype##mklop(                                             \
                    mkl_max,                                                    \
                    in + ind * mkl_max,                                         \
                    out + ind * mkl_max,                                        \
                    VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE);              \
            }                                                                   \
            vm##mkltype##mklop(                                                 \
                rest,                                                           \
                in + ind * mkl_max,                                             \
                out + ind * mkl_max,                                            \
                VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE);                  \
        }                                                                       \
    }

#define RAXIS_IMPLEMENT_VML_MKL(op, mklop)              \
    RAXIS_IMPLEMENT_VML_MKL_STUB(op, mklop, float, s)   \
    RAXIS_IMPLEMENT_VML_MKL_STUB(op, mklop, double, d)

RAXIS_IMPLEMENT_VML_MKL(exp, Exp)
RAXIS_IMPLEMENT_VML_MKL(log, Ln)
RAXIS_IMPLEMENT_VML_MKL(log10, Log10)
RAXIS_IMPLEMENT_VML_MKL(sqrt, Sqrt)
RAXIS_IMPLEMENT_VML_MKL(trunc, Trunc)
RAXIS_IMPLEMENT_VML_MKL(sin, Sin)
RAXIS_IMPLEMENT_VML_MKL(cos, Cos)
RAXIS_IMPLEMENT_VML_MKL(tan, Tan)
RAXIS_IMPLEMENT_VML_MKL(acos, Acos)
RAXIS_IMPLEMENT_VML_MKL(asin, Asin)
RAXIS_IMPLEMENT_VML_MKL(atan, Atan)
RAXIS_IMPLEMENT_VML_MKL(tanh, Tanh)
RAXIS_IMPLEMENT_VML_MKL(erf, Erf)
RAXIS_IMPLEMENT_VML_MKL(erfc, Erfc)

// PyTorch disables these due to accuracy bugs on certain configs:
// RAXIS_IMPLEMENT_VML_MKL(expm1, Expm1)
// RAXIS_IMPLEMENT_VML_MKL(log1p, Log1p)

#if INTEL_MKL_VERSION >= 20180406
RAXIS_IMPLEMENT_VML_MKL(log2, Log2)
#endif

#undef RAXIS_IMPLEMENT_VML_MKL_STUB
#undef RAXIS_IMPLEMENT_VML_MKL

#endif // RAXIS_BUILD_WITH_MKL

} // namespace CPU_CAPABILITY
} // namespace raxis::vml