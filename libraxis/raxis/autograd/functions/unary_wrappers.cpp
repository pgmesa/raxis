#include <raxis/ops/unary_ops.h>

#include <raxis/autograd/autograd_macros.h>
#include <raxis/autograd/functions/unary_backward.h>
#include <raxis/autograd/functions/common.h>


namespace raxis {

RAXIS_AUTOGRAD_UNARY_ALL_SAVE_NONE(neg, NegBackward0)
RAXIS_AUTOGRAD_UNARY_ALL_SAVE_NONE(sign, SignBackward0)

RAXIS_AUTOGRAD_UNARY_ALL_SAVE_SELF(abs, AbsBackward0)
RAXIS_AUTOGRAD_UNARY_ALL_SAVE_SELF(log, LogBackward0)
RAXIS_AUTOGRAD_UNARY_ALL_SAVE_SELF(log2, Log2Backward0)
RAXIS_AUTOGRAD_UNARY_ALL_SAVE_SELF(log10, Log10Backward0)
RAXIS_AUTOGRAD_UNARY_ALL_SAVE_SELF(log1p, Log1PBackward0)
RAXIS_AUTOGRAD_UNARY_ALL_SAVE_SELF(sin, SinBackward0)
RAXIS_AUTOGRAD_UNARY_ALL_SAVE_SELF(cos, CosBackward0)
RAXIS_AUTOGRAD_UNARY_ALL_SAVE_SELF(asin, AsinBackward0)
RAXIS_AUTOGRAD_UNARY_ALL_SAVE_SELF(acos, AcosBackward0)
RAXIS_AUTOGRAD_UNARY_ALL_SAVE_SELF(atan, AtanBackward0)
RAXIS_AUTOGRAD_UNARY_ALL_SAVE_SELF(sinh, SinhBackward0)
RAXIS_AUTOGRAD_UNARY_ALL_SAVE_SELF(cosh, CoshBackward0)
RAXIS_AUTOGRAD_UNARY_ALL_SAVE_SELF(erf, ErfBackward0)
RAXIS_AUTOGRAD_UNARY_ALL_SAVE_SELF(erfc, ErfcBackward0)

RAXIS_AUTOGRAD_UNARY_ALL_SAVE_RESULT(exp, ExpBackward0)
RAXIS_AUTOGRAD_UNARY_ALL_SAVE_RESULT(expm1, Expm1Backward0)
RAXIS_AUTOGRAD_UNARY_ALL_SAVE_RESULT(sqrt, SqrtBackward0)
RAXIS_AUTOGRAD_UNARY_ALL_SAVE_RESULT(rsqrt, RsqrtBackward0)
RAXIS_AUTOGRAD_UNARY_ALL_SAVE_RESULT(reciprocal, ReciprocalBackward0)
RAXIS_AUTOGRAD_UNARY_ALL_SAVE_RESULT(sigmoid, SigmoidBackward0)
RAXIS_AUTOGRAD_UNARY_ALL_SAVE_RESULT(tanh, TanhBackward0)
RAXIS_AUTOGRAD_UNARY_ALL_SAVE_RESULT(tan, TanBackward0)

RAXIS_AUTOGRAD_UNARY_ZERO_GRAD(ceil,  "CeilBackward0")
RAXIS_AUTOGRAD_UNARY_ZERO_GRAD(floor, "FloorBackward0")
RAXIS_AUTOGRAD_UNARY_ZERO_GRAD(round, "RoundBackward0")
RAXIS_AUTOGRAD_UNARY_ZERO_GRAD(trunc, "TruncBackward0")

// x - trunc(x): the gradient passes as it is
RAXIS_AUTOGRAD_UNARY_ALL_SAVE_NONE(frac, FracBackward0)

} // namespace raxis