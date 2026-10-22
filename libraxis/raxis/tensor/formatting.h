#pragma once

#include <cstdint>
#include <iosfwd>

#include <raxis/export.h>


namespace raxis {

class Tensor;

/// Prints the values of a tensor followed by its type and shape, as libtorch's
/// operator<< does, e.g. for a 2 x 3 float tensor:
///
///    0.1000  0.2000  0.3000
///    1.0000  2.0000  3.0000
///   [ CPUFloatType{2,3} ]
///
/// All values of a matrix share one format (integer, fixed with 4 decimals,
/// fixed with a common scale factor or scientific) and matrices wider than
/// linesize characters are split into column blocks. The Python repr
/// (raxis/_tensor_str.py) follows torch's Python format instead.
/// From: aten/src/ATen/core/Formatting.cpp
RAXIS_API std::ostream& print(std::ostream& stream, const Tensor& tensor, int64_t linesize = 80);

RAXIS_API std::ostream& operator<<(std::ostream& stream, const Tensor& tensor);

} // namespace raxis
