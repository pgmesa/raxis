#pragma once

#include <raxis/export.h>
#include <raxis/tensor/tensor.h>


namespace raxis {

// Zeroes each element with probability p and scales the rest by 1 / (1 - p)
// so the expected value is unchanged. Identity when !train or p == 0.
RAXIS_API Tensor dropout(const Tensor& input, double p, bool train);

} // namespace raxis
