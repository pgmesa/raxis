#pragma once

#include <cstdint>

#include <raxis/export.h>

#include <raxis/core/dtype.h>

#include <raxis/ops/dispatch_stub.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/tensor_options.h>


namespace raxis {

// The kernels fill a contiguous tensor in parallel from a Philox stream (see
// core/philox.h): element i takes its words from the blocks
// [offset, offset + philox_blocks(numel, words_per_element)) of the stream
// keyed by seed, so the values depend on the seed and on the generator's
// offset, never on the number of threads or the CPU capability.
//
// PyTorch's CPU kernels draw serially from mt19937 under the generator lock
// (aten/src/ATen/native/cpu/DistributionTemplates.h), so raxis gives other
// values than torch for the same seed; the distributions are the same.
using uniform_fn = void(*)(const Tensor& self, double low, double high, uint64_t seed, uint64_t offset);
using normal_fn  = void(*)(const Tensor& self, double mean, double std, uint64_t seed, uint64_t offset);
using randint_fn = void(*)(const Tensor& self, int64_t low, uint64_t range, uint64_t seed, uint64_t offset);

DECLARE_DISPATCH(uniform_fn, uniform_stub);
DECLARE_DISPATCH(normal_fn,  normal_stub);
DECLARE_DISPATCH(randint_fn, randint_stub);

// 32-bit words per uniform or normal sample: float takes 24 random bits from
// one word and double 53 bits from two, as PyTorch's uniform_real_distribution
inline int real_words_per_element(Dtype dtype) {
    return dtype == dtypes::Float64 ? 2 : 1;
}

// randint draws 64 bits per element
inline constexpr int kRandintWordsPerElement = 2;

/// Fills tensor with values drawn from U[low, high).
RAXIS_API Tensor& uniform_(Tensor& self, double low = 0.0, double high = 1.0);

/// Fills tensor with values drawn from N(mean, std^2).
RAXIS_API Tensor& normal_(Tensor& self, double mean = 0.0, double std = 1.0);

/// Fills tensor with integers drawn from U[low, high).
RAXIS_API Tensor& randint_(Tensor& self, int64_t low, int64_t high);

/// num_samples category indices drawn from each row of self (1-D or 2-D),
/// with probabilities proportional to its non-negative weights. Without
/// replacement a category is drawn at most once per row.
RAXIS_API Tensor multinomial(const Tensor& self, int64_t num_samples, bool replacement = false);

namespace ops {

RAXIS_API Tensor& uniform_(Tensor& self, double low = 0.0, double high = 1.0);
RAXIS_API Tensor& normal_(Tensor& self, double mean = 0.0, double std = 1.0);
RAXIS_API Tensor& randint_(Tensor& self, int64_t low, int64_t high);
// From: aten/src/ATen/native/Distributions.cpp (multinomial_out)
RAXIS_API Tensor multinomial(const Tensor& self, int64_t num_samples, bool replacement = false);

} // namespace ops

} // namespace raxis
