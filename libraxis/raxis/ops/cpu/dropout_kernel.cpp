#include <raxis/ops/dropout.h>
#include <raxis/ops/dispatch_stub.h>
#include <raxis/ops/cpu/philox_batch.h>

#include <raxis/core/dtype.h>
#include <raxis/core/philox.h>
#include <raxis/tensor/tensor.h>
#include <raxis/util/parallel.h>

#include <cmath>
#include <cstdint>
#include <type_traits>


namespace raxis {

inline namespace CPU_CAPABILITY {

// Elements per parallel task
constexpr int64_t kDropoutGrain = 8192;

// From: aten/src/ATen/native/Dropout.cpp (native_dropout_cpu) for the math,
// and PyTorch's CUDA dropout for the generation (Philox counters reserved
// from the generator, so any thread can draw any batch).
//
// Element i is kept when its 32-bit draw r is below p_keep * 2^32, i.e.
// u = r / 2^32 < p_keep: an exact integer comparison with no conversion to
// floating point. One pass draws, writes the mask and
// output = input * keep * scale, the same operation order as PyTorch's
// input.mul(mask).mul_(scale) (so NaN inputs stay NaN when dropped).
template <typename scalar_t>
void fused_dropout_impl(
    scalar_t* out, bool* mask, const scalar_t* in, int64_t n,
    double p_keep, uint64_t seed, uint64_t offset
) {
    const auto threshold = static_cast<uint64_t>(std::ldexp(p_keep, 32));
    const auto scale = static_cast<scalar_t>(p_keep == 0 ? 0.0 : 1.0 / p_keep);

    philox_for_each_batch(n, kPhiloxBatchWords, seed, offset, kDropoutGrain,
        [&](const uint32_t* r, int64_t base, int64_t count) {
            if (count == kPhiloxBatchWords) {
                // Constant trip counts and one simple statement per loop, so
                // the compiler vectorizes both
                bool* m = mask + base;
                for (int64_t j = 0; j < kPhiloxBatchWords; ++j) {
                    m[j] = r[j] < threshold;
                }
                for (int64_t j = 0; j < kPhiloxBatchWords; ++j) {
                    out[base + j] = in[base + j] * static_cast<scalar_t>(m[j]) * scale;
                }
                return;
            }
            for (int64_t j = 0; j < count; ++j) {
                const bool keep = r[j] < threshold;
                mask[base + j] = keep;
                out[base + j] = in[base + j] * static_cast<scalar_t>(keep) * scale;
            }
        });
}

// grad_input = grad * mask * scale, in PyTorch's order
// From: aten/src/ATen/native/Dropout.cpp (native_dropout_backward)
template <typename scalar_t>
void dropout_backward_impl(scalar_t* grad_in, const scalar_t* grad, const bool* mask, int64_t n, double scale_) {
    const auto scale = static_cast<scalar_t>(scale_);
    parallel_for(0, n, kDropoutGrain, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            grad_in[i] = grad[i] * static_cast<scalar_t>(mask[i]) * scale;
        }
    });
}

void fused_dropout_kernel(
    const Tensor& output, const Tensor& mask, const Tensor& input,
    double p_keep, uint64_t seed, uint64_t offset
) {
    input.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        fused_dropout_impl<scalar_t>(
            output.mutable_data_ptr<scalar_t>(), mask.mutable_data_ptr<bool>(),
            input.data_ptr<scalar_t>(), input.numel(), p_keep, seed, offset);
    });
}

void dropout_backward_kernel(const Tensor& grad_input, const Tensor& grad, const Tensor& mask, double scale) {
    grad.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        dropout_backward_impl<scalar_t>(
            grad_input.mutable_data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(),
            mask.data_ptr<bool>(), grad.numel(), scale);
    });
}

} // namespace CPU_CAPABILITY

// The Philox batch fits one AVX512 register per word (1.3x); the backward
// is a masked multiply, memory-bound
ALSO_REGISTER_AVX512_DISPATCH(fused_dropout_stub, &fused_dropout_kernel);
REGISTER_DISPATCH(dropout_backward_stub, &dropout_backward_kernel);

} // namespace raxis
