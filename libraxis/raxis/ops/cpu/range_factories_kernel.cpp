#include <raxis/ops/range_factories.h>
#include <raxis/ops/cpu/vec/vec.h>
#include <raxis/ops/dispatch_stub.h>

#include <raxis/core/scalar.h>
#include <raxis/tensor/tensor.h>
#include <raxis/tensor/tensor_iterator.h>
#include <raxis/util/parallel.h>

#include <cstdint>
#include <type_traits>


namespace raxis {
inline namespace CPU_CAPABILITY {

using namespace vec;

namespace {

// From: aten/src/ATen/native/cpu/RangeFactoriesKernel.cpp (arange_kernel)
// Every vector starts from start + step * idx computed in the accumulate
// type (double for floats), so the rounding error does not build up along
// the range as it would by adding step repeatedly
void arange_kernel(const Tensor& result, Scalar scalar_start, Scalar scalar_step) {
    result.dtype().dispatch_all_types_except_bool([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        using acc_t = std::conditional_t<std::is_floating_point_v<scalar_t>, double, int64_t>;
        using Vec = Vectorized<scalar_t>;
        const acc_t start = scalar_start.to<acc_t>();
        const acc_t step = scalar_step.to<acc_t>();
        scalar_t* data = result.mutable_data_ptr<scalar_t>();
        parallel_for(0, result.numel(), detail::GRAIN_SIZE, [&](int64_t begin, int64_t end) {
            int64_t idx = begin;
            // Two vectors per step, as torch's vectorized loop, so the same
            // elements come from a vector base or from the scalar tail and
            // the results match torch bit for bit
            for (; idx + 2 * Vec::size() <= end; idx += 2 * Vec::size()) {
                const int64_t idx2 = idx + Vec::size();
                Vec::arange(static_cast<scalar_t>(start + step * idx), step).store(data + idx);
                Vec::arange(static_cast<scalar_t>(start + step * idx2), step).store(data + idx2);
            }
            for (; idx < end; ++idx) {
                data[idx] = static_cast<scalar_t>(start + step * idx);
            }
        });
    });
}

} // namespace

} // namespace CPU_CAPABILITY

REGISTER_DISPATCH(arange_stub, &arange_kernel);

} // namespace raxis
