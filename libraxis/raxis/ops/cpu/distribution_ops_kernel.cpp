
#include <raxis/ops/dispatch_stub.h>
#include <raxis/ops/cpu/loops.h>
#include <raxis/ops/distribution_ops.h>
#include <raxis/ops/numeric_utils.h>

#include <raxis/core/generator.h>

#include <raxis/tensor/tensor_iterator.h>


#include <cmath>
#include <random>


namespace raxis {

namespace {

void uniform_kernel(TensorIterator& iter, double low, double high, Generator& gen) {
    iter.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        std::uniform_real_distribution<double> dist(low, high);

        std::lock_guard<std::mutex> lock(gen.mutex());
        auto& engine = gen.engine();

        cpu_serial_kernel(iter, [&]() -> scalar_t {
            return static_cast<scalar_t>(dist(engine));
        });
    });
}

void normal_kernel(TensorIterator& iter, double mean, double std_val, Generator& gen) {
    iter.dtype().dispatch_floating_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        // U(0,1) open interval, exclude 0 to avoid log(0)
        std::uniform_real_distribution<double> uniform(0.0, 1.0);

        std::lock_guard<std::mutex> lock(gen.mutex());
        auto& engine = gen.engine();

        bool has_spare = false;
        double spare = 0.0;

        cpu_serial_kernel(iter, [&]() -> scalar_t {
            if (has_spare) {
                has_spare = false;
                return static_cast<scalar_t>(mean + std_val * spare);
            }

            // Draw two fresh uniforms. Reject u1 == 0.0 to keep log() safe.
            double u1, u2;
            do {
                u1 = uniform(engine);
            } while (u1 == 0.0);
            u2 = uniform(engine);

            constexpr double two_pi = 2.0 * 3.14159265358979323846;
            double mag = std::sqrt(-2.0 * std::log(u1));

            spare = mag * std::sin(two_pi * u2);
            has_spare = true;

            double z0 = mag * std::cos(two_pi * u2);
            return static_cast<scalar_t>(mean + std_val * z0);
        });
    });
}

void randint_kernel(TensorIterator& iter, int64_t low, uint64_t range, Generator& gen) {
    iter.dtype().dispatch_integral_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        std::lock_guard<std::mutex> lock(gen.mutex());
        auto& engine = gen.engine();

        cpu_serial_kernel(iter, [&]() -> scalar_t {
            auto val = low + static_cast<int64_t>(
                uniform_int_from<uint64_t>(engine, range)
            );
            return static_cast<scalar_t>(val);
        });
    }, /*include_bool=*/true);
}

} // namespace

REGISTER_DISPATCH(uniform_stub, &uniform_kernel);
REGISTER_DISPATCH(normal_stub,  &normal_kernel);
REGISTER_DISPATCH(randint_stub, &randint_kernel);

} // namespace raxis