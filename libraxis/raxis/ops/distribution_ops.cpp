
#include <raxis/ops/distribution_ops.h>

#include <raxis/core/dtype.h>
#include <raxis/core/generator.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/tensor_iterator.h>
#include <raxis/tensor/factories.h>

#include <raxis/util/exception.h>


namespace raxis {

Tensor& uniform_(Tensor& self, double low, double high) {
    RAXIS_CHECK(
        self.dtype().is_floating(),
        "uniform_ expects a floating-point dtype, got ", self.dtype()
    );
    RAXIS_CHECK(
        low <= high,
        "uniform_ expects low (", low, ") <= high (", high, ")"
    );
    if (self.numel() == 0) return self;

    auto iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .add_output(self)
        .build();

    uniform_stub(iter.device_type(), iter, low, high, get_default_generator());
    return self;
}

Tensor& normal_(Tensor& self, double mean, double std) {
    RAXIS_CHECK(
        self.dtype().is_floating(),
        "normal_ expects a floating-point dtype, got ", self.dtype()
    );
    RAXIS_CHECK(std >= 0.0, "normal_ expects std >= 0, got ", std);
    if (self.numel() == 0) return self;

    auto iter = TensorIteratorConfig()
        .check_all_same_dtype(false)
        .add_output(self)
        .build();

    normal_stub(iter.device_type(), iter, mean, std, get_default_generator());
    return self;
}

DEFINE_DISPATCH(uniform_stub);
DEFINE_DISPATCH(normal_stub);

} // namespace raxis