
#include <raxis/ops/scalar_ops.h>

#include <raxis/core/scalar.h>
#include <raxis/core/dtype.h>
#include <raxis/tensor/tensor.h>



namespace raxis {

namespace {

template <typename scalar_t>
inline void fill_inplace(Tensor& self, Scalar value_scalar) {
    scalar_t value = value_scalar.to<scalar_t>();  
    scalar_t* dptr = self.mutable_data_ptr<scalar_t>();
    *dptr = value;
}
    
} // namespace

namespace detail {

Tensor& scalar_fill(Tensor& self, Scalar value) {
    self.dtype().dispatch_all_types([&]<typename scalar_t>(std::type_identity<scalar_t>) {
        fill_inplace<scalar_t>(self, value);
    });
    return self;
}

Tensor scalar_tensor_static(
    Scalar s,
    std::optional<Dtype> dtype_opt,
    std::optional<Device> device_opt
) {
    auto opts = TensorOptions()
        .dtype(dtype_opt.value_or(s.dtype()))
        .device(device_opt.value_or(DeviceType::CPU));
    Tensor result = empty({}, opts);
    scalar_fill(result, s);
    return result;
}
    
} // namespace detail

} // namespace raxis
