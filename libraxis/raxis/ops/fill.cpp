
#include <raxis/ops/fill.h>

#include <cstring>

#include <raxis/ops/scalar_ops.h>

#include <raxis/core/scalar.h>
#include <raxis/core/dtype.h>
#include <raxis/core/device.h>

#include <raxis/tensor/tensor.h>
#include <raxis/tensor/tensor_iterator.h>


namespace raxis {

Tensor& fill_(Tensor& self, Scalar value) {
    // Shortcut
    if (self.device() == DeviceType::CPU && self.numel() == 1) {
        return detail::scalar_fill(self, value);
    }

    auto iter = TensorIteratorConfig()
        .check_mem_overlap(false)  // Fill is idempotent, so overlap is okay
        .check_all_same_dtype(false)
        .add_output(self)
        .build();

    fill_stub(iter.device_type(), iter, value);
    return self;
}

static Tensor& zero_cpu_(Tensor &self, int64_t nelements) {
    void* ptr = self.mutable_data_ptr();
    if (nullptr == ptr) {
        return self.fill_(0);
    }
    auto size_bytes = nelements * self.dtype().itemsize();
    if (size_bytes > 0) {
        std::memset(ptr, 0, size_bytes);
    }
    return self;
}

Tensor& zero_(Tensor& self) {
    int64_t numel = self.numel();
    // memset is only valid when the elements fill one block of memory
    if (self.device() == DeviceType::CPU && self.is_non_overlapping_and_dense() &&
        numel < detail::GRAIN_SIZE) {
        return zero_cpu_(self, numel);
    }

    return self.fill_(0);
}

DEFINE_DISPATCH(fill_stub);

} // namespace raxis
