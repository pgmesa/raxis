
#include <raxis/array.hpp>
#include <raxis/array_utils.hpp>

#include <vector>
#include <memory>
#include <format>

#include <iostream>


namespace raxis {

    struct NDArray::Impl {
        void* data;
        IntArray shape;
        IntArray strides;
        Dtype dtype;
        bool owns_data;
    };

    NDArray::NDArray(void* data, const IntArray& shape, const Dtype& dtype)
        : impl_(std::make_shared<Impl>()) 
    {
        impl_->data = data;
        impl_->shape = shape;
        impl_->dtype = dtype;
        impl_->owns_data = false;
        impl_->strides = compute_strides(shape);
    }

    void NDArray::init_from_buffer(
        const void* data, 
        size_t count,
        const IntArray& shape,
        const Dtype& dtype
    ) {
        // auto array = NDArray(const_cast<void*>(data), shape, dtype);

        // impl_->owns_data = true;
        // size_t nbytes = numel() * dtype().itemsize();
        // impl_->data = std::malloc(nbytes);
        // std::memcpy(impl_->data, data.data(), nbytes);
    }

    NDArray::~NDArray() {
        if (impl_ && impl_->owns_data && impl_->data) {
            std::free(impl_->data);
        }
    }

    size_t NDArray::numel() const {
        size_t total = 1;
        for (auto dim : impl_->shape) total *= dim;
        return total;
    }

    size_t NDArray::len() const {
        return impl_->shape.empty() ? 0 : impl_->shape[0];
    }

    IntArray NDArray::shape() const { return impl_->shape; }
    IntArray NDArray::strides() const { return impl_->strides; }
    const Dtype& NDArray::dtype() const { return impl_->dtype; }
    void* NDArray::data() const { return impl_->data; }

    std::string NDArray::to_string() const {
        return std::format(
            "ndarray(shape={}, strides={})", 
            impl_->shape, impl_->strides
        );
    }

    // Operators
    NDArray NDArray::operator+(const NDArray& other) const {
        IntArray result_shape = broadcast_shapes(impl_->shape, other.shape());
        IntArray strides_a = broadcast_strides(impl_->shape, impl_->strides, result_shape);
        IntArray strides_b = broadcast_strides(other.shape(), other.strides(), result_shape);
        IntArray strides_out = compute_strides(result_shape);

        std::cout << std::format("strides_a: {}\n", strides_a);
        std::cout << std::format("strides_b: {}\n", strides_b);

        return zeros(result_shape, other.dtype());
    }


    // Factory implementations
    NDArray zeros(const IntArray& shape, const Dtype& dtype) {
        size_t n = 1;
        for (auto dim : shape) n *= dim;
        
        void* data = std::calloc(n, dtype.itemsize());
        if (!data) throw std::runtime_error("Allocation failed");

        return NDArray(data, shape, dtype);
    }

    NDArray full(const IntArray& shape, double value, const Dtype& dtype) {
        size_t n = 1;
        for (auto dim : shape) n *= dim;
        
        size_t itemsize = dtype.itemsize();
        void* data = std::malloc(n * itemsize);
        if (!data) throw std::runtime_error("Allocation failed");
        
        dtype.fill(data, itemsize, value);

        return NDArray(data, shape, dtype);
    }

    NDArray ones(const IntArray& shape, const Dtype& dtype) {
        return full(shape, 1, dtype);
    }
}