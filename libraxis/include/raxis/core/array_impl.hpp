#pragma once

#include <raxis/array_types.hpp>
#include <raxis/dtype.hpp>
#include <raxis/core/utils/unique_void_ptr.hpp>
#include <raxis/core/shape.hpp>


namespace raxis {

struct ArrayImpl {
public:
    ArrayImpl(
        detail::UniqueVoidPtr data,
        const IntArray& shape,
        Dtype dtype,
        const IntArray& strides);

    ArrayImpl(
        detail::UniqueVoidPtr data,
        const IntArray& shape,
        Dtype dtype);

    ArrayImpl() = delete;
    ~ArrayImpl() = default;
    ArrayImpl(const ArrayImpl&) = delete;
    ArrayImpl& operator=(const ArrayImpl&) = delete;
    ArrayImpl(ArrayImpl&&) = delete;
    ArrayImpl& operator=(ArrayImpl&&) = delete;
    
    const void* data() const noexcept;
    void* mutable_data() noexcept;

    const IntArray& shape() const noexcept;
    const IntArray& strides() const noexcept;
    Dtype dtype() const noexcept;
    int ndim() const noexcept;
    int64_t numel() const noexcept;

private:
    detail::UniqueVoidPtr data_;
    IntArray shape_;
    Dtype dtype_;
    IntArray strides_;

    int64_t numel_;
};

}