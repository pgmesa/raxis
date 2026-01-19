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
            const Dtype& dtype,
            const IntArray& strides);

        ArrayImpl(
            detail::UniqueVoidPtr data,
            const IntArray& shape,
            const Dtype& dtype);

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
        size_t numel() const noexcept;
        size_t ndim() const noexcept;

    private:
        detail::UniqueVoidPtr data_;
        IntArray shape_;
        Dtype dtype_;
        IntArray strides_;

        size_t numel_;
    };

}