#pragma once

#include <raxis/core/types.h>
#include <raxis/core/dtype.h>

#include <raxis/util/unique_void_ptr.h>

#include <raxis/tensor/shape_utils.h>


namespace raxis {

struct TensorImpl {
public:
    TensorImpl(
        detail::UniqueVoidPtr data,
        const IntArray& shape,
        Dtype dtype);

    TensorImpl(
        detail::UniqueVoidPtr data,
        const IntArray& shape,
        Dtype dtype,
        const IntArray& strides);

    TensorImpl() = delete;
    ~TensorImpl() = default;
    TensorImpl(const TensorImpl&) = delete;
    TensorImpl& operator=(const TensorImpl&) = delete;
    TensorImpl(TensorImpl&&) = delete;
    TensorImpl& operator=(TensorImpl&&) = delete;
    
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