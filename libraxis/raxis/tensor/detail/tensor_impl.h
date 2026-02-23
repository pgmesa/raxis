#pragma once

#include <raxis/core/types.h>
#include <raxis/core/dtype.h>
#include <raxis/core/device.h>

#include <raxis/util/data_ptr.h>

#include <raxis/tensor/shape_utils.h>


namespace raxis::detail {

struct TensorImpl {
public:
    TensorImpl(
        DataPtr data,
        const IntArray& shape,
        Dtype dtype);

    TensorImpl(
        DataPtr data,
        const IntArray& shape,
        Dtype dtype,
        const IntArray& strides);

    TensorImpl() = delete;
    ~TensorImpl() = default;
    TensorImpl(const TensorImpl&) = delete;
    TensorImpl& operator=(const TensorImpl&) = delete;
    TensorImpl(TensorImpl&&) = delete;
    TensorImpl& operator=(TensorImpl&&) = delete;
    
    const void* data() const;
    void* mutable_data();

    void set_wrapped_number(bool);

    const IntArray& shape() const;
    const IntArray& strides() const;
    Dtype dtype() const;
    Device device() const;
    int ndim() const;
    int64_t numel() const;

private:
    DataPtr data_ptr_;
    IntArray shape_;
    Dtype dtype_;
    IntArray strides_;

    int64_t numel_;
    bool is_wrapped_number_ = false;
};

}