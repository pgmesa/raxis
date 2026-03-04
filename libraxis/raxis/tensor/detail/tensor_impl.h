#pragma once

#include <optional>

#include <raxis/core/types.h>
#include <raxis/core/dtype.h>
#include <raxis/core/device.h>

#include <raxis/util/data_ptr.h>

#include <raxis/tensor/detail/storage_impl.h>


namespace raxis::detail {

struct TensorImpl {
public:
    TensorImpl(
        DataPtr data,
        size_t nbytes,
        const IntArray& shape,
        std::optional<IntArray> strides,
        Dtype dtype);

    TensorImpl(
        std::shared_ptr<StorageImpl> storage,
        int64_t storage_offset,
        const IntArray& shape,
        const IntArray& strides,
        Dtype dtype);

    TensorImpl() = delete;
    ~TensorImpl() = default;
    TensorImpl(const TensorImpl&) = delete;
    TensorImpl& operator=(const TensorImpl&) = delete;
    TensorImpl(TensorImpl&&) = delete;
    TensorImpl& operator=(TensorImpl&&) = delete;
    
    const void* data() const;
    void* mutable_data();

    void set_wrapped_number(bool);
    bool is_wrapped_number() const;

    const IntArray& shape() const;
    const IntArray& strides() const;
    Dtype dtype() const;
    Device device() const;
    int ndim() const;
    int64_t numel() const;
    int64_t storage_offset() const;

    const std::shared_ptr<StorageImpl>& storage() const;

private:
    std::shared_ptr<StorageImpl> storage_;
    int64_t storage_offset_ = 0;

    IntArray shape_;
    IntArray strides_;
    Dtype dtype_;

    int64_t numel_;
    bool is_wrapped_number_ = false;
};

}