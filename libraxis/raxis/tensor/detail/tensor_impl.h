#pragma once

#include <memory>
#include <optional>

#include <raxis/core/types.h>
#include <raxis/core/dtype.h>
#include <raxis/core/device.h>

#include <raxis/util/data_ptr.h>

#include <raxis/tensor/detail/storage_impl.h>


namespace raxis {

namespace autograd {
struct AutogradMeta;
} // namespace autograd


namespace detail {

struct TensorImpl {
public:
    TensorImpl(
        DataPtr data,
        size_t nbytes,
        IntArrayRef shape,
        std::optional<IntArrayRef> strides,
        Dtype dtype);

    TensorImpl(
        std::shared_ptr<StorageImpl> storage,
        int64_t storage_offset,
        IntArrayRef shape,
        IntArrayRef strides,
        Dtype dtype);

    TensorImpl() = delete;
    ~TensorImpl();
    TensorImpl(const TensorImpl&) = delete;
    TensorImpl& operator=(const TensorImpl&) = delete;
    TensorImpl(TensorImpl&&) = delete;
    TensorImpl& operator=(TensorImpl&&) = delete;
    
    const void* data() const;
    void* mutable_data();

    void set_wrapped_number(bool);
    bool is_wrapped_number() const;

    IntArrayRef shape() const;
    IntArrayRef strides() const;
    Dtype dtype() const;
    Device device() const;
    int ndim() const;
    int64_t numel() const;
    int64_t storage_offset() const;

    const std::shared_ptr<StorageImpl>& storage() const;

    // -- Autograd --
    autograd::AutogradMeta* autograd_meta() const;
    void set_autograd_meta(std::unique_ptr<autograd::AutogradMeta> meta);

private:
    std::shared_ptr<StorageImpl> storage_;
    int64_t storage_offset_ = 0;

    DimVector shape_;
    DimVector strides_;
    Dtype dtype_;

    int64_t numel_;
    bool is_wrapped_number_ = false;

    // Autograd metadata
    std::unique_ptr<autograd::AutogradMeta> autograd_meta_;
};

} // namespace detail
} // namespace raxis