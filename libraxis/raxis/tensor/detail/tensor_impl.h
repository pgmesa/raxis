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
enum class CreationMeta : uint8_t;
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

    bool is_contiguous() const;
    bool is_non_overlapping_and_dense() const;

    const std::shared_ptr<StorageImpl>& storage() const;

    // -- Autograd --
    autograd::AutogradMeta* autograd_meta() const;
    void set_autograd_meta(std::unique_ptr<autograd::AutogradMeta> meta);

    // -- Views --
    // From: torch/csrc/autograd/variable.h (DifferentiableViewMeta)
    // A view remembers its base: the tensor (never itself a view) whose
    // memory it looks at. An in-place op on the view writes the base too, so
    // autograd records it on the base (CopySlices); an in-place op on the
    // base or on another view changes this view, so its grad_fn is rebuilt
    // from the base. PyTorch keeps this in an autograd meta allocated for
    // every view; here it is a few fields, with no extra allocation.
    bool is_view() const { return view_base_ != nullptr; }
    const std::shared_ptr<TensorImpl>& view_base() const { return view_base_; }
    autograd::CreationMeta creation_meta() const { return creation_meta_; }
    // Version of the memory when the grad_fn of the view was last set
    uint32_t view_version() const { return view_version_; }

    void set_view_base(std::shared_ptr<TensorImpl> base, autograd::CreationMeta creation_meta);
    void set_creation_meta(autograd::CreationMeta creation_meta) { creation_meta_ = creation_meta; }
    void set_view_version(uint32_t version) { view_version_ = version; }

private:
    // Shape and strides never change after construction, so the layout
    // properties are computed once instead of on every check
    void compute_layout();

    std::shared_ptr<StorageImpl> storage_;
    int64_t storage_offset_ = 0;

    DimVector shape_;
    DimVector strides_;
    Dtype dtype_;

    int64_t numel_;
    bool is_wrapped_number_ = false;
    bool is_contiguous_ = true;
    bool is_non_overlapping_and_dense_ = true;

    // Autograd metadata
    std::unique_ptr<autograd::AutogradMeta> autograd_meta_;

    std::shared_ptr<TensorImpl> view_base_;
    uint32_t view_version_ = 0;
    autograd::CreationMeta creation_meta_{};
};

} // namespace detail
} // namespace raxis