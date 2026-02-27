#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <initializer_list>

#include <raxis/core/dtype.h>
#include <raxis/core/scalar.h>
#include <raxis/core/types.h>
#include <raxis/util/exception.h>


namespace raxis::detail {

// ============================================================================
// TensorDataContainer
// ============================================================================
//
// A lightweight recursive type whose only purpose is to let the compiler
// convert arbitrarily nested braced-init-lists into a flat buffer of
// scalars plus a shape:
//
//     raxis::tensor({{1, 2}, {3, 4}})   →  shape={2,2}, data=[1,2,3,4]
//     raxis::tensor({1.0, 2.0, 3.0})    →  shape={3},   data=[1,2,3]
//     raxis::tensor(42)                 →  shape={},    data=[42]


struct TensorDataContainer {

    TensorDataContainer(bool v)     : dtype_(dtypes::Bool),    scalar_(v) {}
    TensorDataContainer(uint8_t v)  : dtype_(dtypes::Uint8),   scalar_(v) {}
    TensorDataContainer(uint16_t v) : dtype_(dtypes::Uint16),  scalar_(v) {}
    TensorDataContainer(int32_t v)  : dtype_(dtypes::Int32),   scalar_(v) {}
    TensorDataContainer(int64_t v)  : dtype_(dtypes::Int64),   scalar_(v) {}
    TensorDataContainer(float v)    : dtype_(dtypes::Float32), scalar_(v) {}
    TensorDataContainer(double v)   : dtype_(dtypes::Float64), scalar_(v) {}

    TensorDataContainer(std::initializer_list<TensorDataContainer> init_list)
        : is_scalar_(false)
    {
        if (init_list.size() == 0) {
            // e.g. tensor({}) → shape {0}, dtype from default
            sizes_.push_back(0);
            dtype_ = default_dtype();
            return;
        }

        const auto& first = *init_list.begin();
        dtype_ = first.dtype_;

        for (const auto& elem : init_list) {
            RAXIS_CHECK(
                elem.sizes_ == first.sizes_,
                "Expected all sub-lists to have the same size, "
                "but got mismatched dimensions"
            );
            RAXIS_CHECK(
                elem.dtype_ == first.dtype_,
                "Expected all elements to have the same scalar type: ",
                first.dtype_.name(), ", but got: ", elem.dtype_.name()
            );
        }

        sizes_.reserve(first.sizes_.size() + 1);
        sizes_.push_back(static_cast<int64_t>(init_list.size()));
        sizes_.insert(sizes_.end(), first.sizes_.begin(), first.sizes_.end());

        init_list_ = init_list;
    }

    bool is_scalar() const { return is_scalar_; }
    Dtype dtype() const { return dtype_; }
    const IntArray& sizes() const { return sizes_; }

    const Scalar& scalar() const {
        RAXIS_INTERNAL_ASSERT(is_scalar_, "scalar() called on non-scalar container");
        return scalar_;
    }

    // Flatten all leaf scalars into row-major order.
    void flatten_into(std::vector<Scalar>& out) const {
        if (is_scalar_) {
            out.push_back(scalar_);
        } else {
            for (const auto& child : init_list_) {
                child.flatten_into(out);
            }
        }
    }

private:
    bool is_scalar_ = true;
    Dtype dtype_ = dtypes::Undefined;
    IntArray sizes_;

    // Active when is_scalar_ == true
    Scalar scalar_;

    // Active when is_scalar_ == false (lifetime tied to caller expression)
    std::initializer_list<TensorDataContainer> init_list_;
};

} // namespace raxis::detail