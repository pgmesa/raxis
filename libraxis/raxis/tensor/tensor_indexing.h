#pragma once

#include <cstdint>
#include <optional>
#include <ostream>
#include <type_traits>
#include <vector>

#include <raxis/export.h>


namespace raxis {

class Scalar;
class Tensor;

} // namespace raxis


namespace raxis::indexing {

constexpr std::nullopt_t None = std::nullopt;

struct RAXIS_API EllipsisIndexType { 
    constexpr EllipsisIndexType() = default; 
};
inline constexpr EllipsisIndexType Ellipsis{};

enum class TensorIndexType : uint8_t {
    None,
    Ellipsis,
    Integer,
    Boolean,
    Slice,
    Tensor
};

// =====================================================================
// Slice
// =====================================================================
//
// Represents a Python-style slice  start:stop:step .
//
//  Slice()              →  :       (entire dimension)
//  Slice(1, 5)          →  1:5
//  Slice(1, None)       →  1:      (from 1 to end)
//  Slice(None, 3)       →  :3      (from start to 3)
//  Slice(0, 10, 2)      →  0:10:2

struct RAXIS_API ResolvedSlice {
    int64_t start;
    int64_t stop;
    int64_t step;
};

struct RAXIS_API Slice {
    std::optional<int64_t> start;
    std::optional<int64_t> stop;
    std::optional<int64_t> step;

    Slice(
        std::optional<int64_t> start_ = None,
        std::optional<int64_t> stop_ = None,
        std::optional<int64_t> step_ = 1
    ) : start(start_), stop(stop_), step(step_) {}

    ResolvedSlice resolve(int64_t dim_size) const;
};

std::ostream& operator<<(std::ostream& os, const Slice& s);

// =====================================================================
// TensorIndex
// =====================================================================
// TensorIndex is used for converting C++ tensor indices such as
// `{None, "...", Ellipsis, 0, true, Slice(1, None, 2), tensor({1, 2})}`
// into its equivalent `vector<TensorIndex>`, so that further tensor
// indexing operations can be performed using the supplied indices.
//
// There is one-to-one correspondence between Python and C++ tensor 
// index types (take a look at pytorch/aten/src/ATen/TensorIndexing.h 
// for more information)

struct RAXIS_API TensorIndex {

    TensorIndex(std::nullopt_t)
        : type_(TensorIndexType::None) {}

    TensorIndex(EllipsisIndexType)
        : type_(TensorIndexType::Ellipsis) {}

    TensorIndex(int64_t value)
        : type_(TensorIndexType::Integer), integer_(value) {}

    template <typename T, std::enable_if_t<std::is_same_v<T, bool>, int> = 0>
    TensorIndex(T boolean)
        : type_(TensorIndexType::Boolean), boolean_(boolean) {}

    TensorIndex(Slice s)
        : type_(TensorIndexType::Slice), slice_(s) {}

    TensorIndexType type() const { return type_; }

    bool is_none() const {
        return type_ == TensorIndexType::None;
    }

    bool is_ellipsis() const {
        return type_ == TensorIndexType::Ellipsis;
    }

    bool is_integer() const { 
        return type_ == TensorIndexType::Integer; 
    }

    bool is_boolean() const {
        return type_ == TensorIndexType::Boolean;
    }

    bool is_slice() const {
        return type_ == TensorIndexType::Slice; 
    }

    bool is_tensor() const {
        return type_ == TensorIndexType::Tensor;
    }

    int64_t integer() const {
        return integer_; 
    }
    
    const Slice& slice() const {
        return slice_;
    }

    bool boolean() const {
        return boolean_;
    }

private:
    TensorIndexType type_;
    int64_t integer_ = 0;
    bool boolean_ = false;
    Slice slice_;
};

std::ostream& operator<<(std::ostream& os, const TensorIndex& idx);

RAXIS_API Tensor get_item(const Tensor& self, ArrayRef<TensorIndex> indices);

RAXIS_API void set_item(const Tensor& self, ArrayRef<TensorIndex> indices, const Tensor& value);
RAXIS_API void set_item(const Tensor& self, ArrayRef<TensorIndex> indices, Scalar value);

} // namespace raxis::indexing