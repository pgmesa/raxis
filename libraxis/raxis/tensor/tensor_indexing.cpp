
#include <raxis/tensor/tensor_indexing.h>

#include <ostream>
#include <vector>

#include <raxis/tensor/tensor.h>
#include <raxis/core/scalar.h>
#include <raxis/util/exception.h>


namespace raxis::indexing {


ResolvedSlice Slice::resolve(int64_t dim_size) const {
    int64_t start = this->start.value_or(0);
    int64_t stop = this->stop.value_or(dim_size);
    int64_t step = this->step.value_or(1);

    RAXIS_CHECK(step != 0, "slice step cannot be zero");
    RAXIS_CHECK(step > 0,
        "Negative step not yet supported in indexing (got step=", step, ")");

    auto clamp = [](int64_t v, int64_t lo, int64_t hi) -> int64_t {
        if (v < lo) return lo;
        if (v > hi) return hi;
        return v;
    };

    // Handle negative indexing
    if (start < 0) start += dim_size;
    if (stop  < 0) stop  += dim_size;

    // Clamp to valid range
    start = clamp(start, 0, dim_size);
    stop = clamp(stop,  0, dim_size);

    return {start, stop, step};
}


std::ostream& operator<<(std::ostream& os, const Slice& s) {
    auto fmt = [&](std::optional<int64_t> v) {
        if (v.has_value()) os << *v;
        else os << "None";
    };
    fmt(s.start); os << ":"; fmt(s.stop); os << ":"; fmt(s.step);
    return os;
}


std::ostream& operator<<(std::ostream& os, const TensorIndex& idx) {
    switch (idx.type()) {
        case TensorIndexType::Integer:   os << idx.integer(); break;
        case TensorIndexType::Slice:     os << idx.slice();   break;
        case TensorIndexType::None:      os << "None";        break;
        case TensorIndexType::Ellipsis:  os << "...";         break;
        case TensorIndexType::Boolean:   os << (idx.boolean() ? "true" : "false"); break;
        case TensorIndexType::Tensor:    os << "<tensor>";    break;
    }
    return os;
}


namespace {

/// Count how many dimensions the indices actually consume.
int64_t count_specified_dims(const std::vector<TensorIndex>& indices) {
    int64_t count = 0;
    for (const auto& idx : indices) {
        switch (idx.type()) {
            case TensorIndexType::Integer:
            case TensorIndexType::Slice:
                ++count;
                break;
            case TensorIndexType::Boolean:
            case TensorIndexType::Tensor:
                RAXIS_INTERNAL_ASSERT(false,
                    "count_specified_dims: advanced indexing not yet handled correctly");
            default:
                break;
        }
    }
    return count;
}

/// Check if any index requires advanced (copy-based) indexing.
bool needs_advanced_indexing(const std::vector<TensorIndex>& indices) {
    for (const auto& idx : indices) {
        if (idx.is_tensor() || idx.is_boolean()) {
            return true;
        }
    }
    return false;
}


} // anonymous namespace


Tensor get_item(
    const Tensor& self,
    const std::vector<TensorIndex>& indices
) {
    RAXIS_CHECK(self.defined(), "Cannot index an undefined tensor");
    RAXIS_CHECK(!indices.empty(), "Empty index list is not valid");

    if (needs_advanced_indexing(indices)) {
        RAXIS_CHECK(false,
            "Advanced indexing (bool/tensor indices) is not yet implemented. "
            "Only integer, slice, None, and Ellipsis are currently supported.");
    }

    // Validate and expand Ellipsis
    int64_t num_ellipsis = 0;
    int64_t ellipsis_pos = -1;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i].is_ellipsis()) {
            ++num_ellipsis;
            ellipsis_pos = static_cast<int64_t>(i);
        }
    }
    RAXIS_CHECK(num_ellipsis <= 1,
        "Only one Ellipsis ('...') is allowed in an index expression");

    const int64_t specified = count_specified_dims(indices);
    RAXIS_CHECK(specified <= self.ndim(),
        "Too many indices for tensor of dimension ", self.ndim(),
        " (got ", specified, " index dimensions)");

    // Expand Ellipsis into full-dim slices
    std::vector<TensorIndex> expanded_buf;
    if (num_ellipsis == 1) {
        int64_t expand_count = self.ndim() - specified;

        expanded_buf.reserve(indices.size() - 1 + expand_count);

        for (size_t i = 0; i < indices.size(); ++i) {
            if (static_cast<int64_t>(i) == ellipsis_pos) {
                for (int64_t j = 0; j < expand_count; ++j) {
                    expanded_buf.push_back(Slice{});   // full dimension
                }
            } else {
                expanded_buf.push_back(indices[i]);
            }
        }
    } 
    const auto& expanded = (num_ellipsis == 1) ? expanded_buf : indices;

    // Walk through expanded indices and build the view
    Tensor result = self;
    int64_t dim = 0;

    for (const auto& idx : expanded) {
        if (idx.is_integer()) {
            // select removes the dimension, so dim stays the same
            // for the next index (the dims above shift down).
            result = result.select(dim, idx.integer());
        } else if (idx.is_slice()) {
            const auto [start, stop, step] = \
                idx.slice().resolve(result.shape()[dim]);
            // Skip the slice call if it's a no-op
            // (selects the entire dimension with step 1).
            bool is_full_slice = (start == 0)
                && (stop >= result.shape()[dim])
                && (step == 1);

            if (!is_full_slice) {
                result = result.slice(dim, start, stop, step);
            }
            ++dim;
        } else if (idx.is_none()) {
            result = result.unsqueeze(dim);
            ++dim;
        } else if (idx.is_ellipsis()) {
            // Should have been expanded above, but if it somehow
            // appears here, it's a no-op.
        } else {
            RAXIS_INTERNAL_ASSERT(false,
                "Unhandled TensorIndex Type in apply_indexing");
        }
    }

    // If result is the exact same TensorImpl (e.g. all indices were
    // full slices or a lone Ellipsis), return an alias to guarantee
    // we always return a distinct Tensor object (PyTorch/numpy semantics).
    if (result.is_same(self)) {
        return self.alias();
    }

    return result;
}

void set_item(
    const Tensor& self,
    const std::vector<TensorIndex>& indices,
    const Tensor& value
) {
    Tensor view = get_item(self, indices);
    view.copy_(value);
}

void set_item(
    const Tensor& self,
    const std::vector<TensorIndex>& indices,
    Scalar value
) {
    Tensor view = get_item(self, indices);
    view.fill_(value);
}


} // namespace raxis::indexing