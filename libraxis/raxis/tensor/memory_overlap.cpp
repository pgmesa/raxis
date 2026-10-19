
#include <raxis/tensor/memory_overlap.h>

#include <raxis/util/exception.h>

#include <raxis/tensor/tensor.h>


namespace raxis {

// From: aten/src/ATen/MemoryOverlap.cpp
MemOverlap has_internal_overlap(const Tensor& t) {
    if (t.is_non_overlapping_and_dense()) {
        return MemOverlap::No;
    }

    IntArrayRef shape = t.shape();
    IntArrayRef strides = t.strides();
    for (size_t i = 0; i < shape.size(); i++) {
        if (shape[i] > 1 && strides[i] == 0) {
            return MemOverlap::Yes;
        }
    }

    // Sliced tensors have gaps but no repeats; overlapping strides that are
    // not zero (only reachable through as_strided) are not detected
    return MemOverlap::TooHard;
}

void assert_no_internal_overlap(const Tensor& t) {
    RAXIS_CHECK(has_internal_overlap(t) != MemOverlap::Yes,
        "unsupported operation: more than one element of the written-to tensor "
        "refers to a single memory location. Please clone() the tensor before "
        "performing the operation.");
}

// From: aten/src/ATen/MemoryOverlap.cpp
MemOverlapStatus get_overlap_status(const Tensor& a, const Tensor& b) {
    if (a.is_same(b)) return MemOverlapStatus::Full;
    if (a.numel() == 0 || b.numel() == 0) {
        return MemOverlapStatus::No;
    }
    // With gaps the address ranges can interleave without sharing elements
    // (a[::2] and a[1::2]), so only dense tensors are compared
    if (!a.is_non_overlapping_and_dense() || !b.is_non_overlapping_and_dense()) {
        return MemOverlapStatus::TooHard;
    }

    // Dense tensors occupy exactly [data, data + nbytes), whatever the order
    // of their dims, so comparing the two ranges is enough
    if (a.unsafe_get_tensor_impl()->storage() == b.unsafe_get_tensor_impl()->storage()) {
        const auto* a_begin = static_cast<const char*>(a.data_ptr());
        const auto* a_end = a_begin + a.numel() * static_cast<int64_t>(a.dtype().itemsize());
        const auto* b_begin = static_cast<const char*>(b.data_ptr());
        const auto* b_end = b_begin + b.numel() * static_cast<int64_t>(b.dtype().itemsize());

        if (a_begin == b_begin && a_end == b_end) {
            // Same block but another layout (a and a.t()): elements map to
            // different addresses
            return a.strides() == b.strides() ? MemOverlapStatus::Full : MemOverlapStatus::Partial;
        }
        if (a_begin < b_end && b_begin < a_end) {
            return MemOverlapStatus::Partial;
        }
    }
    return MemOverlapStatus::No;
}

void assert_no_partial_overlap(const Tensor& a, const Tensor& b) {
    RAXIS_CHECK(get_overlap_status(a, b) != MemOverlapStatus::Partial,
        "unsupported operation: some elements of the input tensor and "
        "the written-to tensor refer to a single memory location. "
        "Please clone() the tensor before performing the operation.");
}

void assert_no_overlap(const Tensor& a, const Tensor& b) {
    const auto lap = get_overlap_status(a, b);
    RAXIS_CHECK(lap != MemOverlapStatus::Partial && lap != MemOverlapStatus::Full,
        "unsupported operation: some elements of the input tensor and "
        "the written-to tensor refer to a single memory location. "
        "Please clone() the tensor before performing the operation.");
}

} // namespace raxis
