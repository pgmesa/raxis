#pragma once

#include <cstdint>

#include <raxis/export.h>


namespace raxis {

class Tensor;

// From: aten/src/ATen/MemoryOverlap.h
// Ops that write into an existing tensor (in-place ops, copy_, out=) check
// that the writes are well defined. Two cases are not:
//   - Internal overlap: the written tensor has elements that share memory (a
//     broadcast view, stride 0), so several results land on the same address
//     and which one survives depends on the iteration order and threads.
//   - Partial overlap between an output and an input: the op would read
//     elements it has already overwritten. Full overlap (same elements, same
//     layout, e.g. a.add_(a)) is fine: each element is read before it is
//     written.
// Exact answers need integer programming on the strides, so only the cheap
// cases are decided and the rest is reported as TooHard (and allowed).

enum class MemOverlap : int8_t { No, Yes, TooHard };

enum class MemOverlapStatus : int8_t { Full, Partial, No, TooHard };

RAXIS_API MemOverlap has_internal_overlap(const Tensor& t);
RAXIS_API void assert_no_internal_overlap(const Tensor& t);

RAXIS_API MemOverlapStatus get_overlap_status(const Tensor& a, const Tensor& b);
RAXIS_API void assert_no_partial_overlap(const Tensor& a, const Tensor& b);
RAXIS_API void assert_no_overlap(const Tensor& a, const Tensor& b);

} // namespace raxis
