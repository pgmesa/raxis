#pragma once

#include <array>
#include <cstdint>

#include <raxis/tensor/tensor.h>
#include <raxis/util/array_ref.h>
#include <raxis/util/exception.h>


// One kernel launch for many tensors.
// From: aten/src/ATen/native/cuda/MultiTensorApply.cuh
//
// Optimizer steps and foreach ops touch every parameter: tens to hundreds of
// tensors, many of them small. One launch per tensor (or one per elementwise
// op per tensor) pays a launch for little work each time. multi_tensor_apply
// packs many tensors into one launch: the kernel arguments carry a table of
// their pointers, and each block looks up which tensor and which chunk of
// kChunkSize elements it owns (block_to_tensor, block_to_chunk). When the
// table or the block list is full, the kernel is launched and the table
// starts over; a tensor cut by a launch continues in slot 0 of the next one.
// The table must fit in the 4 KB of kernel arguments, so the more lists per
// tensor (depth), the fewer tensors per launch.

namespace raxis::cuda {

constexpr int64_t kChunkSize = 65536;
constexpr int kMultiTensorBlockSize = 512;
constexpr int kMaxBlocksPerLaunch = 320;
// Tensors per launch for 1 to 4 lists
constexpr int kMaxTensorsPerLaunch[4] = {80, 64, 48, 36};

template <int depth>
struct TensorListMetadata {
    static constexpr int kMaxTensors = kMaxTensorsPerLaunch[depth - 1];
    void* addresses[depth][kMaxTensors];
    int64_t numel[kMaxTensors];
    int32_t tensor_index[kMaxTensors];   // position of the tensor in the lists
    float value[kMaxTensors];            // a number per tensor (Adam's step count)
    uint8_t block_to_tensor[kMaxBlocksPerLaunch];
    int32_t block_to_chunk[kMaxBlocksPerLaunch];
};

static_assert(sizeof(TensorListMetadata<1>) < 3800 && sizeof(TensorListMetadata<4>) < 3800,
              "the metadata and the kernel's other arguments must fit in 4 KB");

/// Calls launch(metadata, n_blocks) once per batch of tensors; the caller
/// launches its kernel with n_blocks blocks of kMultiTensorBlockSize threads.
/// lists[d][i] is the d-th tensor of entry i; values[i] (optional) goes to
/// metadata.value. Empty tensors are skipped.
/// From: MultiTensorApply.cuh (multi_tensor_apply_for_fused_optimizer)
template <int depth, typename Launch>
void multi_tensor_apply(const std::array<ArrayRef<Tensor>, depth>& lists, ArrayRef<double> values,
                        const Launch& launch) {
    using Metadata = TensorListMetadata<depth>;
    Metadata tl;
    int n_tensors = 0;
    int n_blocks = 0;
    for (size_t i = 0; i < lists[0].size(); i++) {
        const int64_t numel = lists[0][i].numel();
        if (numel == 0) {
            continue;
        }
        for (int d = 0; d < depth; d++) {
            tl.addresses[d][n_tensors] = const_cast<void*>(lists[d][i].data_ptr());
        }
        tl.numel[n_tensors] = numel;
        tl.tensor_index[n_tensors] = static_cast<int32_t>(i);
        tl.value[n_tensors] = values.empty() ? 0.0f : static_cast<float>(values[i]);
        n_tensors++;
        const int64_t chunks = (numel + kChunkSize - 1) / kChunkSize;
        RAXIS_CHECK(chunks <= INT32_MAX, "multi_tensor_apply: tensor too large");
        for (int64_t chunk = 0; chunk < chunks; chunk++) {
            tl.block_to_tensor[n_blocks] = static_cast<uint8_t>(n_tensors - 1);
            tl.block_to_chunk[n_blocks] = static_cast<int32_t>(chunk);
            n_blocks++;
            const bool tensors_full = n_tensors == Metadata::kMaxTensors && chunk == chunks - 1;
            if (!tensors_full && n_blocks < kMaxBlocksPerLaunch) {
                continue;
            }
            launch(tl, n_blocks);
            n_blocks = 0;
            if (chunk == chunks - 1) {
                n_tensors = 0;
            } else {
                // The rest of this tensor's chunks go in the next launch, from slot 0
                for (int d = 0; d < depth; d++) {
                    tl.addresses[d][0] = tl.addresses[d][n_tensors - 1];
                }
                tl.numel[0] = tl.numel[n_tensors - 1];
                tl.tensor_index[0] = tl.tensor_index[n_tensors - 1];
                tl.value[0] = tl.value[n_tensors - 1];
                n_tensors = 1;
            }
        }
    }
    if (n_blocks > 0) {
        launch(tl, n_blocks);
    }
}

/// The part of a tensor a block owns: its first element and its length
template <typename T>
struct Chunk {
    T* data;
    int64_t n;
};

/// The chunk of list d that block blockIdx.x owns
template <typename T, int depth>
__device__ inline Chunk<T> block_chunk(const TensorListMetadata<depth>& tl, int d) {
    const int t = tl.block_to_tensor[blockIdx.x];
    const int64_t begin = static_cast<int64_t>(tl.block_to_chunk[blockIdx.x]) * kChunkSize;
    const int64_t n = tl.numel[t] - begin < kChunkSize ? tl.numel[t] - begin : kChunkSize;
    return {static_cast<T*>(tl.addresses[d][t]) + begin, n};
}

} // namespace raxis::cuda
