#include <raxis/cuda/cuda_caching_allocator.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include <raxis/core/allocator.h>
#include <raxis/cuda/cuda_exception.h>
#include <raxis/cuda/cuda_functions.h>
#include <raxis/util/data_ptr.h>


// A simplified port of PyTorch's CUDA caching allocator.
// From: c10/cuda/CUDACachingAllocator.cpp
//
// cudaMalloc and cudaFree are slow (cudaFree also synchronizes the device),
// and a training step allocates and frees thousands of tensors. So memory
// is never returned to CUDA: a freed tensor's block goes back to a pool, and
// the next allocation of a similar size reuses it. CUDA is only asked for
// memory when no cached block fits, and only gets it back with empty_cache()
// or when an allocation would otherwise fail.
//
// - A segment is one cudaMalloc. Allocations of up to 1 MiB come from 2 MiB
//   segments shared by many small blocks; larger ones from 20 MiB segments
//   (under 10 MiB) or a segment of their own rounded to 2 MiB.
// - A segment is split into blocks, a doubly linked list in address order.
//   Allocating takes the smallest free block that fits (best fit) and
//   splits off the rest; freeing merges the block with its free neighbours,
//   so the segment can be whole again.
// - Small and large blocks live in separate pools, so the long-lived large
//   tensors (weights) do not fragment the segments the many short-lived
//   small ones cycle through.
//
// Left out from PyTorch's: streams (with one stream per device a freed block
// can be reused at once, see current_stream()), record_stream and its
// events, expandable segments, CUDA graph pools, the allocator config
// (max_split_size, garbage collection), memory snapshots.

namespace raxis::cuda {
namespace {

// From: c10/core/AllocatorConfig.h
constexpr size_t kMinBlockSize = 512;       // every size is rounded to 512 bytes
constexpr size_t kSmallSize = 1048576;      // the largest "small" allocation, 1 MiB
constexpr size_t kSmallBuffer = 2097152;    // small allocations share 2 MiB segments
constexpr size_t kLargeBuffer = 20971520;   // allocations under 10 MiB share 20 MiB segments
constexpr size_t kMinLargeAlloc = 10485760; // ... larger ones get their own segment
constexpr size_t kRoundLarge = 2097152;     // rounded to 2 MiB

struct BlockPool;

struct Block {
    DeviceIndex device;
    BlockPool* pool;
    void* ptr;
    size_t size;                // bytes of the block (rounded)
    size_t requested_size = 0;  // bytes asked for, when allocated
    bool allocated = false;
    Block* prev = nullptr;      // neighbours in the same segment
    Block* next = nullptr;

    Block(DeviceIndex device, BlockPool* pool, void* ptr, size_t size)
        : device(device), pool(pool), ptr(ptr), size(size) {}

    bool is_split() const {
        return prev != nullptr || next != nullptr;
    }
};

// Ordered by size, then address: lower_bound(size) is the smallest block
// that fits, and among equal sizes the lowest address, which keeps
// allocations packed at the start of the segments. (PyTorch orders by
// stream first: a block is only reused on the stream that freed it.)
struct BlockComparator {
    bool operator()(const Block* a, const Block* b) const {
        if (a->size != b->size) {
            return a->size < b->size;
        }
        return reinterpret_cast<uintptr_t>(a->ptr) < reinterpret_cast<uintptr_t>(b->ptr);
    }
};

/// The free blocks of one size class
struct BlockPool {
    explicit BlockPool(bool is_small) : is_small(is_small) {}

    std::set<Block*, BlockComparator> blocks;
    const bool is_small;
};

/// Current and peak value of a statistic
struct Stat {
    int64_t current = 0;
    int64_t peak = 0;

    void increase(int64_t amount) {
        current += amount;
        peak = std::max(peak, current);
    }

    void decrease(int64_t amount) {
        current -= amount;
    }
};

size_t round_size(size_t size) {
    if (size < kMinBlockSize) {
        return kMinBlockSize;
    }
    return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
}

/// Size of the segment to cudaMalloc for a block of `size` bytes
size_t get_allocation_size(size_t size) {
    if (size <= kSmallSize) {
        return kSmallBuffer;
    } else if (size < kMinLargeAlloc) {
        return kLargeBuffer;
    } else {
        return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
}

std::string format_size(size_t bytes) {
    if (bytes < 1024 * 1024) {
        return std::to_string(bytes / 1024) + " KiB";
    } else if (bytes < 1024ull * 1024 * 1024) {
        return std::to_string(bytes / (1024 * 1024)) + " MiB";
    }
    double gib = static_cast<double>(bytes) / (1024.0 * 1024 * 1024);
    std::string s = std::to_string(gib);
    return s.substr(0, s.find('.') + 3) + " GiB";
}


/// The pools and statistics of one device
class DeviceCachingAllocator {
public:
    explicit DeviceCachingAllocator(DeviceIndex device) : device_(device) {}

    Block* malloc(size_t orig_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t size = round_size(orig_size);
        BlockPool& pool = size <= kSmallSize ? small_blocks_ : large_blocks_;

        Block* block = get_free_block(pool, size);
        if (block == nullptr) {
            block = alloc_block(pool, get_allocation_size(size), orig_size);
        }

        // From: DeviceCachingAllocator::alloc_found_block
        // The rest of the block stays in the pool. A large block is only
        // split when more than 1 MiB is left, so a large allocation does not
        // leave behind a sliver nobody uses
        size_t remaining_size = block->size - size;
        bool should_split = pool.is_small ? remaining_size >= kMinBlockSize
                                          : remaining_size > kSmallSize;
        if (should_split) {
            Block* remaining = block;
            block = new Block(device_, &pool, remaining->ptr, size);
            block->prev = remaining->prev;
            if (block->prev) {
                block->prev->next = block;
            }
            block->next = remaining;
            remaining->prev = block;
            remaining->ptr = static_cast<char*>(remaining->ptr) + size;
            remaining->size -= size;
            pool.blocks.insert(remaining);
        }

        block->allocated = true;
        block->requested_size = orig_size;
        allocated_bytes_.increase(static_cast<int64_t>(block->size));
        return block;
    }

    void free(Block* block) {
        std::lock_guard<std::mutex> lock(mutex_);
        block->allocated = false;
        allocated_bytes_.decrease(static_cast<int64_t>(block->size));

        // From: DeviceCachingAllocator::free_block
        BlockPool& pool = *block->pool;
        for (Block* neighbour : {block->prev, block->next}) {
            try_merge_blocks(block, neighbour, pool);
        }
        pool.blocks.insert(block);
    }

    /// Returns every whole cached segment to CUDA
    void empty_cache() {
        std::lock_guard<std::mutex> lock(mutex_);
        release_cached_blocks();
    }

    CUDAMemoryStats stats() {
        std::lock_guard<std::mutex> lock(mutex_);
        CUDAMemoryStats s;
        s.allocated_bytes = allocated_bytes_.current;
        s.max_allocated_bytes = allocated_bytes_.peak;
        s.reserved_bytes = reserved_bytes_.current;
        s.max_reserved_bytes = reserved_bytes_.peak;
        return s;
    }

    void reset_peak_stats() {
        std::lock_guard<std::mutex> lock(mutex_);
        allocated_bytes_.peak = allocated_bytes_.current;
        reserved_bytes_.peak = reserved_bytes_.current;
    }

private:
    Block* get_free_block(BlockPool& pool, size_t size) {
        Block key(device_, &pool, nullptr, size);
        auto it = pool.blocks.lower_bound(&key);
        if (it == pool.blocks.end()) {
            return nullptr;
        }
        Block* block = *it;
        pool.blocks.erase(it);
        return block;
    }

    /// A new segment from cudaMalloc. On failure, frees the cached segments
    /// and tries once more before raising out of memory
    Block* alloc_block(BlockPool& pool, size_t alloc_size, size_t orig_size) {
        void* ptr = nullptr;
        cudaError_t err = cudaMalloc(&ptr, alloc_size);
        if (err == cudaErrorMemoryAllocation) {
            cudaGetLastError();  // cudaMalloc's error is sticky until read
            release_cached_blocks();
            err = cudaMalloc(&ptr, alloc_size);
        }
        if (err == cudaErrorMemoryAllocation) {
            cudaGetLastError();
            size_t device_free = 0;
            size_t device_total = 0;
            cudaMemGetInfo(&device_free, &device_total);
            RAXIS_CHECK(false,
                "CUDA out of memory. Tried to allocate ", format_size(orig_size),
                ". GPU ", static_cast<int>(device_), " has a total capacity of ",
                format_size(device_total), " of which ", format_size(device_free),
                " is free. raxis holds ", format_size(allocated_bytes_.current),
                " in tensors and ", format_size(reserved_bytes_.current),
                " reserved (allocated plus cached).");
        }
        RAXIS_CUDA_CHECK(err);
        reserved_bytes_.increase(static_cast<int64_t>(alloc_size));
        return new Block(device_, &pool, ptr, alloc_size);
    }

    /// Merges `src` into `dst` when `src` is a free neighbour in the segment
    void try_merge_blocks(Block* dst, Block* src, BlockPool& pool) {
        if (src == nullptr || src->allocated) {
            return;
        }
        if (dst->prev == src) {  // [src dst]
            dst->ptr = src->ptr;
            dst->prev = src->prev;
            if (dst->prev) {
                dst->prev->next = dst;
            }
        } else {                 // [dst src]
            dst->next = src->next;
            if (dst->next) {
                dst->next->prev = dst;
            }
        }
        dst->size += src->size;
        pool.blocks.erase(src);
        delete src;
    }

    /// cudaFree of every cached block that is a whole segment (a split
    /// segment still holds a tensor in one of its blocks)
    void release_cached_blocks() {
        for (BlockPool* pool : {&small_blocks_, &large_blocks_}) {
            for (auto it = pool->blocks.begin(); it != pool->blocks.end();) {
                Block* block = *it;
                if (block->is_split()) {
                    ++it;
                    continue;
                }
                // cudaFree waits for the device: no kernel can still be
                // using the memory
                RAXIS_CUDA_CHECK(cudaFree(block->ptr));
                reserved_bytes_.decrease(static_cast<int64_t>(block->size));
                it = pool->blocks.erase(it);
                delete block;
            }
        }
    }

    const DeviceIndex device_;
    std::mutex mutex_;
    BlockPool small_blocks_{true};
    BlockPool large_blocks_{false};
    Stat allocated_bytes_;  // in blocks handed to tensors
    Stat reserved_bytes_;   // in segments from cudaMalloc
};


class CUDACachingAllocator final : public Allocator {
public:
    DataPtr allocate(size_t nbytes) override {
        DeviceIndex device = current_device();
        if (nbytes == 0) {
            return DataPtr(nullptr, Device(DeviceType::CUDA, device));
        }
        Block* block = device_allocator(device).malloc(nbytes);
        {
            std::lock_guard<std::mutex> lock(blocks_mutex_);
            allocated_blocks_[block->ptr] = block;
        }
        return DataPtr(block->ptr, block->ptr, &raw_delete, Device(DeviceType::CUDA, device));
    }

    void free(void* ptr) {
        Block* block = nullptr;
        {
            std::lock_guard<std::mutex> lock(blocks_mutex_);
            auto it = allocated_blocks_.find(ptr);
            RAXIS_INTERNAL_ASSERT(it != allocated_blocks_.end(), "freeing a pointer the CUDA allocator does not own");
            block = it->second;
            allocated_blocks_.erase(it);
        }
        device_allocator(block->device).free(block);
    }

    DeviceCachingAllocator& device_allocator(DeviceIndex device) {
        // Created on the first allocation: loading the library must not
        // touch CUDA (a machine without a GPU still imports raxis)
        std::call_once(init_, [this] {
            for (int i = 0; i < device_count(); i++) {
                device_allocators_.push_back(std::make_unique<DeviceCachingAllocator>(static_cast<DeviceIndex>(i)));
            }
        });
        RAXIS_CHECK(device >= 0 && device < static_cast<int>(device_allocators_.size()),
            "Invalid CUDA device index ", static_cast<int>(device));
        return *device_allocators_[device];
    }

    int num_initialized_devices() const {
        return static_cast<int>(device_allocators_.size());
    }

private:
    // The deleter of every DataPtr: finds the block of the pointer
    static void raw_delete(void* ptr);

    std::once_flag init_;
    std::vector<std::unique_ptr<DeviceCachingAllocator>> device_allocators_;
    std::mutex blocks_mutex_;
    std::unordered_map<void*, Block*> allocated_blocks_;
};

// Never destroyed: tensors that outlive the library's static destructors
// (Python frees objects after them at exit) still call its deleter
CUDACachingAllocator* const g_allocator = new CUDACachingAllocator();
REGISTER_ALLOCATOR(DeviceType::CUDA, g_allocator)

void CUDACachingAllocator::raw_delete(void* ptr) {
    g_allocator->free(ptr);
}

} // anonymous namespace

CUDAMemoryStats memory_stats(DeviceIndex device) {
    return g_allocator->device_allocator(device).stats();
}

void reset_peak_memory_stats(DeviceIndex device) {
    g_allocator->device_allocator(device).reset_peak_stats();
}

void empty_cache() {
    for (int i = 0; i < g_allocator->num_initialized_devices(); i++) {
        g_allocator->device_allocator(static_cast<DeviceIndex>(i)).empty_cache();
    }
}

} // namespace raxis::cuda
