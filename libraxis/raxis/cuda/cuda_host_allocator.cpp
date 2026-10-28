#include <raxis/cuda/cuda_host_allocator.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <deque>
#include <map>
#include <mutex>
#include <unordered_set>
#include <utility>
#include <vector>

#include <raxis/cuda/cuda_exception.h>


// A caching allocator for pinned host memory.
// From: aten/src/ATen/core/CachingHostAllocator.h (CachingHostAllocatorImpl)
// and aten/src/ATen/cuda/CachingHostAllocator.cpp
//
// cudaHostAlloc is slow (it page-locks the memory and takes a global lock in
// the driver), and a data loader pins a fresh batch every step. So, as the
// device allocator, freed blocks are kept and reused: sizes are rounded up
// to a power of two, and a free list per size gives a block back in O(log).
//
// The catch is that a pinned tensor may be freed while an asynchronous copy
// still reads it (x.pin_memory().to("cuda", non_blocking=True), then x goes
// out of scope). Reusing its block then would let new data overwrite what
// the GPU has not read yet. So the copy records its stream on the block
// (host_allocator_record_stream); freeing a block with recorded streams puts
// an event on each of them and the block waits in a queue until the events
// complete. Every allocation first moves the blocks whose events completed
// back to the free lists.
//
// Left out from PyTorch's: per-size event processing, background threads,
// cudaHostRegister, statistics.

namespace raxis::cuda {
namespace {

struct HostBlock {
    size_t size;
    void* ptr;
    std::unordered_set<cudaStream_t> streams;   // used by copies since it was allocated
    int pending_events = 0;
};

class CachingHostAllocator final : public Allocator {
public:
    DataPtr allocate(size_t nbytes) override {
        if (nbytes == 0) {
            return DataPtr(nullptr, Device(DeviceType::CPU));
        }
        // From: CachingHostAllocator.h (c10::llvm::PowerOf2Ceil)
        const size_t size = std::bit_ceil(nbytes);
        std::lock_guard<std::mutex> lock(mutex_);
        process_events();
        HostBlock* block = nullptr;
        auto it = free_.find(size);
        if (it != free_.end() && !it->second.empty()) {
            block = it->second.back();
            it->second.pop_back();
        } else {
            void* ptr = nullptr;
            RAXIS_CUDA_CHECK(cudaHostAlloc(&ptr, size, cudaHostAllocDefault));
            block = new HostBlock{size, ptr, {}};
            blocks_.push_back(block);
        }
        return DataPtr(block->ptr, block, &delete_block, Device(DeviceType::CPU));
    }

    void free(HostBlock* block) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (block->streams.empty()) {
            free_[block->size].push_back(block);
            return;
        }
        for (cudaStream_t stream : block->streams) {
            cudaEvent_t event;
            RAXIS_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
            RAXIS_CUDA_CHECK(cudaEventRecord(event, stream));
            events_.emplace_back(event, block);
            block->pending_events++;
        }
        block->streams.clear();
    }

    void record_stream(const DataPtr& data_ptr, cudaStream_t stream) {
        if (data_ptr.get_deleter() != &delete_block) {
            return;
        }
        std::lock_guard<std::mutex> lock(mutex_);
        static_cast<HostBlock*>(data_ptr.get_context())->streams.insert(stream);
    }

    void empty_cache() {
        std::lock_guard<std::mutex> lock(mutex_);
        process_events();
        for (auto& [size, blocks] : free_) {
            for (HostBlock* block : blocks) {
                RAXIS_CUDA_CHECK(cudaFreeHost(block->ptr));
                blocks_.erase(std::find(blocks_.begin(), blocks_.end(), block));
                delete block;
            }
        }
        free_.clear();
    }

private:
    // Moves the blocks whose events all completed to the free lists. Events
    // are queued in recording order, so the first one not ready stops it
    void process_events() {
        while (!events_.empty()) {
            auto [event, block] = events_.front();
            const cudaError_t status = cudaEventQuery(event);
            if (status == cudaErrorNotReady) {
                (void)cudaGetLastError();   // clear the "error" the query left
                break;
            }
            RAXIS_CUDA_CHECK(status);
            RAXIS_CUDA_CHECK(cudaEventDestroy(event));
            events_.pop_front();
            if (--block->pending_events == 0) {
                free_[block->size].push_back(block);
            }
        }
    }

    static void delete_block(void* ctx);

    std::mutex mutex_;
    std::vector<HostBlock*> blocks_;                    // every block, for empty_cache
    std::map<size_t, std::vector<HostBlock*>> free_;    // free blocks by size
    std::deque<std::pair<cudaEvent_t, HostBlock*>> events_;
};

// Never destroyed: tensors may outlive static destruction at exit
CachingHostAllocator& instance() {
    static auto* allocator = new CachingHostAllocator();
    return *allocator;
}

void CachingHostAllocator::delete_block(void* ctx) {
    instance().free(static_cast<HostBlock*>(ctx));
}

} // anonymous namespace

Allocator* host_allocator() {
    return &instance();
}

void host_allocator_record_stream(const DataPtr& data_ptr, cudaStream_t stream) {
    instance().record_stream(data_ptr, stream);
}

void host_empty_cache() {
    instance().empty_cache();
}

} // namespace raxis::cuda
