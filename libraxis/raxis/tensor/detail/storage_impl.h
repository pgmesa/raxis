#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <utility>

#include <raxis/core/device.h>
#include <raxis/util/data_ptr.h>


namespace raxis {
struct Allocator;
} // namespace raxis

namespace raxis::detail {

/// Owns the raw data allocation for one or more tensors.
class StorageImpl {
public:
    StorageImpl(DataPtr data, size_t nbytes, Allocator* allocator = nullptr)
        : data_ptr_(std::move(data)),
          nbytes_(nbytes),
          allocator_(allocator),
          resizable_(allocator != nullptr) {}

    StorageImpl() = delete;
    ~StorageImpl() = default;
    StorageImpl(const StorageImpl&) = delete;
    StorageImpl& operator=(const StorageImpl&) = delete;
    StorageImpl(StorageImpl&&) = delete;
    StorageImpl& operator=(StorageImpl&&) = delete;

    const void* data() const {
        return data_ptr_.get(); 
    }
    
    void* mutable_data() {
        return data_ptr_.mutable_get();
    }

    size_t nbytes() const {
        return nbytes_;
    }

    Device device() const {
        return data_ptr_.device();
    }

    const DataPtr& data_ptr() const {
        return data_ptr_;
    }

    // From: c10/core/StorageImpl.h (allocator, resizable)
    // The allocator that made this memory, to allocate more when resize_
    // grows it (on the same device, pinned if it was pinned). Memory owned
    // by someone else (from_blob, a NumPy array) has none and cannot grow
    Allocator* allocator() const {
        return allocator_;
    }

    bool resizable() const {
        return resizable_;
    }

    // From: c10/core/StorageImpl.h (set_resizable). Turned off when the
    // memory is lent out (a NumPy array over it): growing would free it
    void set_resizable(bool resizable) {
        resizable_ = resizable && allocator_ != nullptr;
    }

    // Replaces the memory: every tensor over this storage sees the new one
    void set_data_ptr(DataPtr data, size_t nbytes) {
        data_ptr_ = std::move(data);
        nbytes_ = nbytes;
    }

    // From: c10/core/TensorImpl.h (VariableVersion)
    // Bumped by every in-place op that writes this memory. A backward node
    // records the version of each tensor it saves and refuses to run if it
    // moved: the values it needs were overwritten. PyTorch keeps the counter
    // in the TensorImpl and hands it to every view and detach() (so that
    // x.data can opt out); every tensor that shares the memory shares the
    // storage, so here it lives in the storage.
    uint32_t version() const {
        return version_.load(std::memory_order_relaxed);
    }

    void bump_version() {
        version_.fetch_add(1, std::memory_order_relaxed);
    }

private:
    DataPtr data_ptr_;
    size_t  nbytes_;
    Allocator* allocator_ = nullptr;
    bool resizable_ = false;
    std::atomic<uint32_t> version_{0};
};

} // namespace raxis::detail