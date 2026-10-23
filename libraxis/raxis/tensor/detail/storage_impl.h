#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <utility>

#include <raxis/core/device.h>
#include <raxis/util/data_ptr.h>


namespace raxis::detail {

/// Owns the raw data allocation for one or more tensors.
class StorageImpl {
public:
    StorageImpl(DataPtr data, size_t nbytes)
        : data_ptr_(std::move(data)),
          nbytes_(nbytes) {}

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
    std::atomic<uint32_t> version_{0};
};

} // namespace raxis::detail