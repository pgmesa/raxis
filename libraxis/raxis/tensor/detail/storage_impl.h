#pragma once

#include <cstddef>
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

private:
    DataPtr data_ptr_;
    size_t  nbytes_;
};

} // namespace raxis::detail