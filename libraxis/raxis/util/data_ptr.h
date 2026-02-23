#pragma once

#include <raxis/core/device.h>
#include <raxis/util/unique_void_ptr.h>


namespace raxis {

class DataPtr {
private:
    UniqueVoidPtr ptr_;
    Device device_;

public:
    DataPtr() : device_(DeviceType::CPU) {}
    DataPtr(void* data, Device device) : ptr_(data), device_(device) {}
    DataPtr(void* data, void* ctx, DeleterFnPtr ctx_deleter, Device device)
        : ptr_(data, ctx, ctx_deleter), device_(device) {}

    void* operator->() const {
        return ptr_.get();
    }

    void clear() {
        ptr_.clear();
    }

    void* get() const {
        return ptr_.get();
    }

    void* mutable_get() {
        return ptr_.get();
    }

    void* get_context() const {
        return ptr_.get_context();
    }
    
    void* release_context() {
        return ptr_.release_context();
    }

    operator bool() const {
        return static_cast<bool>(ptr_);
    }

    DeleterFnPtr get_deleter() const {
        return ptr_.get_deleter();
    }

    Device device() const {
        return device_;
    }

};

}