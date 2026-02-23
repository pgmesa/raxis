#pragma once

#include <memory>

// ===============================================
// Adapted from: pytorch/c10/util/UniqueVoidPtr.h
// ===============================================

namespace raxis {

using DeleterFnPtr = void (*)(void*);
inline void delete_nothing(void*) noexcept {}

class UniqueVoidPtr {
private:
    // Lifetime tied to ctx_
    void* data_;
    std::unique_ptr<void, DeleterFnPtr> ctx_;

public:
    UniqueVoidPtr() : data_(nullptr), ctx_(nullptr, &delete_nothing) {}
    UniqueVoidPtr(void* data) : data_(data), ctx_(nullptr, &delete_nothing) {}
    UniqueVoidPtr(void* data, void* ctx, DeleterFnPtr ctx_deleter)
        : data_(data), ctx_(ctx, ctx_deleter? ctx_deleter : &delete_nothing) {}
    
    void* operator->() const {
        return data_;
    }

    void* get() const {
        return data_;
    }

    void clear() {
        ctx_ = nullptr;
        data_ = nullptr;
    }

    void* get_context() const {
        return ctx_.get();
    }

    void* release_context() {
        return ctx_.release();
    }

    operator bool() const {
        return data_ || ctx_;
    }

    DeleterFnPtr get_deleter() const {
        return ctx_.get_deleter();
    }
};
    
}

    