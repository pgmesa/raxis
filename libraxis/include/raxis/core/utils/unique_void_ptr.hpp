#pragma once

#include <memory>

#include <raxis/api.hpp>

// ===============================================
// Adapted from: pytorch/c10/util/UniqueVoidPtr.h
// ===============================================

namespace raxis::detail {

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
    };
    
}

    