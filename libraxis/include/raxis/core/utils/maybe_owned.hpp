#pragma once

#include <optional>


namespace raxis::detail {

template<typename T>
class MaybeOwned {
public:

    MaybeOwned(MaybeOwned&& other) noexcept
        : storage_(std::move(other.storage_)), owned_(other.owned_) {
        if (owned_) {
            ptr_ = storage_.has_value() ? &storage_.value() : nullptr;
        } else {
            ptr_ = other.ptr_;
        }
        other.ptr_ = nullptr;
    }
    
    MaybeOwned& operator=(MaybeOwned&& other) noexcept {
        if (this != &other) {
            storage_ = std::move(other.storage_);
            owned_ = other.owned_;
            
            if (owned_) {
                ptr_ = storage_.has_value() ? &storage_.value() : nullptr;
            } else {
                ptr_ = other.ptr_;
            }
            other.ptr_ = nullptr;
        }
        return *this;
    }
    
    MaybeOwned() : ptr_(nullptr), owned_(false) {}
    ~MaybeOwned() = default;

    // Borrowed - doesn't increment ref count
    static MaybeOwned borrowed(const T& ref) {
        MaybeOwned mo;
        mo.ptr_ = &ref;
        mo.owned_ = false;
        return mo;
    }
    
   // Owned from lvalue - copy
    static MaybeOwned owned(const T& value) {
        MaybeOwned mo;
        mo.storage_ = value;
        mo.ptr_ = &mo.storage_.value();
        mo.owned_ = true;
        return mo;
    }
    
    // Owned from rvalue - move
    static MaybeOwned owned(T&& value) {
        MaybeOwned mo;
        mo.storage_ = std::move(value);
        mo.ptr_ = &mo.storage_.value();
        mo.owned_ = true;
        return mo;
    }
    
    const T& operator*() const { return *ptr_; }
    const T* operator->() const { return ptr_; }

    bool is_borrowed() const { return !owned_; }
    bool is_owned() const { return owned_; }

private:
    const T* ptr_ = nullptr;
    std::optional<T> storage_;  // Only used if owned
    bool owned_ = false;
};
    
} // namespace raxis

