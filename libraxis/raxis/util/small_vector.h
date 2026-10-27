#pragma once

#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <algorithm>
#include <type_traits>
#include <utility>
#include <new>


namespace raxis {

// A growable array with inline storage for the first N elements.
// Behaves like std::vector but avoids heap allocation when size <= N.
// For trivially copyable T, uses memcpy for optimal codegen.
template <typename T, size_t N>
class SmallVector {
    static_assert(N > 0, "SmallVector requires at least 1 inline element.");

public:
    using value_type      = T;
    using size_type       = size_t;
    using iterator        = T*;
    using const_iterator  = const T*;

    SmallVector() = default;

    explicit SmallVector(size_t n, const T& value = T{}) {
        reserve(n);
        std::uninitialized_fill_n(data(), n, value);
        size_ = n;
    }

    SmallVector(std::initializer_list<T> il) {
        reserve(il.size());
        std::uninitialized_copy(il.begin(), il.end(), data());
        size_ = il.size();
    }

    template <typename InputIt,
        std::enable_if_t<
            std::is_convertible_v<
                typename std::iterator_traits<InputIt>::iterator_category,
                std::input_iterator_tag>,
        int> = 0>
    SmallVector(InputIt first, InputIt last) {
        auto count = static_cast<size_t>(std::distance(first, last));
        reserve(count);
        std::uninitialized_copy(first, last, data());
        size_ = count;
    }

    SmallVector(const SmallVector& other) {
        reserve(other.size_);
        std::uninitialized_copy_n(other.data(), other.size_, data());
        size_ = other.size_;
    }

    SmallVector(SmallVector&& other) noexcept(std::is_nothrow_move_constructible_v<T>) {
        move_from(std::move(other));
    }

    template <
        typename Container,
        std::enable_if_t<
            std::is_convertible_v<
                typename std::iterator_traits<
                    decltype(std::declval<Container>().begin())>::iterator_category,
                std::input_iterator_tag> &&
            std::is_convertible_v<
                typename std::iterator_traits<
                    decltype(std::declval<Container>().end())>::iterator_category,
                std::input_iterator_tag>,
        int> = 0>
    explicit SmallVector(Container&& c) {
        auto count = static_cast<size_t>(std::distance(c.begin(), c.end()));
        reserve(count);
        std::uninitialized_copy(c.begin(), c.end(), data());
        size_ = count;
    }

    SmallVector& operator=(const SmallVector& other) {
        if (this == &other) return *this;
        clear();
        reserve(other.size_);
        std::uninitialized_copy_n(other.data(), other.size_, data());
        size_ = other.size_;
        return *this;
    }

    SmallVector& operator=(SmallVector&& other) noexcept(std::is_nothrow_move_constructible_v<T>) {
        if (this == &other) return *this;
        destroy_all();
        free_heap();
        move_from(std::move(other));
        return *this;
    }

    ~SmallVector() {
        destroy_all();
        free_heap();
    }

    T& operator[](size_t i) noexcept {
        return data()[i];
    }

    const T& operator[](size_t i) const noexcept {
        return data()[i];
    }

    T& front() noexcept {
        return data()[0];
    }

    T& back() noexcept {
        return data()[size_ - 1];
    }

    const T& front() const noexcept {
        return data()[0];
    }

    const T& back() const noexcept {
        return data()[size_ - 1];
    }

    T* data() noexcept {
        return is_heap() ? heap_ : reinterpret_cast<T*>(inline_);
    }

    const T* data() const noexcept {
        return is_heap() ? heap_ : reinterpret_cast<const T*>(inline_);
    }

    iterator begin() noexcept {
        return data();
    }

    iterator end() noexcept {
        return data() + size_;
    }

    const_iterator begin() const noexcept {
        return data();
    }

    const_iterator end() const noexcept {
        return data() + size_;
    }

    std::reverse_iterator<iterator> rbegin() noexcept {
        return std::reverse_iterator<iterator>(end());
    }
    
    std::reverse_iterator<iterator> rend() noexcept {
        return std::reverse_iterator<iterator>(begin());
    }

    size_t size() const noexcept {
        return size_;
    }

    size_t capacity() const noexcept {
        return capacity_;
    }

    bool empty() const noexcept {
        return size_ == 0;
    }

    void reserve(size_t new_cap) {
        if (new_cap <= capacity_) return;
        grow(new_cap);
    }

    void resize(size_t n, const T& value = T{}) {
        if (n > capacity_) grow(n);
        if (n > size_) {
            std::uninitialized_fill(data() + size_, data() + n, value);
        } else if (n < size_) {
            std::destroy(data() + n, data() + size_);
        }
        size_ = n;
    }

    void clear() noexcept {
        destroy_all();
        size_ = 0;
    }

    void push_back(const T& v) {
        if (size_ == capacity_) grow(capacity_ * 2);
        ::new (static_cast<void*>(data() + size_)) T(v);
        ++size_;
    }

    void push_back(T&& v) {
        if (size_ == capacity_) grow(capacity_ * 2);
        ::new (static_cast<void*>(data() + size_)) T(std::move(v));
        ++size_;
    }

    template <typename... Args>
    T& emplace_back(Args&&... args) {
        if (size_ == capacity_) grow(capacity_ * 2);
        T* p = ::new (static_cast<void*>(data() + size_)) T(std::forward<Args>(args)...);
        ++size_;
        return *p;
    }

    iterator insert(const_iterator pos, const T& value) {
        size_t idx = pos - data();
        if (size_ == capacity_) {
            grow(capacity_ * 2);
        }
        T* p = data() + idx;
        if (idx < size_) {
            // Shift elements right by one
            if constexpr (std::is_trivially_copyable_v<T>) {
                std::memmove(p + 1, p, (size_ - idx) * sizeof(T));
            } else {
                ::new (static_cast<void*>(data() + size_)) T(std::move(data()[size_ - 1]));
                std::move_backward(p, data() + size_ - 1, data() + size_);
                *p = value;
                ++size_;
                return p;
            }
        }
        ::new (static_cast<void*>(p)) T(value);
        ++size_;
        return p;
    }

    iterator insert(const_iterator pos, T&& value) {
        size_t idx = pos - data();
        if (size_ == capacity_) {
            grow(capacity_ * 2);
        }
        T* p = data() + idx;
        if (idx < size_) {
            if constexpr (std::is_trivially_copyable_v<T>) {
                std::memmove(p + 1, p, (size_ - idx) * sizeof(T));
            } else {
                ::new (static_cast<void*>(data() + size_)) T(std::move(data()[size_ - 1]));
                std::move_backward(p, data() + size_ - 1, data() + size_);
                *p = std::move(value);
                ++size_;
                return p;
            }
        }
        ::new (static_cast<void*>(p)) T(std::move(value));
        ++size_;
        return p;
    }

    iterator erase(const_iterator pos) {
        T* p = data() + (pos - data());
        std::move(p + 1, data() + size_, p);
        --size_;
        data()[size_].~T();
        return p;
    }

    bool operator==(const SmallVector& other) const {
        return size_ == other.size_ && std::equal(begin(), end(), other.begin());
    }

    bool operator!=(const SmallVector& other) const {
        return !(*this == other);
    }

private:
    bool is_heap() const noexcept {
        return capacity_ > N;
    }

    void destroy_all() noexcept {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            std::destroy_n(data(), size_);
        }
    }

    void free_heap() noexcept {
        if (is_heap()) {
            ::operator delete(heap_);
            heap_ = nullptr;
            capacity_ = N;
        }
    }

    void grow(size_t new_cap) {
        new_cap = std::max(new_cap, capacity_ * 2);  // amortized growth
        new_cap = std::max(new_cap, N + 1);           // always grow past inline

        T* new_data = static_cast<T*>(::operator new(new_cap * sizeof(T)));

        if (size_ > 0) {
            if constexpr (std::is_trivially_copyable_v<T>) {
                std::memcpy(new_data, data(), size_ * sizeof(T));
            } else {
                std::uninitialized_move_n(data(), size_, new_data);
                destroy_all();
            }
        }

        free_heap();
        heap_ = new_data;
        capacity_ = new_cap;
    }

    void move_from(SmallVector&& other) noexcept(std::is_nothrow_move_constructible_v<T>) {
        if (other.is_heap()) {
            heap_ = other.heap_;
            capacity_ = other.capacity_;
            other.heap_ = nullptr;
            other.capacity_ = N;
        } else {
            if constexpr (std::is_trivially_copyable_v<T>) {
                std::memcpy(inline_, other.inline_, other.size_ * sizeof(T));
            } else {
                std::uninitialized_move_n(other.data(), other.size_, data());
                std::destroy_n(other.data(), other.size_);
            }
            capacity_ = N;
        }
        size_ = other.size_;
        other.size_ = 0;
    }

    size_t size_ = 0;
    size_t capacity_ = N;
    union {
        alignas(T) char inline_[N * sizeof(T)];
        T* heap_ = nullptr;
    };
};

} // namespace raxis