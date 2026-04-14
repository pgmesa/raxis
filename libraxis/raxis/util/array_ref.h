#pragma once

#include <cstddef>
#include <vector>
#include <algorithm>
#include <type_traits>

namespace raxis {

template <typename T>
class ArrayRef {
public:
    using iterator       = const T*;
    using const_iterator = const T*;
    using value_type     = T;
    using size_type      = size_t;

    constexpr ArrayRef() noexcept : data_(nullptr), size_(0) {}

    constexpr ArrayRef(const T* data, size_t size) noexcept
        : data_(data), size_(size) {}

    constexpr ArrayRef(const T* begin, const T* end) noexcept
        : data_(begin), size_(static_cast<size_t>(end - begin)) {}

    constexpr explicit ArrayRef(const T& one) noexcept
        : data_(&one), size_(1) {}

    constexpr ArrayRef(std::initializer_list<T> il) noexcept
        : data_(std::begin(il)), size_(il.size()) {}

    template <
      typename Container,
      typename U = decltype(std::declval<Container>().data()),
      typename = std::enable_if_t<
          (std::is_same_v<U, T*> || std::is_same_v<U, T const*>)>>
    ArrayRef(const Container& c) noexcept
        : data_(c.data()), size_(c.size()) {}

    template <size_t N>
    constexpr ArrayRef(const T (&arr)[N]) noexcept
        : data_(arr), size_(N) {}

    constexpr const T* data() const noexcept {
        return data_;
    }

    constexpr size_t size() const noexcept {
        return size_;
    }

    constexpr bool empty() const noexcept {
        return size_ == 0;
    }

    constexpr const T& operator[](size_t i) const noexcept {
        return data_[i];
    }

    constexpr const T& front() const noexcept {
        return data_[0];
    }

    constexpr const T& back() const noexcept {
        return data_[size_ - 1];
    }

    constexpr iterator begin() const noexcept {
        return data_;
    }

    constexpr iterator end() const noexcept {
        return data_ + size_;
    }

    constexpr bool equals(ArrayRef other) const noexcept {
        return size_ == other.size_ && std::equal(begin(), end(), other.begin());
    }

    constexpr bool operator==(ArrayRef other) const noexcept {
        return equals(other);
    }

    constexpr bool operator!=(ArrayRef other) const noexcept {
        return !equals(other);
    }

    std::vector<T> vec() const {
        return {data_, data_ + size_};
    }

private:
    const T* data_;
    size_t size_;
};

} // namespace raxis