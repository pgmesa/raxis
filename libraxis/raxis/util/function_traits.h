#pragma once

#include <tuple>
#include <cstddef>

// Function traits: extracts compile-time information from callable types
// Extracted from PyTorch's implementation (aten/src/ATen/detail/FunctionTraits.h)
// 
// Usage:
//   auto lambda = [](int x, float y) -> double { return x + y; };
//   using traits = function_traits<decltype(lambda)>;
//   
//   traits::arity           // 2
//   traits::result_type     // double
//   traits::arg<0>::type    // int
//   traits::arg<1>::type    // float

// Fallback: for any callable with operator() (lambdas, functors, etc.)
// Redirects to the appropriate specialization by extracting the operator() signature

namespace raxis {
    
template <typename T>
struct function_traits : public function_traits<decltype(&T::operator())> {};

template <typename ClassType, typename T>
struct function_traits<T ClassType::*> : public function_traits<T> {};

// Const class member functions
template <typename ClassType, typename ReturnType, typename... Args>
struct function_traits<ReturnType(ClassType::*)(Args...) const> : public function_traits<ReturnType(Args...)> {};

// Reference types
template <typename T>
struct function_traits<T&> : public function_traits<T> {};
template <typename T>
struct function_traits<T*> : public function_traits<T> {};

// Base case: free function signature
// This is where we actually extract the information
template <typename ReturnType, typename... Args>
struct function_traits<ReturnType(Args...)> {
    // Number of arguments
    static constexpr size_t arity = sizeof...(Args);
    
    using ArgsTuple = std::tuple<Args...>;
    using result_type = ReturnType;

    // Type of the i-th argument
    template <size_t i>
    struct arg {
        using type = typename std::tuple_element<i, ArgsTuple>::type;
    };
};

}
