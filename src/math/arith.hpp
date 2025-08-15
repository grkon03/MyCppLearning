#pragma once

#include <utility>
#include <concepts>

namespace MCL::math
{
    using Integer = int;
    using Real = double;

    template <typename T>
    concept arith = requires(T a, T b, int n) {
        { a + b } -> std::common_with<T>;
        { a - b } -> std::common_with<T>;
        { a * b } -> std::common_with<T>;
        { a / b } -> std::common_with<T>;
        { a * n } -> std::common_with<T>;
        { n * a } -> std::common_with<T>;
        static_cast<T>(std::declval<int>());
    };
}