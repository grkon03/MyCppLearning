#include <iostream>
#include "src/math/math.hpp"

using namespace MCL;

int main()
{
    math::Rmatrix m1({{1, 2, -2}, {3, 4, -1}}), m2({{-3, 1}, {2, 1}, {-1, 1}});

    std::cout << m1 << std::endl;
    std::cout << m2 << std::endl;
    std::cout << m1 * m2 << std::endl;
}