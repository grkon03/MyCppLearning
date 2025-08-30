#include "random.hpp"

#include <random>

namespace MCL::util
{
    namespace
    {
        std::mt19937 mt = std::mt19937(std::random_device()());
    }

    math::Rmatrix randomMatrixFromNormalDistribution(size_t rows, size_t cols, math::Real mean, math::Real stddev)
    {
        std::random_device rnd;
        std::normal_distribution nd(mean, stddev);

        math::Rmatrix ret(rows, cols);
        size_t i;

        for (i = 0; i < rows * cols; ++i)
        {
            ret.direct(i) = nd(mt);
        }

        return ret;
    }
}