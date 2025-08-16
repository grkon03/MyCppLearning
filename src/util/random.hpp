#pragma once

#include "../math/math.hpp"

namespace MCL::util
{
    math::Rmatrix randomMatrixFromNormalDistribution(size_t rows, size_t cols, math::Real mean, math::Real stddev);
}