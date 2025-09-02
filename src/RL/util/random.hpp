#pragma once

#include <vector>
#include <random>

#include "../../math/math.hpp"

namespace MCL::RL::util
{
    std::vector<math::Real> uniformDirichletSample(math::Real uniformAlpha, size_t size, std::mt19937 &rndgen);
}