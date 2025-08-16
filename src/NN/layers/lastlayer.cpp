#include "lastlayer.hpp"

namespace MCL::NN
{
    math::Rmatrix LastLayer::backward(math::Rmatrix)
    {
        assert(false); // don't use any pure backward methods of LastLayer
    }

    std::vector<math::Rmatrix *> LastLayer::getParameterRefs()
    {
        return std::vector<math::Rmatrix *>();
    }

    std::vector<math::Rmatrix> LastLayer::getGradients() const
    {
        return std::vector<math::Rmatrix>();
    }
}