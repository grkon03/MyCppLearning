#pragma once

#include <vector>
#include <memory>
#include "../../math/math.hpp"

namespace MCL::NN
{
    class Layer
    {
    public:
        virtual ~Layer() = default;

        virtual size_t inputSize() const = 0;
        virtual size_t outputSize() const = 0;
        virtual math::Rmatrix forward(math::Rmatrix) = 0;  // data should be vertical vectors
        virtual math::Rmatrix backward(math::Rmatrix) = 0; // data should be vertical vectors

        virtual std::unique_ptr<Layer> copy() const = 0;

        // learn

        virtual std::vector<math::Rmatrix *> getParameterRefs() = 0;
        virtual std::vector<math::Rmatrix> getGradients() const = 0;
    };
}