#pragma once

#include "layer.hpp"

namespace MCL::NN
{
    class LastLayer : public Layer
    {
    public:
        // backward

        math::Rmatrix backward(math::Rmatrix) override final; // don't use pure backward method
        virtual math::Rmatrix backwardWithCorrectAnswer(math::Rmatrix) = 0;

        // fixed basic methods

        std::vector<math::Rmatrix *> getParameterRefs() override final; // last layers must not have parameters
        std::vector<math::Rmatrix> getGradients() const override final; // last layers must not have parameters

        // last layer methods

        virtual math::Rmatrix prediction() const = 0;
        virtual math::Real loss() const = 0;
    };
}