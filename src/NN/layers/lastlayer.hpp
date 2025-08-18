#pragma once

#include "layer.hpp"

namespace MCL::NN
{
    class LastLayer : public Layer
    {
    public:
        virtual LastLayer *copy() const override = 0;

        // backward

        math::Rmatrix backward(math::Rmatrix) override final; // don't use pure backward method
        /**
         * @brief "compared" is a target value to be estimated:
         * e.g.) for supervised learning, "compared" should be the correct answers
         *
         * @return math::Rmatrix
         */
        virtual math::Rmatrix backwardByComparing(math::Rmatrix compared) = 0;

        // fixed basic methods

        std::vector<math::Rmatrix *> getParameterRefs() override final; // last layers must not have parameters
        std::vector<math::Rmatrix> getGradients() const override final; // last layers must not have parameters

        // last layer methods

        virtual math::Rmatrix prediction() const = 0;
        virtual math::Real loss() const = 0;
    };
}