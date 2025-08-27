#pragma once

#include "layer.hpp"

namespace MCL::NN
{
    class LastLayer
    {
    public:
        virtual ~LastLayer() = default;

        virtual math::Rmatrix forward(math::Rmatrix) = 0; // data should be vertical vectors
        /**
         * @brief "compared" is a target value to be estimated:
         * e.g.) for supervised learning, "compared" should be the correct answers
         *
         * @return math::Rmatrix
         */
        virtual math::Rmatrix backwardByComparing(math::Rmatrix compared) = 0;

        virtual size_t inputSize() const = 0;
        virtual size_t outputSize() const = 0;

        virtual math::Rmatrix prediction() const = 0;
        virtual math::Real loss() const = 0;

        virtual std::unique_ptr<LastLayer> copy() const = 0;
    };
}