#pragma once

#include "layer.hpp"

namespace MCL::NN
{
    class LastLayer : public Layer
    {
    protected:
        math::Rmatrix correctAnswer;

    public:
        // fixed basic methods

        size_t outputSize() const override final;                       // returns 1
        std::vector<math::Rmatrix *> getParamaterRefs() override final; // last layers must not have paramaters
        std::vector<math::Rmatrix> getGradients() const override final; // last layers must not have paramaters

        // last layer methods

        virtual void setCorrectAnswer(math::Rmatrix) final;
        virtual math::Rmatrix prediction() const = 0;
        virtual math::Real loss() const = 0;
    };
}