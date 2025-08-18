#pragma once

#include "layer.hpp"

namespace MCL::NN
{
    class ReLULayer : public Layer
    {
    private:
        size_t size;

        // data

        math::Rmatrix lastinput;

    public:
        // constructors

        ReLULayer(size_t size);

        // basic methods

        size_t inputSize() const override;
        size_t outputSize() const override;
        math::Rmatrix forward(math::Rmatrix) override;
        math::Rmatrix backward(math::Rmatrix) override;

        std::vector<math::Rmatrix *> getParameterRefs() override;
        std::vector<math::Rmatrix> getGradients() const override;

        ReLULayer *copy() const override;
    };
}