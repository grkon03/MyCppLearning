#pragma once

#include "layer.hpp"

namespace MCL::NN
{
    class SigmoidLayer : public Layer
    {
    private:
        size_t size;

        // data

        math::Rmatrix lastinput;
        math::Rmatrix lastoutput;

    public:
        // constructors

        SigmoidLayer(size_t size);

        // basic methods

        size_t inputSize() const override;
        size_t outputSize() const override;
        math::Rmatrix forward(math::Rmatrix) override;
        math::Rmatrix backward(math::Rmatrix) override;

        std::vector<math::Rmatrix *> getParamaterRefs() override;
        std::vector<math::Rmatrix> getGradients() const override;
    };
}