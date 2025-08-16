#pragma once

#include "layer.hpp"

namespace MCL::NN
{
    class AffineLayer : public Layer
    {
    private:
        size_t inputsize;
        size_t outputsize;
        math::Rmatrix weight;
        math::Rmatrix bias; // should be a vector

        // data

        math::Rmatrix lastinput;
        math::Rmatrix gradWeight; // same shape as weight
        math::Rmatrix gradBias;   // same shape as bias

    public:
        // constructors

        AffineLayer(size_t inputsize, size_t outputsize);
        AffineLayer(math::Rmatrix _weight, math::Rmatrix _bias);

        // basic methods

        size_t inputSize() const override;
        size_t outputSize() const override;
        math::Rmatrix forward(math::Rmatrix) override;
        math::Rmatrix backward(math::Rmatrix) override;

        std::vector<math::Rmatrix *> getParameterRefs() override;
        std::vector<math::Rmatrix> getGradients() const override;
    };
}