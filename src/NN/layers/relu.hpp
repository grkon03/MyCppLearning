#pragma once

#include "layer.hpp"

namespace MCL::NN::Layers
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
        ReLULayer(const ReLULayer &);

        // basic methods

        size_t inputSize() const override;
        size_t outputSize() const override;
        math::Rmatrix forward(const math::Rmatrix &) override;
        math::Rmatrix backward(const math::Rmatrix &) override;

        std::vector<math::Rmatrix *> getParameterRefs() override;
        std::vector<math::Rmatrix> getGradients() const override;

        std::unique_ptr<Layer> copy() const override;
    };
}