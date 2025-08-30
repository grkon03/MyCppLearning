#pragma once

#include "layer.hpp"

namespace MCL::NN::Layers
{

    class AffineLayer : public Layer
    {
    public:
        enum class WeightInitType
        {
            Zero,
            He,
            Xavier,
        };

        enum class BiasInitType
        {
            Zero,
            SmallPositive,
        };

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
        AffineLayer(size_t inputsize, size_t outputsize, WeightInitType wit, BiasInitType bit);
        AffineLayer(math::Rmatrix _weight, math::Rmatrix _bias);
        AffineLayer(const AffineLayer &);

        // basic methods

        size_t inputSize() const override;
        size_t outputSize() const override;
        math::Rmatrix forward(const math::Rmatrix &) override;
        math::Rmatrix backward(const math::Rmatrix &) override;

        virtual std::unique_ptr<Layer> copy() const override;

        std::vector<math::Rmatrix *> getParameterRefs() override;
        std::vector<math::Rmatrix> getGradients() const override;
    };
}