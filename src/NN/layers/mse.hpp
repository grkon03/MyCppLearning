#pragma once

#include "lastlayer.hpp"

namespace MCL::NN::Layers
{
    class MSELastLayer : public LastLayer
    {
    private:
        size_t size;

        // data

        math::Rmatrix _prediction;
        math::Real _loss;

    public:
        // constructors

        MSELastLayer(size_t size);
        MSELastLayer(const MSELastLayer &);

        // basic methods

        size_t inputSize() const override;
        size_t outputSize() const override;
        math::Rmatrix forward(const math::Rmatrix &) override;
        math::Rmatrix backwardByComparing(const math::Rmatrix &compared) override;

        std::unique_ptr<LastLayer> copy() const override;

        // last layer methods

        math::Rmatrix prediction() const override;
        math::Real loss() const override;
    };
}