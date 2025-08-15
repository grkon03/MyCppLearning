#pragma once

#include "lastlayer.hpp"

namespace MCL::NN
{
    /**
     * @brief softmax with the cross entropy loss
     *
     */
    class SoftmaxLastLayer : public LastLayer
    {
    private:
        size_t inputsize;

        // data

        math::Rmatrix _prediction;
        math::Real _loss;

    public:
        // constructors

        SoftmaxLastLayer(size_t inutsize);

        // basic methods

        size_t inputSize() const override;
        math::Rmatrix forward(math::Rmatrix) override;
        math::Rmatrix backward(math::Rmatrix) override;

        // last layer methods

        math::Rmatrix prediction() const override;
        math::Real loss() const override;
    };
}