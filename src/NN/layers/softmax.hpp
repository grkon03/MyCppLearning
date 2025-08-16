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
        size_t size;

        // data

        math::Rmatrix _prediction;
        math::Real _loss;

    public:
        // constructors

        SoftmaxLastLayer(size_t size);

        // basic methods

        size_t inputSize() const override;
        size_t outputSize() const override;
        math::Rmatrix forward(math::Rmatrix) override;
        math::Rmatrix backwardWithCorrectAnswer(math::Rmatrix) override;

        // last layer methods

        math::Rmatrix prediction() const override;
        math::Real loss() const override;
    };
}