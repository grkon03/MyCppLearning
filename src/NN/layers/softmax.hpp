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
        SoftmaxLastLayer(const SoftmaxLastLayer &);

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