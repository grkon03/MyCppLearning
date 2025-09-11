#pragma once

#include "layer.hpp"

namespace MCL::NN
{
    class ConnectedLayers : public Layer
    {
    private:
        std::vector<std::unique_ptr<Layer>> line;

        size_t inputsize;
        size_t outputsize;

    public:
        ConnectedLayers();
        ConnectedLayers(const ConnectedLayers &);

        void addLayer(const Layer *);
        Layer *operator[](size_t);
        const Layer *operator[](size_t) const;
        size_t length() const;

        size_t inputSize() const override;
        size_t outputSize() const override;
        math::Rmatrix forward(const math::Rmatrix &) override;
        math::Rmatrix backward(const math::Rmatrix &) override;

        std::vector<math::Rmatrix *> getParameterRefs() override;
        std::vector<math::Rmatrix> getGradients() const override;

        std::unique_ptr<Layer> copy() const override;
    };
}