#include "connected.hpp"

namespace MCL::NN
{
    ConnectedLayers::ConnectedLayers() {}
    ConnectedLayers::ConnectedLayers(const ConnectedLayers &ll) : line(ll.line.size()), inputsize(ll.inputsize), outputsize(ll.outputsize)
    {
        size_t i;
        for (i = 0; i < ll.line.size(); ++i)
        {
            line[i] = std::unique_ptr<Layer>(ll.line[i]->copy().release());
        }
    }

    void ConnectedLayers::addLayer(const Layer *layer)
    {
        line.push_back(std::unique_ptr<Layer>(layer->copy().release()));
    }

    Layer *ConnectedLayers::operator[](size_t i) { return line[i].get(); }
    const Layer *ConnectedLayers::operator[](size_t i) const { return line[i].get(); }

    size_t ConnectedLayers::length() const { return line.size(); }

    size_t ConnectedLayers::inputSize() const { return inputsize; }
    size_t ConnectedLayers::outputSize() const { return outputsize; }

    math::Rmatrix ConnectedLayers::forward(const math::Rmatrix &input)
    {
        size_t i;
        math::Rmatrix output = input;
        for (i = 0; i < line.size(); ++i)
        {
            output = line[i]->forward(output);
        }

        return output;
    }

    math::Rmatrix ConnectedLayers::backward(const math::Rmatrix &gradOutput)
    {
        size_t i, lastindex = line.size() - 1;
        math::Rmatrix grad = gradOutput;
        for (i = 0; i < line.size(); ++i)
        {
            grad = line[lastindex - i]->backward(grad);
        }

        return grad;
    }

    std::vector<math::Rmatrix *> ConnectedLayers::getParameterRefs()
    {
        std::vector<math::Rmatrix *> allparams;

        size_t i, j;
        for (i = 0; i < line.size(); ++i)
        {
            auto params = line[i]->getParameterRefs();
            for (j = 0; j < params.size(); ++j)
            {
                allparams.push_back(params[j]);
            }
        }

        return allparams;
    }

    std::vector<math::Rmatrix> ConnectedLayers::getGradients() const
    {
        std::vector<math::Rmatrix> allgrads;

        size_t i, j;
        for (i = 0; i < line.size(); ++i)
        {
            auto grads = line[i]->getGradients();
            for (j = 0; j < grads.size(); ++j)
            {
                allgrads.push_back(grads[j]);
            }
        }

        return allgrads;
    }

    std::unique_ptr<Layer> ConnectedLayers::copy() const
    {
        return std::unique_ptr<Layer>(new ConnectedLayers(*this));
    }
}