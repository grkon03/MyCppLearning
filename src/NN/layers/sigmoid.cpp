#include <math.h>
#include "sigmoid.hpp"

namespace MCL::NN::Layers
{
    SigmoidLayer::SigmoidLayer(size_t size) : size(size) {}
    SigmoidLayer::SigmoidLayer(const SigmoidLayer &s) : size(s.size) {}

    size_t SigmoidLayer::inputSize() const
    {
        return size;
    }

    size_t SigmoidLayer::outputSize() const
    {
        return size;
    }

    math::Rmatrix SigmoidLayer::forward(const math::Rmatrix &input)
    {
        assert(input.noRows() == size);
        lastinput = input;
        return lastoutput = input.map<math::Real>([](math::Real x)
                                                  { return 1 / (1 + std::exp(-x)); });
    }

    math::Rmatrix SigmoidLayer::backward(const math::Rmatrix &gradOutput)
    {
        assert(gradOutput.noRows() == size);
        return gradOutput.map<math::Real>([&](math::Real x, size_t i)
                                          { return x * lastoutput.direct(i) * lastoutput.direct(i) * std::exp(-lastinput.direct(i)); });
    }

    std::unique_ptr<Layer> SigmoidLayer::copy() const
    {
        return std::unique_ptr<Layer>(new SigmoidLayer(*this));
    }

    std::vector<math::Rmatrix *> SigmoidLayer::getParameterRefs()
    {
        return std::vector<math::Rmatrix *>();
    }

    std::vector<math::Rmatrix> SigmoidLayer::getGradients() const
    {
        return std::vector<math::Rmatrix>();
    }
}