#include <math.h>
#include "sigmoid.hpp"

namespace MCL::NN
{
    SigmoidLayer::SigmoidLayer(size_t size) : size(size) {}

    size_t SigmoidLayer::inputSize() const
    {
        return size;
    }

    size_t SigmoidLayer::outputSize() const
    {
        return size;
    }

    math::Rmatrix SigmoidLayer::forward(math::Rmatrix input)
    {
        assert(input.isVVector(size));
        lastinput = input;
        return lastoutput = input.map<math::Real>([](math::Real x)
                                                  { return 1 / (1 + std::exp(-x)); });
    }

    math::Rmatrix SigmoidLayer::backward(math::Rmatrix gradOutput)
    {
        assert(gradOutput.isVVector(size));
        return gradOutput.map<math::Real>([&](math::Real x, size_t i)
                                          { return x * lastoutput.direct(i) * lastoutput.direct(i) * std::exp(-lastinput.direct(i)); });
    }

    SigmoidLayer *SigmoidLayer::copy() const
    {
        return new SigmoidLayer(size);
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