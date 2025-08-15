#include "relu.hpp"

namespace MCL::NN
{
    namespace
    {
        math::Real reluR(math::Real x)
        {
            return x > 0 ? x : 0;
        }
    }

    ReLULayer::ReLULayer(size_t size) : size(size) {}

    size_t ReLULayer::inputSize() const { return size; }
    size_t ReLULayer::outputSize() const { return size; }

    math::Rmatrix ReLULayer::forward(math::Rmatrix input)
    {
        assert(input.isVVector(size));

        lastinput = input;

        return input.map<math::Real>(reluR);
    }

    math::Rmatrix ReLULayer::backward(math::Rmatrix gradOutput)
    {
        assert(gradOutput.isVVector(size));

        auto fn = [&](math::Real x, size_t i) -> math::Real
        {
            return lastinput.direct(i) > 0 ? x : 0;
        };

        return gradOutput.map<math::Real>(fn);
    }

    std::vector<math::Rmatrix *> ReLULayer::getParamaterRefs()
    {
        return std::vector<math::Rmatrix *>();
    }

    std::vector<math::Rmatrix> ReLULayer::getGradients() const
    {
        return std::vector<math::Rmatrix>();
    }
}