#include "relu.hpp"

namespace MCL::NN::Layers
{
    namespace
    {
        math::Real reluR(math::Real x)
        {
            return x > 0 ? x : 0;
        }
    }

    ReLULayer::ReLULayer(size_t size) : size(size) {}
    ReLULayer::ReLULayer(const ReLULayer &r) : size(r.size) {}

    size_t ReLULayer::inputSize() const { return size; }
    size_t ReLULayer::outputSize() const { return size; }

    math::Rmatrix ReLULayer::forward(const math::Rmatrix &input)
    {
        assert(input.noRows() == size);

        lastinput = input;

        return input.map<math::Real>(reluR);
    }

    math::Rmatrix ReLULayer::backward(const math::Rmatrix &gradOutput)
    {
        assert(gradOutput.noRows() == size);

        auto fn = [&](math::Real x, size_t i) -> math::Real
        {
            return lastinput.direct(i) > 0 ? x : 0;
        };

        return gradOutput.map<math::Real>(fn);
    }

    std::vector<math::Rmatrix *> ReLULayer::getParameterRefs()
    {
        return std::vector<math::Rmatrix *>();
    }

    std::vector<math::Rmatrix> ReLULayer::getGradients() const
    {
        return std::vector<math::Rmatrix>();
    }

    std::unique_ptr<Layer> ReLULayer::copy() const
    {
        return std::unique_ptr<Layer>(new ReLULayer(*this));
    }
}