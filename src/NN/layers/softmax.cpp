#include "softmax.hpp"

namespace MCL::NN
{
    namespace
    {
        math::Rmatrix softmax(const math::Rmatrix &input)
        {
            math::Real m = input.max();
            auto exped = (input - m).map<math::Real>([](math::Real x)
                                                     { return std::exp(x); });
            return exped / exped.sum();
        }
        math::Real crossentropy(const math::Rmatrix &input, const math::Rmatrix &correct)
        {
            math::Real delta = 1e-7;

            return correct.hadamardProd(
                              (input + delta).map<math::Real>([](math::Real x)
                                                              { return std::log(x); }))
                .sum();
        }
    }

    SoftmaxLastLayer::SoftmaxLastLayer(size_t inputsize) : inputsize(inputsize) {}

    size_t SoftmaxLastLayer::inputSize() const { return inputsize; }

    math::Rmatrix SoftmaxLastLayer::forward(math::Rmatrix input)
    {
        _prediction = softmax(input);
        return math::Rmatrix(_loss = crossentropy(input, correctAnswer));
    }

    math::Rmatrix SoftmaxLastLayer::backward(math::Rmatrix outputGrad)
    {
        assert(outputGrad.isVVector(1));

        return _prediction - correctAnswer;
    }

    math::Rmatrix SoftmaxLastLayer::prediction() const { return _prediction; }
    math::Real SoftmaxLastLayer::loss() const { return _loss; }
}