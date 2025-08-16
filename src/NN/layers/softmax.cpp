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

    SoftmaxLastLayer::SoftmaxLastLayer(size_t size) : size(size) {}

    size_t SoftmaxLastLayer::inputSize() const { return size; }
    size_t SoftmaxLastLayer::outputSize() const { return size; }

    math::Rmatrix SoftmaxLastLayer::forward(math::Rmatrix input)
    {
        return _prediction = softmax(input);
    }

    math::Rmatrix SoftmaxLastLayer::backwardWithCorrectAnswer(math::Rmatrix correctAnswer)
    {
        _loss = crossentropy(_prediction, correctAnswer);
        return _prediction - correctAnswer;
    }

    math::Rmatrix SoftmaxLastLayer::prediction() const { return _prediction; }
    math::Real SoftmaxLastLayer::loss() const { return _loss; }
}