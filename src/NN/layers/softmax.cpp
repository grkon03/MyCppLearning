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
    SoftmaxLastLayer::SoftmaxLastLayer(const SoftmaxLastLayer &s) : size(s.size) {}

    size_t SoftmaxLastLayer::inputSize() const { return size; }
    size_t SoftmaxLastLayer::outputSize() const { return size; }

    math::Rmatrix SoftmaxLastLayer::forward(const math::Rmatrix &input)
    {
        assert(input.noRows() == size);
        return _prediction = softmax(input);
    }

    math::Rmatrix SoftmaxLastLayer::backwardByComparing(const math::Rmatrix &compared)
    {
        assert(compared.noRows() == size);
        _loss = crossentropy(_prediction, compared);
        return _prediction - compared;
    }

    std::unique_ptr<LastLayer> SoftmaxLastLayer::copy() const
    {
        return std::unique_ptr<LastLayer>(new SoftmaxLastLayer(*this));
    }

    math::Rmatrix SoftmaxLastLayer::prediction() const { return _prediction; }
    math::Real SoftmaxLastLayer::loss() const { return _loss; }
}