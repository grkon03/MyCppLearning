#include "mse.hpp"

namespace MCL::NN
{
    namespace
    {
        math::Real __square(math::Real x)
        {
            return x * x;
        }
    }

    MSELastLayer::MSELastLayer(size_t size) : size(size) {}
    MSELastLayer::MSELastLayer(const MSELastLayer &m) : size(m.size) {}

    size_t MSELastLayer::inputSize() const { return size; }
    size_t MSELastLayer::outputSize() const { return size; }

    math::Rmatrix MSELastLayer::forward(math::Rmatrix input)
    {
        return (_prediction = input);
    }

    math::Rmatrix MSELastLayer::backwardByComparing(math::Rmatrix compared)
    {
        auto dif = _prediction - compared;
        _loss = (dif / 2).map<math::Real>(__square).average();

        return dif;
    }

    std::unique_ptr<LastLayer> MSELastLayer::copy() const
    {
        return std::unique_ptr<LastLayer>(new MSELastLayer(*this));
    }

    math::Rmatrix MSELastLayer::prediction() const { return _prediction; }
    math::Real MSELastLayer::loss() const { return _loss; }
}