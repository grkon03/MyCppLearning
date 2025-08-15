#include "lastlayer.hpp"

namespace MCL::NN
{
    size_t LastLayer::outputSize() const
    {
        return 1;
    }

    std::vector<math::Rmatrix *> LastLayer::getParamaterRefs()
    {
        return std::vector<math::Rmatrix *>();
    }

    std::vector<math::Rmatrix> LastLayer::getGradients() const
    {
        return std::vector<math::Rmatrix>();
    }

    void LastLayer::setCorrectAnswer(math::Rmatrix correctAnswer)
    {
        this->correctAnswer = correctAnswer;
    }
}