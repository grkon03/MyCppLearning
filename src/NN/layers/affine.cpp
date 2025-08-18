#include "affine.hpp"

namespace MCL::NN
{
    AffineLayer::AffineLayer(size_t inputsize, size_t outputsize)
        : inputsize(inputsize), outputsize(outputsize) {}

    AffineLayer::AffineLayer(math::Rmatrix _weight, math::Rmatrix _bias)
        : inputsize(_weight.noColumns()), outputsize(_weight.noRows()), weight(_weight), bias(_bias)
    {
        assert(_bias.isVVector(outputsize));
    }

    size_t AffineLayer::inputSize() const
    {
        return inputsize;
    }

    size_t AffineLayer::outputSize() const
    {
        return outputsize;
    }

    math::Rmatrix AffineLayer::forward(math::Rmatrix input)
    {
        assert(input.isVVector(inputsize));

        lastinput = input;
        return (weight * input + bias);
    }

    math::Rmatrix AffineLayer::backward(math::Rmatrix gradOutput)
    {
        assert(gradOutput.isVVector(outputsize));
        gradWeight = gradOutput * lastinput.transpose();
        gradBias = gradOutput;

        return weight.transpose() * gradOutput;
    }

    AffineLayer *AffineLayer::copy() const
    {
        return new AffineLayer(weight, bias);
    }

    std::vector<math::Rmatrix *> AffineLayer::getParameterRefs()
    {
        std::vector<math::Rmatrix *> refs;

        refs.push_back(&weight);
        refs.push_back(&bias);

        return refs;
    }

    std::vector<math::Rmatrix> AffineLayer::getGradients() const
    {
        std::vector<math::Rmatrix> grads;

        grads.push_back(gradWeight);
        grads.push_back(gradBias);

        return grads;
    }

}