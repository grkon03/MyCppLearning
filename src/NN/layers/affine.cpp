#include "affine.hpp"

namespace MCL::NN::Layers
{
    AffineLayer::AffineLayer(size_t inputsize, size_t outputsize)
        : inputsize(inputsize), outputsize(outputsize) {}

    AffineLayer::AffineLayer(math::Rmatrix _weight, math::Rmatrix _bias)
        : inputsize(_weight.noColumns()), outputsize(_weight.noRows()), weight(_weight), bias(_bias)
    {
        assert(_bias.isVVector(outputsize));
    }

    AffineLayer::AffineLayer(const AffineLayer &a) : AffineLayer(a.weight, a.bias) {}

    size_t AffineLayer::inputSize() const
    {
        return inputsize;
    }

    size_t AffineLayer::outputSize() const
    {
        return outputsize;
    }

    math::Rmatrix AffineLayer::forward(const math::Rmatrix &input)
    {
        assert(input.noRows() == inputsize);

        lastinput = input;
        return (weight * input + bias);
    }

    math::Rmatrix AffineLayer::backward(const math::Rmatrix &gradOutput)
    {
        assert(gradOutput.noRows() == outputsize);

        gradWeight = gradOutput * lastinput.transpose();
        gradBias = gradOutput.rowwiseSum();

        return weight.transpose() * gradOutput;
    }

    std::unique_ptr<Layer> AffineLayer::copy() const
    {
        return std::unique_ptr<Layer>(new AffineLayer(*this));
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