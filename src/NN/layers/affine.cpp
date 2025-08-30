#include "affine.hpp"
#include "../../util/random.hpp"

namespace MCL::NN::Layers
{
    AffineLayer::AffineLayer(size_t inputsize, size_t outputsize)
        : inputsize(inputsize), outputsize(outputsize) {}

    AffineLayer::AffineLayer(size_t inputsize, size_t outputsize, WeightInitType wit, BiasInitType bit)
        : inputsize(inputsize), outputsize(outputsize)
    {
        using WIT = WeightInitType;
        switch (wit)
        {
        case WIT::He:
            weight = util::randomMatrixFromNormalDistribution(outputsize, inputsize, 0, std::sqrt((double)2 / inputsize));
            break;
        case WIT::Xavier:
            weight = util::randomMatrixFromNormalDistribution(outputsize, inputsize, 0, std::sqrt((double)1 / inputsize));
            break;
        case WIT::Zero:
            weight = math::Rmatrix(outputsize, inputsize, 0);
            break;
        }

        using BIT = BiasInitType;
        switch (bit)
        {
        case BIT::SmallPositive:
            bias = math::Rmatrix(outputsize, 1, 0.01);
            break;
        case BIT::Zero:
            bias = math::Rmatrix(outputsize, 1, 0);
            break;
        }
    }

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
        return (weight * input).plusEachColumn(bias);
    }

    math::Rmatrix AffineLayer::backward(const math::Rmatrix &gradOutput)
    {
        assert(gradOutput.noRows() == outputsize);

        size_t batchsize = gradOutput.noColumns();

        gradWeight = gradOutput * (lastinput.transpose() / batchsize);
        gradBias = (gradOutput / batchsize).rowwiseSum(false);

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