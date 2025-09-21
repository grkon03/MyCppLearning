#include <mcl.hpp>
#include <iostream>
#include "mnist.hpp"

using namespace MCL;

math::Rmatrix *DataToVector(MCLSamples::mnist::Dataset dataset)
{
    math::Rmatrix *data;

    data = new math::Rmatrix[dataset.noImages];
    size_t i, j;
    for (i = 0; i < dataset.noImages; ++i)
    {
        data[i] = math::Rmatrix(784, 1);
        for (j = 0; j < 784; ++j)
        {
            data[i].direct(j) = (double)dataset.images[i].data[j] / 255;
        }
    }

    return data;
}

math::Rmatrix *CorrectAnswerToVector(MCLSamples::mnist::Dataset dataset)
{
    math::Rmatrix *ans;
    ans = new math::Rmatrix[dataset.noImages];

    math::Rmatrix label[10] =
        {
            {{1, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
            {{0, 1, 0, 0, 0, 0, 0, 0, 0, 0}},
            {{0, 0, 1, 0, 0, 0, 0, 0, 0, 0}},
            {{0, 0, 0, 1, 0, 0, 0, 0, 0, 0}},
            {{0, 0, 0, 0, 1, 0, 0, 0, 0, 0}},
            {{0, 0, 0, 0, 0, 1, 0, 0, 0, 0}},
            {{0, 0, 0, 0, 0, 0, 1, 0, 0, 0}},
            {{0, 0, 0, 0, 0, 0, 0, 1, 0, 0}},
            {{0, 0, 0, 0, 0, 0, 0, 0, 1, 0}},
            {{0, 0, 0, 0, 0, 0, 0, 0, 0, 1}},
        };

    size_t i;

    for (i = 0; i < 10; ++i)
    {
        label[i] = label[i].transpose();
    }

    for (i = 0; i < dataset.noImages; ++i)
    {
        ans[i] = label[dataset.images[i].label];
    }

    return ans;
}

math::Real correctnessCalc(math::Rmatrix a, math::Rmatrix b)
{
    assert(a.isVVector(10));
    assert(b.isVVector(10));
    size_t i, max_a_i = 0, max_b_i = 0;
    math::Real max_a = std::numeric_limits<int>::min(), max_b = std::numeric_limits<int>::min();

    for (i = 0; i < 10; ++i)
    {
        if (a.at(i, 0) > max_a)
        {
            max_a = a.at(i, 0);
            max_a_i = i;
        }
        if (b.at(i, 0) > max_b)
        {
            max_b = b.at(i, 0);
            max_b_i = i;
        }
    }

    return max_a_i == max_b_i ? 1 : 0;
}

int main()
{
    // make NN

    NN::NeuralNetwork nn;

    using WIT = NN::Layers::AffineLayer::WeightInitType;
    using BIT = NN::Layers::AffineLayer::BiasInitType;
    nn.addLayer(new NN::Layers::AffineLayer(784, 100, WIT::He, BIT::SmallPositive));
    nn.addLayer(new NN::Layers::ReLULayer(100));
    nn.addLayer(new NN::Layers::AffineLayer(100, 50, WIT::He, BIT::SmallPositive));
    nn.addLayer(new NN::Layers::ReLULayer(50));
    nn.addLayer(new NN::Layers::AffineLayer(50, 10, WIT::He, BIT::Zero));
    nn.setLastLayer(new NN::Layers::SoftmaxLastLayer(10));

    // load data

    auto train = MCLSamples::mnist::loadTrain();
    auto traindata = DataToVector(train);
    auto trainans = CorrectAnswerToVector(train);

    auto test = MCLSamples::mnist::loadTest();
    auto testdata = DataToVector(test);
    auto testans = CorrectAnswerToVector(test);

    auto engine = NN::Engines::GradientDescentEngine(0.1);

    std::cout << "accuracy before train: " << nn.accuracy(testdata, testans, test.noImages, correctnessCalc) << std::endl;
    nn.trainMinibatch(&engine, traindata, trainans, train.noImages, 100, 10);
    std::cout << "accuracy after train1: " << nn.accuracy(testdata, testans, test.noImages, correctnessCalc) << std::endl;
}