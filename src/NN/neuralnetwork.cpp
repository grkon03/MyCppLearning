#include "neuralnetwork.hpp"
#include <fstream>

namespace MCL::NN
{
    NeuralNetwork::NeuralNetwork() : layers(), lastlayer(nullptr) {}
    NeuralNetwork::NeuralNetwork(const NeuralNetwork &nn)
    {
        lastlayer = std::move(nn.lastlayer->copy());
        size_t i, nolayers = nn.layers.size();
        layers.clear();
        layers.resize(nolayers);

        for (i = 0; i < nolayers; ++i)
        {
            layers[i] = std::move(nn.layers[i]->copy());
        }
    }

    void NeuralNetwork::addLayer(const Layer *layer)
    {
        assert(layer != nullptr);
        layers.push_back(std::move(layer->copy()));
    }

    void NeuralNetwork::setLastLayer(LastLayer *_lastlayer)
    {
        assert(_lastlayer != nullptr);
        lastlayer = _lastlayer->copy();
    }

    bool NeuralNetwork::isPrepared() const
    {
        return lastlayer != nullptr && layers.size() != 0;
    }

    size_t NeuralNetwork::inputSize() const
    {
        if (layers.size() == 0)
            return 0;
        return layers[0]->inputSize();
    }

    size_t NeuralNetwork::outputSize() const
    {
        if (lastlayer == nullptr)
            return 0;
        return lastlayer->outputSize();
    }

    size_t NeuralNetwork::noLayers() const
    {
        return layers.size();
    }

    math::Real NeuralNetwork::loss() const
    {
        return lastlayer->loss();
    }

    math::Rmatrix NeuralNetwork::predict(math::Rmatrix firstinput)
    {
        assert(isPrepared());

        size_t i;
        math::Rmatrix input = firstinput;

        for (i = 0; i < layers.size(); ++i)
        {
            input = layers[i]->forward(input);
        }

        lastlayer->forward(input);

        return lastlayer->prediction();
    }

    void NeuralNetwork::learn(LearningEngine *engine, math::Rmatrix correctAnswer)
    {
        size_t i = 0;
        math::Rmatrix gradOutput = lastlayer->backwardByComparing(correctAnswer);

        for (i = 1; i <= layers.size(); ++i)
        {
            gradOutput = layers[layers.size() - i]->backward(gradOutput);
        }

        engine->run(layers);
    }

    void NeuralNetwork::train(LearningEngine *engine, math::Rmatrix inputs[], math::Rmatrix correctAnswers[], size_t size)
    {
        assert(engine != nullptr);
        size_t i;
        for (i = 0; i < size; ++i)
        {
            this->predict(inputs[i]);
            this->learn(engine, correctAnswers[i]);
        }
    }

    math::Real NeuralNetwork::accuracy(
        math::Rmatrix testinputs[],
        math::Rmatrix correctAnswers[],
        size_t size,
        std::function<math::Real(math::Rmatrix, math::Rmatrix)> correctnessCalculator)
    {
        size_t i;
        math::Real sum = 0;
        for (i = 0; i < size; ++i)
        {
            sum += correctnessCalculator(this->predict(testinputs[i]), correctAnswers[i]);
        }

        return sum / size;
    }

    void NeuralNetwork::saveParameters(std::string filename)
    {
        std::ofstream ofs(filename, std::ios::binary | std::ios::trunc);
        if (!ofs)
        {
            std::cout << "fail to open the file: " << filename << std::endl;
            return;
        }
        size_t i, nolayers = layers.size(), j, k;
        for (i = 0; i < nolayers; ++i)
        {
            auto params = layers[i]->getParameterRefs();
            for (j = 0; j < params.size(); ++j)
            {
                auto param = params[j];
                for (k = 0; k < param->noRows() * param->noColumns(); ++k)
                    ofs.write(reinterpret_cast<const char *>(&param->direct(k)), sizeof(param->direct(k)));
            }
        }

        ofs.close();
    }

    NeuralNetwork &NeuralNetwork::operator=(const NeuralNetwork &nn)
    {
        size_t i, nolayers = nn.layers.size();
        layers = std::vector<std::unique_ptr<Layer>>(nolayers);
        for (i = 0; i < nolayers; ++i)
        {
            layers[i] = std::move(nn.layers[i]->copy());
        }
        lastlayer = lastlayer->copy();

        return *this;
    }
}