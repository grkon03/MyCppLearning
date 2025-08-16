#include "neuralnetwork.hpp"

namespace MCL::NN
{
    namespace
    {
        bool isGoodLayers(std::vector<Layer *> layers)
        {
            size_t i = 0, len = layers.size();

            if (len == 0)
                return false;
            else if (layers[0] == nullptr)
                return false;

            size_t prevOutputSize = layers[0]->outputSize();
            Layer *layer;

            for (i = 1; i < len; ++i)
            {
                layer = layers[i];
                if (layer == nullptr)
                    return false;
                if (layer->inputSize() == prevOutputSize)
                    prevOutputSize = layer->outputSize();
                else
                    return false;
            }

            return true;
        }
    }

    NeuralNetwork::NeuralNetwork() : layers({}), noLayers(0), lastlayer(nullptr), inputsize(0) {}
    NeuralNetwork::NeuralNetwork(std::vector<Layer *> layers, LastLayer *lastlayer)
        : layers(layers), noLayers(layers.size()), lastlayer(lastlayer)
    {
        assert(isGoodLayers(layers));
        inputsize = layers[0]->inputSize();
    }

    void NeuralNetwork::addLayer(Layer *layer)
    {
        assert(layer != nullptr);
        layers.push_back(layer);
        ++noLayers;
    }

    void NeuralNetwork::setLastLayer(LastLayer *_lastlayer)
    {
        assert(_lastlayer != nullptr);
        lastlayer = _lastlayer;
    }

    bool NeuralNetwork::isPrepared() const
    {
        return lastlayer != nullptr && layers.size() != 0;
    }

    math::Rmatrix NeuralNetwork::predict(math::Rmatrix firstinput)
    {
        assert(isPrepared());

        size_t i;
        math::Rmatrix input = firstinput;

        for (i = 0; i < noLayers; ++i)
        {
            input = layers[i]->forward(input);
        }

        lastlayer->forward(input);

        return lastlayer->prediction();
    }

    void NeuralNetwork::learn(LearningEngine *engine, math::Rmatrix correctAnswer)
    {
        size_t i = 0;
        math::Rmatrix gradOutput = lastlayer->backwardWithCorrectAnswer(correctAnswer);

        for (i = 1; i <= noLayers; ++i)
        {
            gradOutput = layers[noLayers - i]->backward(gradOutput);
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
}