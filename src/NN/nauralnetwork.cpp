#include "nauralnetwork.hpp"

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

    NauralNetwork::NauralNetwork(std::vector<Layer *> layers, LastLayer *lastlayer)
        : layers(layers), noLayers(layers.size()), lastlayer(lastlayer)
    {
        assert(isGoodLayers(layers));
        inputsize = layers[0]->inputSize();
    }

    bool NauralNetwork::isPrepared() const
    {
        return lastlayer != nullptr && layers.size() != 0;
    }

    math::Rmatrix NauralNetwork::predict(math::Rmatrix firstinput)
    {
        assert(isPrepared());

        size_t i;
        math::Rmatrix input = firstinput;

        for (i = 0; i < noLayers; ++i)
        {
            input = layers[i]->forward(input);
        }

        lastlayer->forward(input);
    }

    void NauralNetwork::learn(LearningEngine *engine)
    {
        size_t i = 0;
        math::Rmatrix gradOutput = lastlayer->backward(math::Rmatrix(1.0));

        for (i = noLayers - 1; i >= 0; --i)
        {
            gradOutput = layers[i]->backward(gradOutput);
        }

        engine->run(layers);
    }
}