#pragma once

#include "layers/layer.hpp"
#include "layers/lastlayer.hpp"
#include "learn/engine.hpp"

namespace MCL::NN
{
    class NauralNetwork
    {
    private:
        std::vector<Layer *> layers;
        size_t inputsize;
        size_t noLayers;

        LastLayer *lastlayer;

    public:
        // constructors

        NauralNetwork(std::vector<Layer *>, LastLayer *);

        // basic methods

        bool isPrepared() const;

        math::Rmatrix predict(math::Rmatrix firstinput);
        void learn(LearningEngine *);
    };
}