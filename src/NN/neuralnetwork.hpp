#pragma once

#include "layers/layer.hpp"
#include "layers/lastlayer.hpp"
#include "learn/engine.hpp"

namespace MCL::NN
{
    class NeuralNetwork
    {
    private:
        std::vector<Layer *> layers;
        size_t inputsize;
        size_t noLayers;

        LastLayer *lastlayer;

    public:
        // constructors

        NeuralNetwork();
        NeuralNetwork(std::vector<Layer *>, LastLayer *);

        // set up

        void addLayer(Layer *);
        void setLastLayer(LastLayer *);

        // basic methods

        bool isPrepared() const;
        NeuralNetwork *copy() const;
        size_t inputSize() const;
        size_t outputSize() const;

        math::Rmatrix predict(math::Rmatrix firstinput);
        void learn(LearningEngine *, math::Rmatrix);
        math::Real loss() const;

        void train(LearningEngine *engine, math::Rmatrix inputs[], math::Rmatrix correctAnswers[], size_t size);
        /**
         * @brief
         *
         * @param testinputs
         * @param correctAnswers
         * @param size
         * @param correctnessCalculator
         * e.g.) if the problem is a classifying, then correctnessCalculator should be like this:
         * if argmax predict == argmax correctAnswer then it returns 1, otherwise returns 0.
         * @return math::Real
         */
        math::Real accuracy(math::Rmatrix testinputs[], math::Rmatrix correctAnswers[], size_t size,
                            std::function<math::Real(math::Rmatrix, math::Rmatrix)> correctnessCalculator);
    };
}