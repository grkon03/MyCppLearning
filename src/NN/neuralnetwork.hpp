#pragma once

#include "layers/layer.hpp"
#include "layers/lastlayer.hpp"
#include "learn/engine.hpp"

namespace MCL::NN
{
    class NeuralNetwork
    {
    private:
        std::vector<std::unique_ptr<Layer>> layers;
        std::unique_ptr<LastLayer> lastlayer;

    public:
        // constructors

        NeuralNetwork();
        NeuralNetwork(const NeuralNetwork &);

        // set up

        void addLayer(const Layer *);
        void setLastLayer(LastLayer *);

        // basic methods

        bool isPrepared() const;
        size_t inputSize() const;
        size_t outputSize() const;
        size_t noLayers() const;

        math::Rmatrix predict(const math::Rmatrix &firstinput);
        void learn(LearningEngine *, const math::Rmatrix &);
        void learn(LearningEngine *engine, const math::Rmatrix &firstinput, const math::Rmatrix &compare);
        math::Real loss() const;

        void train(LearningEngine *engine, const math::Rmatrix inputs[], const math::Rmatrix compares[], size_t size);
        void trainMinibatch(LearningEngine *engine, const math::Rmatrix inputs[], const math::Rmatrix compares[], size_t size,
                            size_t batchsize, size_t epochsToTrain);

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

        void saveParameters(std::string filename);

        NeuralNetwork &operator=(const NeuralNetwork &);
    };
}