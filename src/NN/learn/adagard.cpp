#include "adagard.hpp"

#include <cmath>

namespace MCL::NN::Engines
{
    AdaGardEngine::AdaGardEngine(math::Real learningRate) : initialized(false), learningRate(learningRate) {}
    AdaGardEngine::AdaGardEngine(bool initialized, math::Real learningRate, math::Real delta, std::vector<std::vector<math::Rmatrix>> h)
        : initialized(initialized), learningRate(learningRate), delta(delta), h(h) {}
    AdaGardEngine::AdaGardEngine(const AdaGardEngine &e)
        : initialized(e.initialized), learningRate(e.learningRate), delta(e.delta), h(e.h) {}

    void AdaGardEngine::reset()
    {
        initialized = false;
    }

    void AdaGardEngine::setLearningRate(math::Real rate)
    {
        learningRate = rate;
    }

    void AdaGardEngine::setDelta(math::Real _delta)
    {
        delta = _delta;
    }

    void AdaGardEngine::initialize(std::vector<std::unique_ptr<Layer>> &layers)
    {
        size_t i, nolayers = layers.size(), j, noparams;
        std::vector<math::Rmatrix *> params;
        h = std::vector<std::vector<math::Rmatrix>>(nolayers);
        for (i = 0; i < nolayers; ++i)
        {
            params = layers[i]->getParameterRefs();
            noparams = params.size();
            h[i] = std::vector<math::Rmatrix>(noparams);
            for (j = 0; j < noparams; ++j)
            {
                h[i][j] = math::Rmatrix(params[j]->noRows(), params[j]->noColumns(), 0);
            }
        }

        initialized = true;
    }

    void AdaGardEngine::run(std::vector<std::unique_ptr<Layer>> &layers)
    {
        if (!initialized)
            initialize(layers);

        size_t i, nolayers = layers.size(), j, noparams;
        std::vector<math::Rmatrix *> params;
        std::vector<math::Rmatrix> grads;
        for (i = 0; i < nolayers; ++i)
        {
            params = layers[i]->getParameterRefs();
            grads = layers[i]->getGradients();

            noparams = params.size();
            for (j = 0; j < noparams; ++j)
            {
                h[i][j] += grads[j].hadamardProd(grads[j]);
                *params[j] -= learningRate * grads[j].hadamardDiv(h[i][j].map<math::Real>([](math::Real x)
                                                                                          { return std::sqrt(x); }),
                                                                  delta);
            }
        }
    }

    std::unique_ptr<LearningEngine> AdaGardEngine::copy() const
    {
        return std::unique_ptr<LearningEngine>(new AdaGardEngine(*this));
    }
}