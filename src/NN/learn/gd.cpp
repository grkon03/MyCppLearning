#include "gd.hpp"

namespace MCL::NN
{
    GradientDescentEngine::GradientDescentEngine(math::Real rate) : rate(rate) {}
    GradientDescentEngine::GradientDescentEngine(const GradientDescentEngine &e) : rate(e.rate) {}

    void GradientDescentEngine::run(std::vector<std::unique_ptr<Layer>> &layers)
    {
        std::vector<math::Rmatrix *> paramrefs;
        std::vector<math::Rmatrix> grads;

        size_t i, j, len;

        for (i = 0; i < layers.size(); ++i)
        {
            paramrefs = layers[i]->getParameterRefs();
            grads = layers[i]->getGradients();

            assert(paramrefs.size() == grads.size());

            len = paramrefs.size();

            for (j = 0; j < len; ++j)
            {
                *(paramrefs[j]) -= rate * grads[j];
            }
        }
    }

    std::unique_ptr<LearningEngine> GradientDescentEngine::copy() const
    {
        return std::unique_ptr<LearningEngine>(new GradientDescentEngine(*this));
    }

    void GradientDescentEngine::setRate(math::Real _rate)
    {
        rate = _rate;
    }
}