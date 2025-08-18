#include "gd.hpp"

namespace MCL::NN
{
    GradientDescentEngine::GradientDescentEngine(math::Real rate) : rate(rate) {}

    void GradientDescentEngine::run(std::vector<Layer *> layers)
    {
        std::vector<math::Rmatrix *> paramrefs;
        std::vector<math::Rmatrix> grads;

        size_t i, len;

        for (auto layer : layers)
        {
            paramrefs = layer->getParameterRefs();
            grads = layer->getGradients();

            assert(paramrefs.size() == grads.size());

            len = paramrefs.size();

            for (i = 0; i < len; ++i)
            {
                *(paramrefs[i]) -= rate * grads[i];
            }
        }
    }

    GradientDescentEngine *GradientDescentEngine::copy() const
    {
        return new GradientDescentEngine(rate);
    }
}