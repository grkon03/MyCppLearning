#include "random.hpp"

namespace MCL::RL::util
{
    std::vector<math::Real> uniformDirichletSample(math::Real uniformAlpha, size_t size, std::mt19937 &rndgen)
    {
        std::vector<math::Real> sample(size);
        std::gamma_distribution<double> gamma(uniformAlpha, 1.0);

        double sum = 0;
        size_t i;

        for (i = 0; i < size; ++i)
        {
            sum += (sample[i] = gamma(rndgen));
        }

        for (i = 0; i < size; ++i)
        {
            sample[i] /= sum;
        }

        return sample;
    }
}