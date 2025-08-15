#include "engine.hpp"

namespace MCL::NN
{
    class GradientDescentEngine : public LearningEngine
    {
    private:
        math::Real rate;

    public:
        // constructors

        GradientDescentEngine(math::Real rate);

        // basic methods

        void run(std::vector<Layer *> layers);
    };
}