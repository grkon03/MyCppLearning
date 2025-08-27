#include "engine.hpp"

namespace MCL::NN::Engines
{
    class GradientDescentEngine : public LearningEngine
    {
    private:
        math::Real rate;

    public:
        // constructors

        GradientDescentEngine(math::Real rate);
        GradientDescentEngine(const GradientDescentEngine &);

        // basic methods

        void run(std::vector<std::unique_ptr<Layer>> &layers) override;
        std::unique_ptr<LearningEngine> copy() const override;

        void setRate(math::Real);
    };
}