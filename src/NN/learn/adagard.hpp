#pragma once

#include "engine.hpp"

namespace MCL::NN
{
    class AdaGardEngine : public LearningEngine
    {
    private:
        bool initialized;
        math::Real delta = 1e-7;
        math::Real learningRate;
        std::vector<std::vector<math::Rmatrix>> h;

        void initialize(std::vector<std::unique_ptr<Layer>> &);

    public:
        AdaGardEngine(math::Real learningRate);
        AdaGardEngine(bool initialized, math::Real learningRate, math::Real delta, std::vector<std::vector<math::Rmatrix>> h);
        AdaGardEngine(const AdaGardEngine &);

        virtual void run(std::vector<std::unique_ptr<Layer>> &layers) override;
        virtual std::unique_ptr<LearningEngine> copy() const override;

        virtual void reset();
        virtual void setLearningRate(math::Real);
        void setDelta(math::Real);
    };
}