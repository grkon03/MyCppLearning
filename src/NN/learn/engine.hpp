#pragma once

#include "../layers/layer.hpp"

namespace MCL::NN
{
    class LearningEngine
    {
    public:
        virtual ~LearningEngine() = default;

        virtual void run(std::vector<std::unique_ptr<Layer>> &layers) = 0;
        virtual std::unique_ptr<LearningEngine> copy() const = 0;
    };
}